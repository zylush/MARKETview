from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from app.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderValidationError,
)
from app.models import Dividend, EODBar, Exchange, Page, Split, Ticker, Usage
from app.validation import validate_cursor, validate_date_range, validate_limit, validate_symbol

T = TypeVar("T")
_SENSITIVE_QUERY_NAMES = frozenset(
    {"access_key", "api_key", "apikey", "password", "secret", "token"}
)


def _redact_query_argument(value: object) -> object:
    if not isinstance(value, (str, httpx.URL)):
        return value
    raw = str(value)
    if "?" not in raw:
        return value
    parsed = urlsplit(raw)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    redacted = [
        (name, "REDACTED" if name.lower() in _SENSITIVE_QUERY_NAMES else item)
        for name, item in pairs
    ]
    if pairs == redacted:
        return value
    query = urlencode(redacted, doseq=True)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))


class _SensitiveQueryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_query_argument(value) for value in record.args)
        return True


def _install_http_log_redaction() -> None:
    httpx_logger = logging.getLogger("httpx")
    if not any(isinstance(item, _SensitiveQueryFilter) for item in httpx_logger.filters):
        httpx_logger.addFilter(_SensitiveQueryFilter())
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class MarketstackProvider:
    """Thin, no-retry adapter from Marketstack v2 payloads to domain models."""

    def __init__(
        self,
        access_key: str,
        *,
        base_url: str = "https://api.marketstack.com/v2",
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not access_key:
            raise ValueError("Marketstack access key is required")
        _install_http_log_redaction()
        self._access_key = access_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def __aenter__(self) -> MarketstackProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = {**(params or {}), "access_key": self._access_key}
        try:
            response = await self._client.get(f"{self._base_url}{path}", params=query)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("market data provider timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError("market data provider is unavailable") from exc
        if response.status_code >= 400:
            self._raise_mapped_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(
                "market data provider returned an invalid response"
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderUnavailableError("market data provider returned an invalid response")
        if payload.get("error"):
            self._raise_payload_error(payload["error"])
        return payload

    def _raise_mapped_error(self, response: httpx.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        code = error.get("code", "") if isinstance(error, dict) else ""
        self._raise_error(response.status_code, str(code))

    def _raise_payload_error(self, error: object) -> None:
        code = error.get("code", "") if isinstance(error, dict) else ""
        self._raise_error(502, str(code))

    @staticmethod
    def _raise_error(status: int, code: str) -> None:
        normalized = code.lower()
        if status in (401, 403) or normalized in {
            "invalid_access_key",
            "missing_access_key",
            "inactive_user",
            "function_access_restricted",
            "https_access_restricted",
        }:
            raise ProviderAuthenticationError("market data provider rejected its credentials")
        if status == 429 or normalized in {"usage_limit_reached", "rate_limit_reached"}:
            raise ProviderRateLimitError("market data provider request limit was reached")
        if status == 404 or normalized in {"404_not_found", "invalid_api_function"}:
            raise ProviderNotFoundError("market data was not found")
        if status in (400, 422) or normalized == "validation_error":
            raise ProviderValidationError("market data provider rejected the request")
        if status >= 500 or normalized == "internal_error":
            raise ProviderUnavailableError("market data provider is unavailable")
        raise ProviderError("market data provider request failed")

    @staticmethod
    def _page(payload: dict[str, Any], mapper: Callable[[dict[str, Any]], T]) -> Page[T]:
        try:
            raw_data = payload.get("data", [])
            data = raw_data if isinstance(raw_data, list) else [raw_data]
            if any(not isinstance(item, dict) for item in data):
                raise ValueError("data items must be objects")
            items = tuple(mapper(item) for item in data)
            pagination = payload.get("pagination", {})
            if not isinstance(pagination, dict):
                raise ValueError("pagination must be an object")
            offset = int(pagination.get("offset", 0) or 0)
            limit = int(pagination.get("limit", len(items)) or len(items))
            count = int(pagination.get("count", len(items)) or 0)
            total_raw = pagination.get("total")
            total = int(total_raw) if total_raw is not None else None
            if offset < 0 or limit < 0 or count < 0 or (total is not None and total < 0):
                raise ValueError("pagination values must not be negative")
            next_cursor = (
                str(offset + limit) if total is not None and offset + count < total else None
            )
            return Page(items=items, next_cursor=next_cursor, total=total)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                "market data provider returned an invalid response"
            ) from exc

    @staticmethod
    def _pagination(limit: int, cursor: str | None) -> dict[str, Any]:
        checked_cursor = validate_cursor(cursor)
        return {"limit": validate_limit(limit), "offset": int(checked_cursor or 0)}

    async def list_tickers(
        self, *, search: str | None = None, limit: int = 100, cursor: str | None = None
    ) -> Page[Ticker]:
        params = self._pagination(limit, cursor)
        if search:
            params = {**params, "search": search.strip()[:100]}
        payload = await self._get("/tickers", params)

        def ticker(item: dict[str, Any]) -> Ticker:
            exchange = item.get("stock_exchange") or {}
            return Ticker(
                symbol=self._required_text(item, "symbol"),
                name=item.get("name"),
                exchange_mic=exchange.get("mic") if isinstance(exchange, dict) else None,
                exchange_name=exchange.get("name") if isinstance(exchange, dict) else None,
                has_intraday=item.get("has_intraday"),
                has_eod=item.get("has_eod"),
            )

        return self._page(payload, ticker)

    async def list_exchanges(
        self, *, search: str | None = None, limit: int = 100, cursor: str | None = None
    ) -> Page[Exchange]:
        params = self._pagination(limit, cursor)
        if search:
            params = {**params, "search": search.strip()[:100]}
        payload = await self._get("/exchanges", params)
        return self._page(payload, lambda item: Exchange(**item))

    async def latest_eod(self, symbol: str) -> EODBar:
        checked = validate_symbol(symbol)
        payload = await self._get("/eod/latest", {"symbols": checked})
        page = self._page(payload, self._bar)
        if not page.items:
            raise ProviderNotFoundError("market data was not found")
        return page.items[0]

    async def eod_history(
        self,
        symbol: str,
        *,
        start_date: date | str,
        end_date: date | str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[EODBar]:
        checked_start, checked_end = validate_date_range(
            self._as_date(start_date), self._as_date(end_date)
        )
        params = {
            **self._pagination(limit, cursor),
            "symbols": validate_symbol(symbol),
            "date_from": checked_start.isoformat(),
            "date_to": checked_end.isoformat(),
        }
        return self._page(await self._get("/eod", params), self._bar)

    async def splits(
        self,
        symbol: str | None = None,
        *,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[Split]:
        params = self._action_params(symbol, start_date, end_date, limit, cursor)
        return self._page(
            await self._get("/splits", params),
            lambda item: Split(
                symbol=item["symbol"],
                date=item["date"],
                ratio=self._required_decimal(item, "split_factor", "ratio"),
            ),
        )

    async def dividends(
        self,
        symbol: str | None = None,
        *,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[Dividend]:
        params = self._action_params(symbol, start_date, end_date, limit, cursor)
        return self._page(
            await self._get("/dividends", params),
            lambda item: Dividend(
                symbol=item["symbol"],
                date=item["date"],
                amount=self._required_decimal(item, "dividend", "amount"),
                currency=item.get("currency"),
            ),
        )

    async def usage(self) -> Usage:
        payload = await self._get("/usage")
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise ProviderUnavailableError("market data provider returned an invalid response")
        try:
            used = int(data.get("requests_used", data.get("requests", data.get("current", 0))) or 0)
            limit_raw = data.get("requests_limit", data.get("limit"))
            remaining_raw = data.get("requests_remaining", data.get("remaining"))
            return Usage(
                requests_used=used,
                requests_limit=int(limit_raw) if limit_raw is not None else None,
                requests_remaining=int(remaining_raw) if remaining_raw is not None else None,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                "market data provider returned an invalid response"
            ) from exc

    @staticmethod
    def _bar(item: dict[str, Any]) -> EODBar:
        return EODBar(
            symbol=MarketstackProvider._required_text(item, "symbol"),
            date=item.get("date", ""),
            open=item.get("open"),
            high=item.get("high"),
            low=item.get("low"),
            close=item.get("close"),
            volume=item.get("volume"),
            adjusted_open=item.get("adjusted_open", item.get("adj_open")),
            adjusted_high=item.get("adjusted_high", item.get("adj_high")),
            adjusted_low=item.get("adjusted_low", item.get("adj_low")),
            adjusted_close=item.get("adjusted_close", item.get("adj_close")),
            adjusted_volume=item.get("adjusted_volume", item.get("adj_volume")),
        )

    @staticmethod
    def _required_decimal(item: dict[str, Any], primary: str, fallback: str) -> Decimal:
        raw = item.get(primary, item.get(fallback))
        try:
            return Decimal(str(raw))
        except (InvalidOperation, ValueError) as exc:
            raise ProviderUnavailableError(
                "market data provider returned an invalid response"
            ) from exc

    @staticmethod
    def _required_text(item: dict[str, Any], name: str) -> str:
        value = item.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ProviderUnavailableError("market data provider returned an invalid response")
        return value

    @staticmethod
    def _as_date(value: date | str) -> date:
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError as exc:
            raise ProviderValidationError("date must use ISO YYYY-MM-DD format") from exc

    def _action_params(
        self,
        symbol: str | None,
        start_date: date | str | None,
        end_date: date | str | None,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        params = self._pagination(limit, cursor)
        if symbol is not None:
            params = {**params, "symbols": validate_symbol(symbol)}
        if (start_date is None) != (end_date is None):
            raise ProviderValidationError("start and end dates must be provided together")
        if start_date is not None and end_date is not None:
            checked_start, checked_end = validate_date_range(
                self._as_date(start_date), self._as_date(end_date)
            )
            params = {
                **params,
                "date_from": checked_start.isoformat(),
                "date_to": checked_end.isoformat(),
            }
        return params
