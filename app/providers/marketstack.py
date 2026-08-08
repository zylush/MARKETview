from __future__ import annotations

import logging
import re
from collections.abc import Callable
from contextvars import ContextVar
from datetime import date
from decimal import Decimal
from typing import Any, Never, TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from app.errors import (
    InputValidationError,
    MarketDataError,
    ProviderAccessRestrictedError,
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
R = TypeVar("R")
_AUTH_ERROR_CODES = frozenset({"invalid_access_key", "missing_access_key", "inactive_user"})
_ACCESS_RESTRICTED_ERROR_CODES = frozenset(
    {
        "endpoint_access_restricted",
        "function_access_restricted",
        "https_access_restricted",
        "subscription_access_restricted",
    }
)
_RATE_LIMIT_ERROR_CODES = frozenset(
    {"rate_limit_reached", "too_many_requests", "usage_limit_reached"}
)
_INTEGRATION_ERROR_CODES = frozenset({"404_not_found", "internal_error", "invalid_api_function"})
_VALIDATION_ERROR_CODES = frozenset({"validation_error"})
_KNOWN_ERROR_CODES = (
    _AUTH_ERROR_CODES
    | _ACCESS_RESTRICTED_ERROR_CODES
    | _RATE_LIMIT_ERROR_CODES
    | _INTEGRATION_ERROR_CODES
    | _VALIDATION_ERROR_CODES
)
_PLACEHOLDER_KEYS = frozenset(
    {
        "change-me",
        "changeme",
        "example",
        "replace-me",
        "your-api-key",
        "your-marketstack-api-key",
    }
)
_SEMANTIC_CODE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_SUPPRESS_HTTP_CLIENT_LOGS = ContextVar("suppress_marketstack_http_logs", default=False)
_SUPPRESSED_HTTP_LOG_MESSAGE = "HTTP client request details suppressed"
_SAFE_INPUT_VALIDATION_MESSAGES = frozenset(
    {
        "cursor must be a non-negative numeric offset",
        "date range cannot exceed one year",
        "date range values must be dates",
        "limit must be between 1 and 1000",
        "start date must not be after end date",
        "symbol must contain 1-32 market identifier characters",
    }
)
_SAFE_LOCAL_PROVIDER_VALIDATION_MESSAGES = frozenset(
    {
        "date must use ISO YYYY-MM-DD format",
        "start and end dates must be provided together",
    }
)


class _MarketstackHttpLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        is_http_client_log = record.name in {"httpx", "httpcore"} or record.name.startswith(
            ("httpx.", "httpcore.")
        )
        return not (is_http_client_log and _SUPPRESS_HTTP_CLIENT_LOGS.get())


class _MarketstackLogRecordFactory:
    def __init__(self, delegate: Callable[..., logging.LogRecord]) -> None:
        self._delegate = delegate

    def __call__(self, *args: Any, **kwargs: Any) -> logging.LogRecord:
        record = self._delegate(*args, **kwargs)
        is_http_client_log = record.name in {"httpx", "httpcore"} or record.name.startswith(
            ("httpx.", "httpcore.")
        )
        if is_http_client_log and _SUPPRESS_HTTP_CLIENT_LOGS.get():
            record.msg = _SUPPRESSED_HTTP_LOG_MESSAGE
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return record


def _install_http_client_log_filter() -> None:
    """Suppress dependency logs only while a Marketstack request is active."""

    logger_names = {"httpx", "httpcore"}
    logger_names.update(
        name for name in logging.root.manager.loggerDict if name.startswith(("httpx.", "httpcore."))
    )
    for name in logger_names:
        logger = logging.getLogger(name)
        if not any(isinstance(item, _MarketstackHttpLogFilter) for item in logger.filters):
            logger.addFilter(_MarketstackHttpLogFilter())
    handlers = list(logging.getLogger().handlers)
    handlers.extend(
        handler
        for logger in logging.root.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
        for handler in logger.handlers
    )
    for handler in handlers:
        if not any(isinstance(item, _MarketstackHttpLogFilter) for item in handler.filters):
            handler.addFilter(_MarketstackHttpLogFilter())
    record_factory = logging.getLogRecordFactory()
    if not isinstance(record_factory, _MarketstackLogRecordFactory):
        logging.setLogRecordFactory(_MarketstackLogRecordFactory(record_factory))


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
        normalized_access_key = self._normalize_access_key(access_key)
        normalized_base_url = self._normalize_base_url(base_url)
        _install_http_client_log_filter()
        self._access_key = SecretStr(normalized_access_key)
        self._base_url = normalized_base_url
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def __aenter__(self) -> MarketstackProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request_normalized(
        self,
        path: str,
        params: dict[str, Any] | None,
        normalizer: Callable[[dict[str, Any]], tuple[R | None, ProviderError | None]],
    ) -> R:
        result, provider_error = await self._fetch_and_normalize(path, params, normalizer)
        if provider_error is not None:
            raise provider_error from None
        if result is None:
            raise ProviderUnavailableError("market data provider returned an invalid response")
        return result

    async def _fetch_and_normalize(
        self,
        path: str,
        params: dict[str, Any] | None,
        normalizer: Callable[[dict[str, Any]], tuple[R | None, ProviderError | None]],
    ) -> tuple[R | None, ProviderError | None]:
        payload, provider_error = await self._send(path, {**(params or {})})
        if provider_error is not None:
            return None, provider_error
        if payload is None:
            return None, ProviderUnavailableError(
                "market data provider returned an invalid response"
            )
        try:
            return normalizer(payload)
        except ProviderError as error:
            return None, self._detached_provider_error(error)
        except Exception:
            return None, ProviderUnavailableError(
                "market data provider returned an invalid response"
            )

    async def _send(
        self, path: str, query: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, ProviderError | None]:
        request_params = {
            **query,
            "access_key": self._access_key.get_secret_value(),
        }
        log_token = _SUPPRESS_HTTP_CLIENT_LOGS.set(True)
        try:
            response = await self._client.get(f"{self._base_url}{path}", params=request_params)
        except httpx.TimeoutException:
            return None, ProviderTimeoutError("market data provider timed out")
        except httpx.HTTPError:
            return None, ProviderUnavailableError("market data provider is unavailable")
        else:
            return self._decode_response(response)
        finally:
            request_params = {}
            _SUPPRESS_HTTP_CLIENT_LOGS.reset(log_token)

    @classmethod
    def _decode_response(
        cls, response: httpx.Response
    ) -> tuple[dict[str, Any] | None, ProviderError | None]:
        status = response.status_code
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                return None, cls._malformed_response_error(status)
            error = payload.get("error")
            if status >= 400 or error:
                return None, cls._mapped_error(status, error)
            return payload, None
        except Exception:
            return None, cls._malformed_response_error(status)

    @classmethod
    def _malformed_response_error(cls, status: int) -> ProviderError:
        if status >= 400:
            return cls._mapped_error(status, None)
        return cls._invalid_response_error(status)

    @staticmethod
    def _invalid_response_error(status: int) -> ProviderUnavailableError:
        exception = ProviderUnavailableError("market data provider returned an invalid response")
        exception.upstream_status = status
        exception.semantic_code = "unknown"
        return exception

    @classmethod
    def _mapped_error(cls, status: int, error: object) -> ProviderError:
        codes = cls._semantic_error_codes(error)
        code_set = frozenset(codes)
        known_code = next((code for code in codes if code in _KNOWN_ERROR_CODES), "unknown")
        exception = cls._semantic_exception(code_set) or cls._status_exception(status)
        exception.upstream_status = status
        exception.semantic_code = known_code
        return exception

    @staticmethod
    def _semantic_exception(codes: frozenset[str]) -> ProviderError | None:
        if codes & _AUTH_ERROR_CODES:
            return ProviderAuthenticationError("market data provider rejected its credentials")
        if codes & _ACCESS_RESTRICTED_ERROR_CODES:
            return ProviderAccessRestrictedError(
                "market data provider does not permit this request"
            )
        if codes & _RATE_LIMIT_ERROR_CODES:
            return ProviderRateLimitError("market data provider request limit was reached")
        if codes & _INTEGRATION_ERROR_CODES:
            return ProviderUnavailableError("market data provider integration is unavailable")
        if codes & _VALIDATION_ERROR_CODES:
            return ProviderValidationError("market data provider rejected the request")
        return None

    @staticmethod
    def _status_exception(status: int) -> ProviderError:
        if status == 401:
            return ProviderAuthenticationError("market data provider rejected its credentials")
        if status == 403:
            return ProviderAccessRestrictedError(
                "market data provider does not permit this request"
            )
        if status == 429:
            return ProviderRateLimitError("market data provider request limit was reached")
        if status in (400, 404):
            return ProviderUnavailableError("market data provider integration is unavailable")
        if status == 422:
            return ProviderValidationError("market data provider rejected the request")
        if status >= 500:
            return ProviderUnavailableError("market data provider is unavailable")
        return ProviderError("market data provider request failed")

    @staticmethod
    def _semantic_error_codes(error: object) -> tuple[str, ...]:
        if not isinstance(error, dict):
            return ()
        normalized: list[str] = []
        for field in ("code", "type"):
            value = error.get(field)
            if not isinstance(value, str):
                continue
            candidate = value.strip().lower()
            if _SEMANTIC_CODE_PATTERN.fullmatch(candidate):
                normalized.append(candidate)
        return tuple(dict.fromkeys(normalized))

    @staticmethod
    def _detached_provider_error(error: ProviderError) -> ProviderError:
        if isinstance(error, ProviderAuthenticationError):
            detached: ProviderError = ProviderAuthenticationError(
                "market data provider rejected its credentials"
            )
        elif isinstance(error, ProviderAccessRestrictedError):
            detached = ProviderAccessRestrictedError(
                "market data provider does not permit this request"
            )
        elif isinstance(error, ProviderRateLimitError):
            detached = ProviderRateLimitError("market data provider request limit was reached")
        elif isinstance(error, ProviderNotFoundError):
            detached = ProviderNotFoundError("market data was not found")
        elif isinstance(error, ProviderValidationError):
            detached = ProviderValidationError("market data provider rejected the request")
        elif isinstance(error, ProviderTimeoutError):
            detached = ProviderTimeoutError("market data provider timed out")
        elif isinstance(error, ProviderUnavailableError):
            detached = ProviderUnavailableError("market data provider returned an invalid response")
        else:
            detached = ProviderError("market data provider request failed")
        detached.upstream_status = error.upstream_status
        detached.semantic_code = error.semantic_code
        return detached

    @classmethod
    def _detached_public_error(cls, error: MarketDataError) -> MarketDataError:
        if isinstance(error, InputValidationError):
            message = str(error)
            if message not in _SAFE_INPUT_VALIDATION_MESSAGES:
                message = "invalid market data request"
            return InputValidationError(message)
        if (
            isinstance(error, ProviderValidationError)
            and error.upstream_status is None
            and str(error) in _SAFE_LOCAL_PROVIDER_VALIDATION_MESSAGES
        ):
            return ProviderValidationError(str(error))
        if isinstance(error, ProviderError):
            return cls._detached_provider_error(error)
        return MarketDataError("market data request failed")

    @staticmethod
    def _raise_public_error(error: MarketDataError) -> Never:
        raise error from None

    @staticmethod
    def _normalize_access_key(access_key: str) -> str:
        if not isinstance(access_key, str):
            raise ValueError("Marketstack access key is required")
        contains_control_character = any(
            ord(character) < 32 or ord(character) == 127 for character in access_key
        )
        normalized = access_key.strip()
        placeholder = normalized.strip("'\"").strip().lower()
        if (
            not normalized
            or contains_control_character
            or placeholder in _PLACEHOLDER_KEYS
            or (placeholder.startswith("<") and placeholder.endswith(">"))
        ):
            raise ValueError("Marketstack access key is required and must not be a placeholder")
        return normalized

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        parsed = urlsplit(normalized)
        if not parsed.scheme or not parsed.netloc or parsed.path != "/v2":
            raise ValueError("Marketstack base URL must target the v2 API root")
        if parsed.query or parsed.fragment:
            raise ValueError("Marketstack base URL must target the v2 API root")
        return normalized

    @staticmethod
    def _page(
        payload: dict[str, Any], mapper: Callable[[dict[str, Any]], T]
    ) -> tuple[Page[T] | None, ProviderError | None]:
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
            return Page(items=items, next_cursor=next_cursor, total=total), None
        except ProviderError as error:
            return None, MarketstackProvider._detached_provider_error(error)
        except Exception:
            return None, ProviderUnavailableError(
                "market data provider returned an invalid response"
            )

    @staticmethod
    def _pagination(limit: int, cursor: str | None) -> dict[str, Any]:
        checked_cursor = validate_cursor(cursor)
        return {"limit": validate_limit(limit), "offset": int(checked_cursor or 0)}

    async def list_tickers(
        self, *, search: str | None = None, limit: int = 100, cursor: str | None = None
    ) -> Page[Ticker]:
        try:
            params = self._pagination(limit, cursor)
            if search:
                params = {**params, "search": search.strip()[:100]}

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

            return await self._request_normalized(
                "/tickers",
                params,
                lambda payload: self._page(payload, ticker),
            )
        except MarketDataError as error:
            detached_error = self._detached_public_error(error)
        search = None
        limit = 0
        cursor = None
        params = {}
        return self._raise_public_error(detached_error)

    async def list_exchanges(
        self, *, search: str | None = None, limit: int = 100, cursor: str | None = None
    ) -> Page[Exchange]:
        try:
            params = self._pagination(limit, cursor)
            if search:
                params = {**params, "search": search.strip()[:100]}
            return await self._request_normalized(
                "/exchanges",
                params,
                lambda payload: self._page(payload, lambda item: Exchange(**item)),
            )
        except MarketDataError as error:
            detached_error = self._detached_public_error(error)
        search = None
        limit = 0
        cursor = None
        params = {}
        return self._raise_public_error(detached_error)

    async def latest_eod(self, symbol: str) -> EODBar:
        try:
            checked = validate_symbol(symbol)
            page = await self._request_normalized(
                "/eod/latest",
                {"symbols": checked},
                lambda payload: self._page(payload, self._bar),
            )
            if not page.items:
                raise ProviderNotFoundError("market data was not found")
            return page.items[0]
        except MarketDataError as error:
            detached_error = self._detached_public_error(error)
        symbol = ""
        checked = ""
        page = Page[EODBar]()
        return self._raise_public_error(detached_error)

    async def eod_history(
        self,
        symbol: str,
        *,
        start_date: date | str,
        end_date: date | str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[EODBar]:
        try:
            checked_start, checked_end = validate_date_range(
                self._as_date(start_date), self._as_date(end_date)
            )
            params = {
                **self._pagination(limit, cursor),
                "symbols": validate_symbol(symbol),
                "date_from": checked_start.isoformat(),
                "date_to": checked_end.isoformat(),
            }
            return await self._request_normalized(
                "/eod",
                params,
                lambda payload: self._page(payload, self._bar),
            )
        except MarketDataError as error:
            detached_error = self._detached_public_error(error)
        symbol = ""
        start_date = date.min
        end_date = date.min
        limit = 0
        cursor = None
        checked_start = date.min
        checked_end = date.min
        params = {}
        return self._raise_public_error(detached_error)

    async def splits(
        self,
        symbol: str | None = None,
        *,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[Split]:
        try:
            params = self._action_params(symbol, start_date, end_date, limit, cursor)
            return await self._request_normalized(
                "/splits",
                params,
                lambda payload: self._page(
                    payload,
                    lambda item: Split(
                        symbol=item["symbol"],
                        date=item["date"],
                        ratio=self._required_decimal(item, "split_factor", "ratio"),
                    ),
                ),
            )
        except MarketDataError as error:
            detached_error = self._detached_public_error(error)
        symbol = None
        start_date = None
        end_date = None
        limit = 0
        cursor = None
        params = {}
        return self._raise_public_error(detached_error)

    async def dividends(
        self,
        symbol: str | None = None,
        *,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[Dividend]:
        try:
            params = self._action_params(symbol, start_date, end_date, limit, cursor)
            return await self._request_normalized(
                "/dividends",
                params,
                lambda payload: self._page(
                    payload,
                    lambda item: Dividend(
                        symbol=item["symbol"],
                        date=item["date"],
                        amount=self._required_decimal(item, "dividend", "amount"),
                        currency=item.get("currency"),
                    ),
                ),
            )
        except MarketDataError as error:
            detached_error = self._detached_public_error(error)
        symbol = None
        start_date = None
        end_date = None
        limit = 0
        cursor = None
        params = {}
        return self._raise_public_error(detached_error)

    async def usage(self) -> Usage:
        try:
            return await self._request_normalized("/usage", None, self._normalize_usage)
        except MarketDataError as error:
            detached_error = self._detached_public_error(error)
        return self._raise_public_error(detached_error)

    def _normalize_usage(
        self, payload: dict[str, Any]
    ) -> tuple[Usage | None, ProviderError | None]:
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            return None, ProviderUnavailableError(
                "market data provider returned an invalid response"
            )
        try:
            used = int(data.get("requests_used", data.get("requests", data.get("current", 0))) or 0)
            limit_raw = data.get("requests_limit", data.get("limit"))
            remaining_raw = data.get("requests_remaining", data.get("remaining"))
            return (
                Usage(
                    requests_used=used,
                    requests_limit=int(limit_raw) if limit_raw is not None else None,
                    requests_remaining=int(remaining_raw) if remaining_raw is not None else None,
                ),
                None,
            )
        except ProviderError as error:
            return None, self._detached_provider_error(error)
        except Exception:
            return None, ProviderUnavailableError(
                "market data provider returned an invalid response"
            )

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
        return Decimal(str(raw))

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
