from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Never
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
from app.models import EODBar, Page
from app.validation import validate_cursor, validate_date_range, validate_limit, validate_symbol

_SUPPRESS_HTTP_CLIENT_LOGS = ContextVar("suppress_marketdata_http_logs", default=False)
_SUPPRESSED_HTTP_LOG_MESSAGE = "HTTP client request details suppressed"
_CANDLE_FIELDS = ("o", "h", "l", "c", "v", "t")
_PLACEHOLDER_TOKENS = frozenset(
    {
        "<your-marketdata-token>",
        "your-marketdata-token",
        "your-api-key",
        "example",
        "change-me",
        "changeme",
        "replace-me",
        "replace-with-your-key",
    }
)
_SAFE_INPUT_MESSAGES = frozenset(
    {
        "cursor must be a non-negative numeric offset",
        "date range cannot exceed one year",
        "date range values must be dates",
        "limit must be between 1 and 1000",
        "start date must not be after end date",
        "symbol must contain 1-32 market identifier characters",
    }
)


def _is_http_client_logger(name: str) -> bool:
    return name in {"httpx", "httpcore"} or name.startswith(("httpx.", "httpcore."))


class _HttpClientLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (_is_http_client_logger(record.name) and _SUPPRESS_HTTP_CLIENT_LOGS.get())


class _SafeLogRecordFactory:
    def __init__(self, delegate: Callable[..., logging.LogRecord]) -> None:
        self._delegate = delegate

    def __call__(self, *args: Any, **kwargs: Any) -> logging.LogRecord:
        record = self._delegate(*args, **kwargs)
        if _is_http_client_logger(record.name) and _SUPPRESS_HTTP_CLIENT_LOGS.get():
            record.msg = _SUPPRESSED_HTTP_LOG_MESSAGE
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return record


def _install_http_client_log_filter() -> None:
    logger_names = {"httpx", "httpcore"}
    logger_names.update(
        name for name in logging.root.manager.loggerDict if name.startswith(("httpx.", "httpcore."))
    )
    for name in logger_names:
        logger = logging.getLogger(name)
        if not any(isinstance(item, _HttpClientLogFilter) for item in logger.filters):
            logger.addFilter(_HttpClientLogFilter())
    handlers = list(logging.getLogger().handlers)
    handlers.extend(
        handler
        for logger in logging.root.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
        for handler in logger.handlers
    )
    for handler in handlers:
        if not any(isinstance(item, _HttpClientLogFilter) for item in handler.filters):
            handler.addFilter(_HttpClientLogFilter())
    record_factory = logging.getLogRecordFactory()
    if not isinstance(record_factory, _SafeLogRecordFactory):
        logging.setLogRecordFactory(_SafeLogRecordFactory(record_factory))


class MarketDataAppProvider:
    """No-retry adapter for Market Data's daily stock-candle API."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.marketdata.app/v1",
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_token = self._normalized_token_or_none(token)
        normalized_base_url = self._normalized_base_url_or_none(base_url)
        token = ""
        base_url = ""
        if normalized_token is None:
            normalized_base_url = ""
            raise ValueError(
                "Market Data token is required and must not be a placeholder"
            ) from None
        if normalized_base_url is None:
            normalized_token = ""
            raise ValueError("Market Data base URL must target a v1 API root") from None
        _install_http_client_log_filter()
        self._token = SecretStr(normalized_token)
        self._base_url = normalized_base_url
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False)
        self._owns_client = client is None

    async def __aenter__(self) -> MarketDataAppProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def latest_eod(self, symbol: str) -> EODBar:
        detached_error: MarketDataError | None = None
        result: tuple[EODBar, ...] | None = None
        try:
            checked_symbol = validate_symbol(symbol)
            result, detached_error = await self._request_candles(
                checked_symbol,
                {"to": "today", "countback": "1", "adjustsplits": "false"},
            )
        except MarketDataError as error:
            detached_error = self._detached_public_error(error)
        except Exception:
            detached_error = InputValidationError("invalid market data request")
        symbol = ""
        checked_symbol = ""
        if detached_error is not None:
            return self._raise_public_error(detached_error)
        if not result:
            return self._raise_public_error(
                ProviderUnavailableError("market data provider returned an invalid response")
            )
        return result[-1]

    async def eod_history(
        self,
        symbol: str,
        *,
        start_date: date | str,
        end_date: date | str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[EODBar]:
        detached_error: MarketDataError | None = None
        rows: tuple[EODBar, ...] | None = None
        checked_limit = 0
        offset = 0
        try:
            checked_symbol = validate_symbol(symbol)
            checked_start, checked_end = validate_date_range(
                self._as_date(start_date), self._as_date(end_date)
            )
            checked_limit = validate_limit(limit)
            checked_cursor = validate_cursor(cursor)
            offset = int(checked_cursor or 0)
            rows, detached_error = await self._request_candles(
                checked_symbol,
                {
                    "from": checked_start.isoformat(),
                    "to": checked_end.isoformat(),
                    "adjustsplits": "false",
                },
            )
        except MarketDataError as error:
            detached_error = self._detached_public_error(error)
        except Exception:
            detached_error = InputValidationError("invalid market data request")
        symbol = ""
        start_date = date.min
        end_date = date.min
        cursor = None
        checked_symbol = ""
        checked_start = date.min
        checked_end = date.min
        checked_cursor = None
        if detached_error is not None:
            rows = None
            limit = 0
            checked_limit = 0
            offset = 0
            return self._raise_public_error(detached_error)
        if rows is None:
            return self._raise_public_error(
                ProviderUnavailableError("market data provider returned an invalid response")
            )
        page_items = rows[offset : offset + checked_limit]
        next_offset = offset + len(page_items)
        return Page(
            items=page_items,
            next_cursor=str(next_offset) if next_offset < len(rows) else None,
            total=len(rows),
        )

    async def _request_candles(
        self, symbol: str, query: Mapping[str, str]
    ) -> tuple[tuple[EODBar, ...] | None, ProviderError | None]:
        payload, status, provider_error = await self._send(
            f"/stocks/candles/D/{symbol}/", dict(query)
        )
        if provider_error is not None:
            return None, self._detached_provider_error(provider_error)
        if payload is None or status is None:
            return None, self._invalid_response_error(status)
        rows, normalization_error = self._normalize_candles(payload, symbol, status)
        if normalization_error is not None:
            return None, self._detached_provider_error(normalization_error)
        return rows, None

    async def _send(
        self, path: str, query: dict[str, str]
    ) -> tuple[Mapping[str, Any] | None, int | None, ProviderError | None]:
        headers = {
            "Authorization": f"Bearer {self._token.get_secret_value()}",
            "Accept": "application/json",
        }
        log_token = _SUPPRESS_HTTP_CLIENT_LOGS.set(True)
        try:
            response = await self._client.get(
                f"{self._base_url}{path}",
                params={**query},
                headers=headers,
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            return None, None, self._transport_error(timeout=True)
        except httpx.HTTPError:
            return None, None, self._transport_error(timeout=False)
        except Exception:
            return None, None, self._transport_error(timeout=False)
        else:
            return self._decode_response(response)
        finally:
            headers = {}
            query = {}
            path = ""
            _SUPPRESS_HTTP_CLIENT_LOGS.reset(log_token)

    @classmethod
    def _decode_response(
        cls, response: httpx.Response
    ) -> tuple[Mapping[str, Any] | None, int | None, ProviderError | None]:
        status = response.status_code
        if status == 204:
            return None, status, cls._status_error(status)
        if status not in {200, 203}:
            return None, status, cls._status_error(status)
        try:
            payload = response.json()
        except Exception:
            return None, status, cls._invalid_response_error(status)
        if not isinstance(payload, Mapping):
            return None, status, cls._invalid_response_error(status)
        semantic = payload.get("s")
        if semantic == "no_data":
            return None, status, cls._status_error(404, upstream_status=status)
        if semantic == "error":
            return None, status, cls._semantic_payload_error(status)
        if semantic != "ok":
            return None, status, cls._invalid_response_error(status)
        return payload, status, None

    @classmethod
    def _normalize_candles(
        cls, payload: Mapping[str, Any], symbol: str, status: int
    ) -> tuple[tuple[EODBar, ...] | None, ProviderError | None]:
        try:
            arrays: dict[str, Sequence[Any]] = {}
            for name in _CANDLE_FIELDS:
                value = payload.get(name)
                if not isinstance(value, (list, tuple)):
                    raise ValueError
                arrays[name] = value
            lengths = {len(value) for value in arrays.values()}
            if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
                raise ValueError
            rows = tuple(
                EODBar(
                    symbol=symbol,
                    date=cls._timestamp_date(arrays["t"][index]),
                    open=cls._decimal(arrays["o"][index]),
                    high=cls._decimal(arrays["h"][index]),
                    low=cls._decimal(arrays["l"][index]),
                    close=cls._decimal(arrays["c"][index]),
                    volume=cls._volume(arrays["v"][index]),
                )
                for index in range(next(iter(lengths)))
            )
            return tuple(sorted(rows, key=lambda row: row.date)), None
        except Exception:
            return None, cls._invalid_response_error(status)

    @staticmethod
    def _decimal(value: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
            raise ValueError
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise ValueError from error
        if not result.is_finite():
            raise ValueError
        return result

    @staticmethod
    def _volume(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
            raise ValueError
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise ValueError from error
        if (
            not decimal_value.is_finite()
            or decimal_value < 0
            or decimal_value != int(decimal_value)
        ):
            raise ValueError
        return int(decimal_value)

    @staticmethod
    def _timestamp_date(value: Any) -> date:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError
        if not float(value).is_integer():
            raise ValueError
        try:
            return datetime.fromtimestamp(int(value), tz=UTC).date()
        except (OverflowError, OSError, ValueError) as error:
            raise ValueError from error

    @staticmethod
    def _as_date(value: date | str) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            raise ProviderValidationError("date must use ISO YYYY-MM-DD format") from None

    @staticmethod
    def _transport_error(*, timeout: bool) -> ProviderError:
        if timeout:
            error: ProviderError = ProviderTimeoutError("market data provider timed out")
            error.semantic_code = "timeout"
        else:
            error = ProviderUnavailableError("market data provider is unavailable")
            error.semantic_code = "transport_error"
        return error

    @classmethod
    def _status_error(cls, status: int, *, upstream_status: int | None = None) -> ProviderError:
        if status in {204, 404}:
            error: ProviderError = ProviderNotFoundError("market data was not found")
            semantic_code = "no_data"
        elif status in {400, 413}:
            error = ProviderValidationError("market data provider rejected the request")
            semantic_code = "request_rejected"
        elif status == 401:
            error = ProviderAuthenticationError("market data provider rejected its credentials")
            semantic_code = "authentication_failed"
        elif status == 402:
            error = ProviderAccessRestrictedError(
                "market data provider plan does not permit this request"
            )
            semantic_code = "plan_restricted"
        elif status == 403:
            error = ProviderAccessRestrictedError(
                "market data provider does not permit this request"
            )
            semantic_code = "access_restricted"
        elif status == 429:
            error = ProviderRateLimitError("market data provider request limit was reached")
            semantic_code = "rate_limited"
        elif status in {504, 524}:
            error = ProviderTimeoutError("market data provider timed out")
            semantic_code = "timeout"
        elif status >= 500:
            error = ProviderUnavailableError("market data provider is unavailable")
            semantic_code = "upstream_unavailable"
        else:
            error = ProviderError("market data provider request failed")
            semantic_code = "request_failed"
        error.upstream_status = upstream_status if upstream_status is not None else status
        error.semantic_code = semantic_code
        return error

    @staticmethod
    def _semantic_payload_error(status: int) -> ProviderError:
        error = ProviderError("market data provider request failed")
        error.upstream_status = status
        error.semantic_code = "provider_error"
        return error

    @staticmethod
    def _invalid_response_error(status: int | None) -> ProviderUnavailableError:
        error = ProviderUnavailableError("market data provider returned an invalid response")
        error.upstream_status = status
        error.semantic_code = "invalid_response"
        return error

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
            return InputValidationError(
                message if message in _SAFE_INPUT_MESSAGES else "invalid market data request"
            )
        if isinstance(error, ProviderValidationError) and error.upstream_status is None:
            return ProviderValidationError("date must use ISO YYYY-MM-DD format")
        if isinstance(error, ProviderError):
            return cls._detached_provider_error(error)
        return MarketDataError("market data request failed")

    @staticmethod
    def _raise_public_error(error: MarketDataError) -> Never:
        raise error from None

    @staticmethod
    def _normalized_token_or_none(token: object) -> str | None:
        if not isinstance(token, str):
            return None
        contains_control = any(ord(character) < 32 or ord(character) == 127 for character in token)
        normalized = token.strip()
        is_quoted = (
            len(normalized) >= 2 and normalized[0] in {"'", '"'} and normalized[-1] == normalized[0]
        )
        placeholder = normalized.lower()
        if (
            not normalized
            or contains_control
            or is_quoted
            or placeholder in _PLACEHOLDER_TOKENS
            or (placeholder.startswith("<") and placeholder.endswith(">"))
        ):
            return None
        return normalized

    @staticmethod
    def _normalized_base_url_or_none(base_url: object) -> str | None:
        if not isinstance(base_url, str):
            return None
        normalized = base_url[:-1] if base_url.endswith("/") else base_url
        try:
            parsed = urlsplit(normalized)
            _ = parsed.port
        except ValueError:
            return None
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != "/v1"
            or parsed.query
            or parsed.fragment
        ):
            return None
        return normalized
