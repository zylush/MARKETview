from __future__ import annotations


class MarketDataError(Exception):
    code = "market_data_error"
    status_code = 500


class InputValidationError(MarketDataError, ValueError):
    code = "validation_error"
    status_code = 422


class ProviderError(MarketDataError):
    code = "provider_error"
    status_code = 502
    upstream_status: int | None = None
    semantic_code = "unknown"


class ProviderAuthenticationError(ProviderError):
    code = "provider_authentication_error"


class ProviderAccessRestrictedError(ProviderError):
    """The provider credentials are valid but the requested capability is unavailable."""

    code = "provider_access_restricted"


class ProviderRateLimitError(ProviderError):
    code = "provider_rate_limit"
    status_code = 429


class ProviderNotFoundError(ProviderError):
    code = "not_found"
    status_code = 404


class ProviderValidationError(ProviderError):
    code = "provider_validation_error"
    status_code = 422


class ProviderUnavailableError(ProviderError):
    code = "provider_unavailable"
    status_code = 502


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout"
    status_code = 504


class CacheError(MarketDataError):
    code = "cache_error"
    status_code = 503


class CacheUnavailableError(CacheError):
    code = "cache_unavailable"


class QuotaExceededError(MarketDataError):
    code = "quota_exceeded"
    status_code = 429
