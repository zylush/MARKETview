from __future__ import annotations

import ipaddress
import os
from collections.abc import Awaitable
from typing import Any, Protocol, runtime_checkable

from fastapi import Request


@runtime_checkable
class MarketDataServiceProtocol(Protocol):
    async def tickers(self, **params: Any) -> Any: ...

    async def exchanges(self, **params: Any) -> Any: ...

    async def latest_eod(self, symbol: str) -> Any: ...

    async def history(self, symbol: str, start: Any, end: Any, **params: Any) -> Any: ...

    async def splits(self, **params: Any) -> Any: ...

    async def dividends(self, **params: Any) -> Any: ...

    async def usage(self) -> Any: ...


@runtime_checkable
class RateLimitCacheProtocol(Protocol):
    def increment_rate(self, key: str, *, window_seconds: int) -> Awaitable[int]: ...


def setting(settings: object, name: str, default: Any = None, *aliases: str) -> Any:
    """Read a setting from typed settings while tolerating conventional aliases."""
    for candidate in (name, *aliases, name.upper(), *(alias.upper() for alias in aliases)):
        if hasattr(settings, candidate):
            return getattr(settings, candidate)
    return default


def rate_limit_identity(request: Request, settings: object) -> str:
    platform = str(setting(settings, "deployment_platform", "", "platform")).lower()
    trust_forwarded = bool(setting(settings, "trust_proxy_headers", False, "is_vercel"))
    trust_forwarded = (
        trust_forwarded
        or platform in {"vercel", "deployed"}
        or os.getenv("VERCEL", "").lower() in {"1", "true"}
    )
    if trust_forwarded:
        for header in ("X-Vercel-Forwarded-For", "X-Forwarded-For"):
            for value in request.headers.get(header, "").split(","):
                candidate = value.strip()
                try:
                    return ipaddress.ip_address(candidate).compressed
                except ValueError:
                    continue
    return request.client.host if request.client else "unknown"


async def increment_rate(cache: object | None, key: str, window_seconds: int) -> int:
    """Increment a rate-limit bucket, supporting the shared cache and simple test fakes."""
    if cache is None:
        raise RuntimeError("rate limiter unavailable")
    method = getattr(cache, "increment_rate", None)
    if method is not None:
        return int(await method(key, window_seconds=window_seconds))
    fallback = getattr(cache, "increment", None)
    if fallback is None:
        raise RuntimeError("rate limiter unavailable")
    return int(await fallback(key, ttl=window_seconds))
