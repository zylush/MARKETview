from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from app.cache.base import Cache
from app.cache.upstash import UpstashCache
from app.config import Settings, get_settings
from app.providers.sec_symbols import SecSymbolDirectoryProvider
from app.services.symbols import SymbolDirectory, SymbolDirectorySource, SymbolSearchService


async def _close(resource: object) -> None:
    closer = getattr(resource, "aclose", None)
    if closer is not None:
        await cast(Awaitable[None], closer())


def _shared_cache(settings: Settings) -> Cache:
    redis_token = (
        settings.upstash_redis_rest_token.get_secret_value()
        if settings.upstash_redis_rest_token
        else ""
    )
    if not settings.upstash_redis_rest_url or not redis_token:
        raise RuntimeError("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are required")
    return UpstashCache(settings.upstash_redis_rest_url, redis_token)


async def refresh_symbol_directory(
    *,
    settings: Settings,
    cache: Cache | None = None,
    source: SymbolDirectorySource | None = None,
) -> SymbolDirectory:
    """Refresh the SEC symbol index from an explicit control-plane invocation."""

    if not settings.sec_user_agent:
        raise RuntimeError("SEC_USER_AGENT is required for symbol directory refresh")

    owns_cache = cache is None
    active_cache = cache or _shared_cache(settings)

    owns_source = source is None
    active_source = source or SecSymbolDirectoryProvider(
        user_agent=settings.sec_user_agent,
        timeout_seconds=min(settings.http_timeout_seconds, 10.0),
    )
    try:
        service = SymbolSearchService(
            active_source,
            active_cache,
            schema_version=settings.symbol_index_schema_version,
            directory_ttl_seconds=settings.symbol_directory_max_age_seconds,
        )
        return await service.refresh_directory()
    finally:
        if owns_source:
            await _close(active_source)
        if owns_cache:
            await _close(active_cache)


async def _run() -> None:
    directory = await refresh_symbol_directory(settings=get_settings())
    print(
        "symbol directory refreshed: "
        f"source={directory.source} records={len(directory.records)} "
        f"as_of={directory.as_of.isoformat()}"
    )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
