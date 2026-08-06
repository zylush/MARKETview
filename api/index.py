from __future__ import annotations

from app.cache.base import Cache
from app.cache.memory import MemoryCache
from app.cache.upstash import UpstashCache
from app.config import Settings, get_settings
from app.main import create_app
from app.providers.marketstack import MarketstackProvider
from app.services.market_data import MarketDataService


def _runtime_dependencies(settings: Settings | None = None) -> tuple[object, object, object]:
    settings = settings or get_settings()
    redis_token = (
        settings.upstash_redis_rest_token.get_secret_value()
        if settings.upstash_redis_rest_token
        else ""
    )
    if settings.upstash_redis_rest_url and redis_token:
        cache: Cache = UpstashCache(settings.upstash_redis_rest_url, redis_token)
    elif settings.is_local_environment:
        cache = MemoryCache()
    else:
        raise RuntimeError("a shared cache is required outside local and test environments")

    if not settings.marketstack_api_key:
        if not settings.is_local_environment:
            raise RuntimeError("MARKETSTACK_API_KEY is required")
        return settings, object(), cache
    provider = MarketstackProvider(
        settings.marketstack_api_key,
        base_url=settings.marketstack_base_url,
        timeout_seconds=settings.marketstack_timeout_seconds,
    )
    service = MarketDataService(
        provider,
        cache,
        monthly_budget=settings.marketstack_monthly_budget,
        schema_version=settings.cache_schema_version,
    )
    return settings, service, cache


_settings, _service, _cache = _runtime_dependencies()
app = create_app(settings=_settings, service=_service, cache=_cache)

__all__ = ["app"]
