from __future__ import annotations

from app.cache.base import Cache
from app.cache.memory import MemoryCache
from app.cache.upstash import UpstashCache
from app.config import Settings, get_settings
from app.main import create_app
from app.providers.marketdata import MarketDataAppProvider
from app.services.dashboard import DashboardService
from app.services.market_data import MarketDataService
from app.services.research import DisabledResearchService, build_research_runtime
from app.services.symbols import SymbolSearchService


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

    marketdata_token = settings.marketdata_token.get_secret_value()
    if not marketdata_token:
        if not settings.is_local_environment:
            raise RuntimeError("MARKETDATA_TOKEN is required")
        return settings, object(), cache
    provider = MarketDataAppProvider(
        marketdata_token,
        base_url=settings.marketdata_base_url,
        timeout_seconds=settings.http_timeout_seconds,
    )
    market_data = MarketDataService(
        provider,
        cache,
        daily_credit_budget=settings.marketdata_daily_credit_budget,
        schema_version=settings.cache_schema_version,
    )
    symbols = SymbolSearchService(
        None,
        cache,
        schema_version=settings.symbol_index_schema_version,
        directory_ttl_seconds=settings.symbol_directory_max_age_seconds,
    )
    research = (
        build_research_runtime(settings) if settings.research_enabled else DisabledResearchService()
    )
    service = DashboardService(market_data, symbols, research)
    return settings, service, cache


_settings, _service, _cache = _runtime_dependencies()
app = create_app(settings=_settings, service=_service, cache=_cache)

__all__ = ["app"]
