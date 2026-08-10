from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

import app.refresh_symbols as refresh_module
from app.cache.memory import MemoryCache
from app.config import Settings
from app.errors import CacheUnavailableError
from app.models import SymbolRecord
from app.refresh_symbols import refresh_symbol_directory
from app.services.symbols import StaticSymbolDirectorySource, SymbolSearchService


def run(coro):
    return asyncio.run(coro)


def test_control_plane_refresh_populates_searchable_generation_without_network() -> None:
    cache = MemoryCache()
    source = StaticSymbolDirectorySource(
        records=(
            SymbolRecord(symbol="AAPL", name="Apple Inc.", exchange="Nasdaq"),
            SymbolRecord(symbol="APP", name="AppLovin Corporation", exchange="Nasdaq"),
        ),
        source="sec-company-tickers-exchange",
        as_of=datetime(2026, 8, 9, tzinfo=UTC),
    )
    settings = Settings(
        environment="test",
        sec_user_agent="MarketView/1.0 operations@example.com",
        symbol_index_schema_version="v9",
        symbol_directory_max_age_seconds=3600,
        _env_file=None,
    )

    refreshed = run(refresh_symbol_directory(settings=settings, cache=cache, source=source))
    search = SymbolSearchService(None, cache, schema_version="v9")
    result = run(search.search_symbols("app"))

    assert len(refreshed.records) == 2
    assert [item.symbol for item in result.items] == ["APP", "AAPL"]
    assert result.source == "sec-company-tickers-exchange"


def test_refresh_rejects_missing_sec_identity_before_constructing_external_resources() -> None:
    settings = Settings(environment="test", _env_file=None)

    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        run(refresh_symbol_directory(settings=settings, cache=MemoryCache()))


def test_refresh_requires_shared_upstash_when_cache_is_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)

    settings = Settings(
        environment="test",
        sec_user_agent="MarketView/1.0 operations@example.com",
        _env_file=None,
    )

    with pytest.raises(RuntimeError, match="UPSTASH_REDIS"):
        run(refresh_symbol_directory(settings=settings))


def test_refresh_failure_traceback_retains_no_upstash_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_sentinel = "refresh-upstash-token-sentinel-7291"

    class SafeFakeCache(MemoryCache):
        def __init__(self, url: str, supplied_token: str) -> None:
            del url, supplied_token
            super().__init__()

        async def aclose(self) -> None:
            return None

    class FailingSource:
        async def load(self):
            raise RuntimeError("source failed")

    monkeypatch.setattr(refresh_module, "UpstashCache", SafeFakeCache)
    settings = Settings(
        environment="test",
        sec_user_agent="MarketView/1.0 operations@example.com",
        upstash_redis_rest_url="https://cache-name.upstash.io",
        upstash_redis_rest_token=provider_sentinel,
        _env_file=None,
    )

    with pytest.raises(CacheUnavailableError) as captured:
        run(refresh_symbol_directory(settings=settings, source=FailingSource()))

    retained: list[str] = []
    traceback = captured.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("refresh_symbols.py"):
            retained.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    assert provider_sentinel not in "\n".join(retained)
