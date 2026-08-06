import asyncio
from datetime import UTC, date, datetime
from typing import Any

import pytest

from app.cache.memory import MemoryCache
from app.errors import CacheUnavailableError, ProviderUnavailableError, QuotaExceededError
from app.models import EODBar, Page, Ticker, Usage
from app.services.market_data import MarketDataService, ServiceTTLs


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class FakeProvider:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.error: Exception | None = None
        self.gate: asyncio.Event | None = None

    async def _result(self, name: str, value: Any) -> Any:
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.gate is not None:
            await self.gate.wait()
        if self.error:
            raise self.error
        return value

    async def list_tickers(self, *, limit: int = 100, cursor: str | None = None) -> Page[Ticker]:
        return await self._result("tickers", Page(items=(Ticker(symbol="MSFT", name="Microsoft"),)))

    async def latest_eod(self, symbol: str) -> EODBar:
        return await self._result(
            "latest",
            EODBar(symbol=symbol, date="2025-01-02", open=1, high=2, low=1, close=2, volume=3),
        )

    async def usage(self) -> Usage:
        return await self._result("usage", Usage(requests_used=1, requests_limit=100))


def test_ttls_match_policy() -> None:
    ttls = ServiceTTLs()
    assert ttls.references == 7 * 86400
    assert ttls.completed_history == 30 * 86400
    assert ttls.latest == 6 * 3600
    assert ttls.corporate_actions == 24 * 3600
    assert ttls.invalid == 3600
    assert ttls.stale == 7 * 86400


def test_service_caches_provider_results_and_reserves_quota_only_on_miss() -> None:
    provider = FakeProvider()
    cache = MemoryCache()
    service = MarketDataService(provider, cache)

    first = run(service.tickers(limit=10))
    second = run(service.tickers(limit=10))

    assert first == second
    assert provider.calls == {"tickers": 1}
    assert run(cache.current_count(service.quota_key)) == 1


def test_service_quota_is_monthly_utc_and_usage_is_local_without_provider_call() -> None:
    provider = FakeProvider()
    cache = MemoryCache()
    now = datetime(2026, 8, 7, tzinfo=UTC)
    service = MarketDataService(provider, cache, monthly_budget=90, now=lambda: now)
    run(service.latest_eod("MSFT"))

    usage = run(service.usage())

    assert usage.requests_used == 1
    assert usage.requests_limit == 90
    assert usage.requests_remaining == 89
    assert provider.calls == {"latest": 1}
    assert "2026-08" not in service.quota_key


def test_service_exposes_task_local_cache_metadata() -> None:
    async def scenario() -> None:
        provider = FakeProvider()
        service = MarketDataService(provider, MemoryCache())
        await service.latest_eod("MSFT")
        first = service.last_metadata
        await service.latest_eod("MSFT")
        second = service.last_metadata

        assert first is not None
        assert first.source == "provider"
        assert not first.cached
        assert second is not None
        assert second.source == "cache"
        assert second.cached
        assert not second.stale

    run(scenario())


def test_service_coalesces_concurrent_misses() -> None:
    async def scenario() -> None:
        class CountingCache(MemoryCache):
            acquired = 0
            released = 0

            async def acquire_lock(self, key: str, token: str, *, ttl_seconds: int) -> bool:
                self.acquired += 1
                return await super().acquire_lock(key, token, ttl_seconds=ttl_seconds)

            async def release_lock(self, key: str, token: str) -> bool:
                self.released += 1
                return await super().release_lock(key, token)

        provider = FakeProvider()
        provider.gate = asyncio.Event()
        cache = CountingCache()
        service = MarketDataService(provider, cache)
        tasks = [asyncio.create_task(service.latest_eod("MSFT")) for _ in range(8)]
        await asyncio.sleep(0)
        provider.gate.set()
        results = await asyncio.gather(*tasks)
        assert len(results) == 8
        assert provider.calls == {"latest": 1}
        assert cache.acquired == 1
        assert cache.released == 1

    run(scenario())


def test_service_returns_stale_on_provider_failure_or_exhausted_quota() -> None:
    async def scenario(error: Exception) -> EODBar:
        clock = [1000.0]
        provider = FakeProvider()
        cache = MemoryCache(clock=lambda: clock[0])
        service = MarketDataService(provider, cache)
        expected = await service.latest_eod("MSFT")
        clock[0] += ServiceTTLs().latest + 1
        provider.error = error
        assert await service.latest_eod("MSFT") == expected
        return expected

    run(scenario(ProviderUnavailableError("temporarily unavailable")))
    run(scenario(QuotaExceededError("quota exhausted")))


def test_service_fails_closed_when_cache_is_unavailable_without_stale() -> None:
    class BrokenCache(MemoryCache):
        async def get(self, key: str):  # type: ignore[no-untyped-def]
            raise CacheUnavailableError("cache down")

    provider = FakeProvider()
    service = MarketDataService(provider, BrokenCache())

    with pytest.raises(CacheUnavailableError):
        run(service.latest_eod("MSFT"))
    assert provider.calls == {}


def test_history_rejects_ranges_over_one_year_before_provider_call() -> None:
    provider = FakeProvider()
    service = MarketDataService(provider, MemoryCache())
    with pytest.raises(ValueError, match="one year"):
        run(service.history("MSFT", date(2024, 1, 1), date(2025, 1, 2)))
    assert provider.calls == {}


def test_service_closes_provider_and_cache_lifecycles() -> None:
    class ClosableProvider(FakeProvider):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    class ClosableCache(MemoryCache):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    provider = ClosableProvider()
    cache = ClosableCache()
    service = MarketDataService(provider, cache)

    run(service.aclose())

    assert provider.closed
    assert cache.closed
    assert service.manages(provider)
    assert service.manages(cache)
    assert not service.manages(object())
