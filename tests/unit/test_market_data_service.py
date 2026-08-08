import asyncio
from datetime import UTC, date, datetime
from typing import Any

import pytest

from app.cache.memory import MemoryCache
from app.errors import CacheUnavailableError, ProviderUnavailableError, QuotaExceededError
from app.models import EODBar, Page
from app.services.market_data import MarketDataService, ServiceTTLs


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class FakeProvider:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.error: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.history_items: tuple[EODBar, ...] = (
            EODBar(symbol="MSFT", date="2026-08-07", close=2),
        )

    async def _result(self, name: str, value: Any) -> Any:
        self.calls = {**self.calls, name: self.calls.get(name, 0) + 1}
        if self.gate is not None:
            await self.gate.wait()
        if self.error:
            raise self.error
        return value

    async def latest_eod(self, symbol: str) -> EODBar:
        return await self._result(
            "latest",
            EODBar(symbol=symbol, date="2026-08-07", open=1, high=2, low=1, close=2, volume=3),
        )

    async def eod_history(
        self,
        symbol: str,
        *,
        start_date: date,
        end_date: date,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[EODBar]:
        del start_date, end_date, limit, cursor
        items = tuple(item.model_copy(update={"symbol": symbol}) for item in self.history_items)
        return await self._result("history", Page(items=items, total=len(items)))


def test_ttls_match_supported_market_data_policy() -> None:
    ttls = ServiceTTLs()
    assert ttls.completed_history == 30 * 86400
    assert ttls.latest == 6 * 3600
    assert ttls.invalid == 3600
    assert ttls.stale == 7 * 86400


def test_service_caches_provider_results_and_reserves_quota_only_on_miss() -> None:
    provider = FakeProvider()
    cache = MemoryCache()
    service = MarketDataService(provider, cache)

    first = run(service.latest_eod("msft"))
    second = run(service.latest_eod("MSFT"))

    assert first == second
    assert provider.calls == {"latest": 1}
    assert run(cache.current_count(service.quota_key)) == 1


def test_history_pages_share_one_complete_range_provider_fetch() -> None:
    async def scenario() -> None:
        provider = FakeProvider()
        provider.history_items = tuple(
            EODBar(symbol="MSFT", date=date(2026, 8, day), close=day) for day in range(1, 6)
        )
        cache = MemoryCache()
        service = MarketDataService(provider, cache, now=lambda: datetime(2026, 8, 7, tzinfo=UTC))

        first = await service.history("MSFT", date(2026, 8, 1), date(2026, 8, 5), limit=2)
        second = await service.history(
            "MSFT", date(2026, 8, 1), date(2026, 8, 5), limit=2, cursor="2"
        )

        assert [item.date.day for item in first.items] == [1, 2]
        assert first.total == 5
        assert first.next_cursor == "2"
        assert [item.date.day for item in second.items] == [3, 4]
        assert second.total == 5
        assert second.next_cursor == "4"
        assert provider.calls == {"history": 1}
        assert await cache.current_count(service.quota_key) == 1

    run(scenario())


def test_service_quota_is_daily_utc_and_usage_is_local_without_provider_call() -> None:
    async def scenario() -> None:
        provider = FakeProvider()
        cache = MemoryCache()
        current = [datetime(2026, 8, 7, 23, 59, 30, tzinfo=UTC)]
        service = MarketDataService(provider, cache, daily_credit_budget=90, now=lambda: current[0])
        first_key = service.quota_key
        await service.latest_eod("MSFT")

        usage = await service.usage()

        assert usage.requests_used == 1
        assert usage.requests_limit == 90
        assert usage.requests_remaining == 89
        assert usage.reset_at == datetime(2026, 8, 8, tzinfo=UTC)
        assert service.last_metadata is not None
        assert service.last_metadata.source == "local"
        assert provider.calls == {"latest": 1}

        current[0] = datetime(2026, 8, 8, tzinfo=UTC)
        assert service.quota_key != first_key
        reset_usage = await service.usage()
        assert reset_usage.requests_used == 0
        assert reset_usage.reset_at == datetime(2026, 8, 9, tzinfo=UTC)

    run(scenario())


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
        assert await cache.current_count(service.quota_key) == 1
        assert cache.acquired == 1
        assert cache.released == 1

    run(scenario())


def test_service_rolls_back_failed_provider_reservation_and_returns_stale() -> None:
    async def scenario() -> None:
        clock = [1000.0]
        provider = FakeProvider()
        cache = MemoryCache(clock=lambda: clock[0])
        service = MarketDataService(provider, cache)
        expected = await service.latest_eod("MSFT")
        assert await cache.current_count(service.quota_key) == 1
        clock[0] += ServiceTTLs().latest + 1
        provider.error = ProviderUnavailableError("temporarily unavailable")

        assert await service.latest_eod("MSFT") == expected
        assert provider.calls == {"latest": 2}
        assert await cache.current_count(service.quota_key) == 1

    run(scenario())


def test_service_returns_stale_without_call_when_quota_is_exhausted() -> None:
    async def scenario() -> None:
        clock = [1000.0]
        provider = FakeProvider()
        cache = MemoryCache(clock=lambda: clock[0])
        service = MarketDataService(provider, cache, daily_credit_budget=1)
        expected = await service.latest_eod("MSFT")
        clock[0] += ServiceTTLs().latest + 1

        assert await service.latest_eod("MSFT") == expected
        assert provider.calls == {"latest": 1}

    run(scenario())


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


def test_quota_rejection_does_not_increment_or_call_provider() -> None:
    provider = FakeProvider()
    cache = MemoryCache()
    service = MarketDataService(provider, cache, daily_credit_budget=1)
    assert run(cache.reserve_quota(service.quota_key, limit=1, window_seconds=60)) == 1

    with pytest.raises(QuotaExceededError, match="daily"):
        run(service.latest_eod("AAPL"))

    assert run(cache.current_count(service.quota_key)) == 1
    assert provider.calls == {}
