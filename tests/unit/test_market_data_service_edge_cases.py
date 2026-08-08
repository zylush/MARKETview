from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.cache.base import CacheKeyBuilder
from app.cache.memory import MemoryCache
from app.errors import (
    CacheUnavailableError,
    ProviderNotFoundError,
    ProviderRequestRejectedError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from app.models import EODBar, Page
from app.services.market_data import MarketDataService, ServiceTTLs


class CompleteProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failure: Exception | None = None

    async def _return(self, name: str, params: dict[str, Any], result: Any) -> Any:
        self.calls = [*self.calls, (name, params)]
        if self.failure is not None:
            raise self.failure
        return result

    async def latest_eod(self, symbol: str) -> EODBar:
        return await self._return(
            "latest",
            {"symbol": symbol},
            EODBar(symbol=symbol, date="2026-08-07", close=Decimal("1.5")),
        )

    async def eod_history(self, symbol: str, **params: Any) -> Page[EODBar]:
        return await self._return(
            "history",
            {"symbol": symbol, **params},
            Page(items=(EODBar(symbol=symbol, date="2026-08-01", close=1),)),
        )


@pytest.mark.asyncio
async def test_service_delegates_normalized_history_parameters() -> None:
    provider = CompleteProvider()
    now = datetime(2026, 8, 7, tzinfo=UTC)
    service = MarketDataService(provider, MemoryCache(), now=lambda: now)

    await service.eod_history(
        "msft", date_from=date(2026, 8, 1), date_to=date(2026, 8, 7), offset=4
    )

    assert provider.calls == [
        (
            "history",
            {
                "symbol": "MSFT",
                "start_date": date(2026, 8, 1),
                "end_date": date(2026, 8, 7),
                "limit": 366,
                "cursor": None,
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("public_end", [date(2026, 8, 7), date(2026, 8, 8), date(2026, 8, 9)])
async def test_history_preserves_weekday_and_weekend_provider_dates(public_end: date) -> None:
    provider = CompleteProvider()
    service = MarketDataService(provider, MemoryCache())

    result = await service.history("AAPL", date(2026, 8, 1), public_end)

    assert result.total == 1
    assert provider.calls[0][1]["start_date"] == date(2026, 8, 1)
    assert provider.calls[0][1]["end_date"] == public_end


@pytest.mark.asyncio
async def test_distinct_friday_and_weekend_ranges_keep_distinct_cache_results() -> None:
    provider = CompleteProvider()
    cache = MemoryCache()
    service = MarketDataService(provider, cache)

    friday = await service.history("AAPL", date(2026, 8, 1), date(2026, 8, 7))
    saturday = await service.history("AAPL", date(2026, 8, 1), date(2026, 8, 8))
    sunday = await service.history("AAPL", date(2026, 8, 1), date(2026, 8, 9))

    assert friday == saturday == sunday
    assert [name for name, _ in provider.calls] == ["history", "history", "history"]
    assert await cache.current_count(service.quota_key) == 3


@pytest.mark.asyncio
async def test_contract_revision_bypasses_all_old_history_negative_cache_keys() -> None:
    provider = CompleteProvider()
    cache = MemoryCache()
    for old_end in (date(2026, 8, 7), date(2026, 8, 8)):
        old_key = CacheKeyBuilder().build(
            "eod_history",
            {"symbol": "AAPL", "start": date(2025, 8, 8), "end": old_end},
        )
        await cache.set(
            old_key,
            {"status": "invalid", "code": "provider_validation_error"},
            ttl_seconds=3600,
        )
    service = MarketDataService(provider, cache)
    await cache.reserve_quota(service.quota_key, limit=90, window_seconds=86400)

    result = await service.history("AAPL", date(2025, 8, 8), date(2026, 8, 8))

    assert result.total == 1
    assert [name for name, _ in provider.calls] == ["history"]
    assert provider.calls[0][1]["end_date"] == date(2026, 8, 8)
    assert await cache.current_count(service.quota_key) == 2


@pytest.mark.asyncio
async def test_contract_revision_bypasses_old_latest_negative_cache_key() -> None:
    provider = CompleteProvider()
    cache = MemoryCache()
    old_key = CacheKeyBuilder().build("latest_eod", {"symbol": "AAPL"})
    await cache.set(
        old_key,
        {"status": "invalid", "code": "provider_validation_error"},
        ttl_seconds=3600,
    )
    service = MarketDataService(provider, cache)

    result = await service.latest_eod("AAPL")

    assert result.symbol == "AAPL"
    assert [name for name, _ in provider.calls] == ["latest"]
    assert await cache.current_count(service.quota_key) == 1


@pytest.mark.asyncio
async def test_new_contract_bypasses_v3_latest_result_cache_key() -> None:
    provider = CompleteProvider()
    cache = MemoryCache()
    v3_key = CacheKeyBuilder().build(
        "latest_eod",
        {"symbol": "AAPL", "provider_contract": "marketdata-candles-v3"},
    )
    await cache.set(
        v3_key,
        {
            "status": "ok",
            "payload": EODBar(symbol="AAPL", date="2026-08-06", close=Decimal("999")).model_dump(
                mode="json"
            ),
            "as_of": "2026-08-07T00:00:00+00:00",
        },
        ttl_seconds=3600,
    )
    service = MarketDataService(provider, cache)

    result = await service.latest_eod("AAPL")

    assert result.close == Decimal("1.5")
    assert [name for name, _ in provider.calls] == ["latest"]
    assert await cache.current_count(service.quota_key) == 1


@pytest.mark.asyncio
async def test_new_contract_bypasses_v3_history_result_cache_key() -> None:
    provider = CompleteProvider()
    cache = MemoryCache()
    v3_key = CacheKeyBuilder().build(
        "eod_history",
        {
            "symbol": "AAPL",
            "start": date(2026, 8, 1),
            "end": date(2026, 8, 7),
            "provider_contract": "marketdata-candles-v3",
        },
    )
    await cache.set(
        v3_key,
        {
            "status": "ok",
            "payload": Page(
                items=(EODBar(symbol="AAPL", date="2026-08-06", close=Decimal("999")),)
            ).model_dump(mode="json"),
            "as_of": "2026-08-07T00:00:00+00:00",
        },
        ttl_seconds=3600,
    )
    service = MarketDataService(provider, cache)

    result = await service.history("AAPL", date(2026, 8, 1), date(2026, 8, 7))

    assert result.items[0].close == Decimal("1")
    assert [name for name, _ in provider.calls] == ["history"]
    assert await cache.current_count(service.quota_key) == 1


@pytest.mark.asyncio
async def test_weekend_only_range_reaches_provider_without_local_semantic_rejection() -> None:
    provider = CompleteProvider()
    cache = MemoryCache()
    service = MarketDataService(provider, cache)

    result = await service.history(
        "AAPL",
        date(2026, 8, 8),
        date(2026, 8, 8),
        limit=2,
        cursor="9",
    )

    assert result.total == 1
    assert provider.calls[0][1]["start_date"] == date(2026, 8, 8)
    assert provider.calls[0][1]["end_date"] == date(2026, 8, 8)
    assert await cache.current_count(service.quota_key) == 1


@pytest.mark.asyncio
async def test_provider_request_rejection_is_not_cached_and_refunds_each_reservation() -> None:
    provider = CompleteProvider()
    provider.failure = ProviderRequestRejectedError("private upstream response")
    cache = MemoryCache()
    service = MarketDataService(provider, cache)

    for _ in range(2):
        with pytest.raises(ProviderRequestRejectedError):
            await service.history("AAPL", date(2025, 8, 8), date(2026, 8, 8))

    assert [name for name, _ in provider.calls] == ["history", "history"]
    assert await cache.current_count(service.quota_key) == 0


@pytest.mark.asyncio
async def test_service_negative_caches_not_found_and_rolls_back_reservation() -> None:
    provider = CompleteProvider()
    provider.failure = ProviderNotFoundError("private upstream response")
    cache = MemoryCache()
    service = MarketDataService(provider, cache)

    with pytest.raises(ProviderNotFoundError):
        await service.latest_eod("MSFT")
    assert await cache.current_count(service.quota_key) == 0
    with pytest.raises(ProviderNotFoundError, match="not found"):
        await service.latest_eod("MSFT")

    assert [name for name, _ in provider.calls] == ["latest"]
    assert await cache.current_count(service.quota_key) == 0


@pytest.mark.asyncio
async def test_service_rolls_back_provider_failure_without_stale_data() -> None:
    provider = CompleteProvider()
    provider.failure = ProviderUnavailableError("timeout")
    cache = MemoryCache()
    service = MarketDataService(provider, cache)

    with pytest.raises(ProviderUnavailableError):
        await service.latest_eod("MSFT")

    assert await cache.current_count(service.quota_key) == 0


@pytest.mark.asyncio
async def test_service_enforces_quota_before_provider_call() -> None:
    class ExhaustedCache(MemoryCache):
        async def reserve_quota(self, key: str, *, limit: int = 90, window_seconds: int = 86400):
            del key, limit, window_seconds
            return

    provider = CompleteProvider()
    service = MarketDataService(provider, ExhaustedCache())

    with pytest.raises(QuotaExceededError):
        await service.latest_eod("MSFT")
    assert provider.calls == []


@pytest.mark.asyncio
async def test_service_fails_closed_when_distributed_refresh_is_busy_without_stale_data() -> None:
    class BusyCache(MemoryCache):
        async def acquire_lock(self, key: str, token: str, *, ttl_seconds: int) -> bool:
            del key, token, ttl_seconds
            return False

    provider = CompleteProvider()
    service = MarketDataService(provider, BusyCache(), lock_wait_attempts=0)

    with pytest.raises(CacheUnavailableError, match="already in progress"):
        await service.latest_eod("MSFT")
    assert provider.calls == []


@pytest.mark.asyncio
async def test_service_returns_stale_when_refresh_lock_backend_fails() -> None:
    class BrokenAcquireCache(MemoryCache):
        fail_acquire = False

        async def acquire_lock(self, key: str, token: str, *, ttl_seconds: int) -> bool:
            if self.fail_acquire:
                raise ConnectionError("private cache endpoint")
            return await super().acquire_lock(key, token, ttl_seconds=ttl_seconds)

    clock = [1000.0]
    cache = BrokenAcquireCache(clock=lambda: clock[0])
    provider = CompleteProvider()
    service = MarketDataService(provider, cache)
    expected = await service.latest_eod("MSFT")
    clock[0] += ServiceTTLs().latest + 1
    cache.fail_acquire = True

    assert await service.latest_eod("MSFT") == expected
    assert [name for name, _ in provider.calls] == ["latest"]


@pytest.mark.asyncio
async def test_service_wraps_unknown_cache_failures_and_never_calls_provider() -> None:
    class BrokenCache(MemoryCache):
        async def get(self, key: str):
            del key
            raise ConnectionError("redis token and host")

    provider = CompleteProvider()
    service = MarketDataService(provider, BrokenCache())

    with pytest.raises(CacheUnavailableError) as caught:
        await service.latest_eod("MSFT")

    assert "redis token" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert provider.calls == []


@pytest.mark.asyncio
async def test_usable_provider_data_remains_counted_when_cache_write_fails() -> None:
    class BrokenSetCache(MemoryCache):
        async def set(
            self, key: str, value: Any, *, ttl_seconds: int, stale_seconds: int = 0
        ) -> None:
            del key, value, ttl_seconds, stale_seconds
            raise CacheUnavailableError("cache down")

    cache = BrokenSetCache()
    provider = CompleteProvider()
    service = MarketDataService(provider, cache)

    with pytest.raises(CacheUnavailableError):
        await service.latest_eod("MSFT")

    assert await cache.current_count(service.quota_key) == 1
    assert [name for name, _ in provider.calls] == ["latest"]


@pytest.mark.asyncio
async def test_quota_rollback_failure_is_sanitized_and_does_not_hide_provider_failure() -> None:
    class BrokenReleaseCache(MemoryCache):
        async def release_quota(self, key: str) -> int:
            del key
            raise ConnectionError("private redis detail")

    provider = CompleteProvider()
    provider.failure = ProviderUnavailableError("safe provider failure")
    service = MarketDataService(provider, BrokenReleaseCache())

    with pytest.raises(ProviderUnavailableError, match="safe provider failure") as caught:
        await service.latest_eod("MSFT")

    assert "redis" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_failure_crossing_utc_midnight_rolls_back_the_original_daily_key() -> None:
    current = [datetime(2026, 8, 7, 23, 59, 59, tzinfo=UTC)]
    cache = MemoryCache()

    class MidnightProvider(CompleteProvider):
        async def latest_eod(self, symbol: str) -> EODBar:
            self.calls = [*self.calls, ("latest", {"symbol": symbol})]
            current[0] = datetime(2026, 8, 8, tzinfo=UTC)
            raise ProviderUnavailableError("provider unavailable")

    service = MarketDataService(MidnightProvider(), cache, now=lambda: current[0])
    old_key = service.quota_key

    with pytest.raises(ProviderUnavailableError):
        await service.latest_eod("MSFT")

    assert await cache.current_count(old_key) == 0
    assert await cache.current_count(service.quota_key) == 0
