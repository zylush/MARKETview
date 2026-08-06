from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.cache.memory import MemoryCache
from app.errors import (
    CacheUnavailableError,
    ProviderNotFoundError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from app.models import Dividend, EODBar, Exchange, Page, Split, Ticker
from app.services.market_data import MarketDataService


class CompleteProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failure: Exception | None = None

    async def _return(self, name: str, params: dict[str, Any], result: Any) -> Any:
        self.calls = [*self.calls, (name, params)]
        if self.failure is not None:
            raise self.failure
        return result

    async def list_tickers(self, **params: Any) -> Page[Ticker]:
        return await self._return("tickers", params, Page(items=(Ticker(symbol="MSFT"),)))

    async def list_exchanges(self, **params: Any) -> Page[Exchange]:
        return await self._return(
            "exchanges", params, Page(items=(Exchange(name="NASDAQ", mic="XNAS"),))
        )

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

    async def splits(self, symbol: str | None, **params: Any) -> Page[Split]:
        return await self._return(
            "splits",
            {"symbol": symbol, **params},
            Page(items=(Split(symbol=symbol or "MSFT", date="2026-01-01", ratio=2),)),
        )

    async def dividends(self, symbol: str | None, **params: Any) -> Page[Dividend]:
        return await self._return(
            "dividends",
            {"symbol": symbol, **params},
            Page(items=(Dividend(symbol=symbol or "MSFT", date="2026-01-01", amount=1),)),
        )


@pytest.mark.asyncio
async def test_service_delegates_normalized_reference_history_and_action_parameters() -> None:
    provider = CompleteProvider()
    now = datetime(2026, 8, 7, tzinfo=UTC)
    service = MarketDataService(provider, MemoryCache(), now=lambda: now)

    await service.tickers(search="  msft  ", limit=10, offset=2)
    await service.exchanges(search="  nasdaq  ", limit=20, cursor="3")
    await service.eod_history(
        "msft", date_from=date(2026, 8, 1), date_to=date(2026, 8, 7), offset=4
    )
    await service.splits("msft", date_from=date(2026, 1, 1), date_to=date(2026, 2, 1), offset=5)
    await service.dividends(limit=25)

    assert provider.calls[0] == ("tickers", {"limit": 10, "cursor": "2", "search": "msft"})
    assert provider.calls[1] == (
        "exchanges",
        {"limit": 20, "cursor": "3", "search": "nasdaq"},
    )
    assert provider.calls[2][0] == "history"
    assert provider.calls[3][1]["start_date"] == date(2026, 1, 1)
    assert provider.calls[4][1]["symbol"] is None


@pytest.mark.asyncio
async def test_service_rejects_partial_action_range_before_provider_or_quota_use() -> None:
    provider = CompleteProvider()
    cache = MemoryCache()
    service = MarketDataService(provider, cache)

    with pytest.raises(ValueError, match="provided together"):
        await service.splits("MSFT", date_from=date(2026, 1, 1))

    assert provider.calls == []
    assert await cache.current_count(service.quota_key) == 0


@pytest.mark.asyncio
async def test_service_negative_caches_not_found_without_second_provider_call() -> None:
    provider = CompleteProvider()
    provider.failure = ProviderNotFoundError("private upstream response")
    service = MarketDataService(provider, MemoryCache())

    with pytest.raises(ProviderNotFoundError):
        await service.latest_eod("MSFT")
    with pytest.raises(ProviderNotFoundError, match="not found"):
        await service.latest_eod("MSFT")

    assert [name for name, _ in provider.calls] == ["latest"]


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
    assert provider.calls == []


@pytest.mark.asyncio
async def test_service_propagates_provider_failure_when_no_stale_data_exists() -> None:
    provider = CompleteProvider()
    provider.failure = ProviderUnavailableError("timeout")
    service = MarketDataService(provider, MemoryCache())

    with pytest.raises(ProviderUnavailableError):
        await service.latest_eod("MSFT")


@pytest.mark.asyncio
async def test_december_quota_window_rolls_to_next_year() -> None:
    class RecordingCache(MemoryCache):
        window_seconds: int | None = None

        async def reserve_quota(self, key: str, *, limit: int = 90, window_seconds: int = 86400):
            self.window_seconds = window_seconds
            return await super().reserve_quota(key, limit=limit, window_seconds=window_seconds)

    now = datetime(2026, 12, 31, 23, 0, tzinfo=UTC)
    cache = RecordingCache()
    await MarketDataService(CompleteProvider(), cache, now=lambda: now).latest_eod("MSFT")

    assert cache.window_seconds == 3600
