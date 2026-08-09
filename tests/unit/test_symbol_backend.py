from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from app.cache.memory import MemoryCache
from app.errors import CacheUnavailableError
from app.models import SymbolRecord
from app.services.symbols import SymbolDirectory, SymbolSearchService


def run(coro):
    return asyncio.run(coro)


class RecordingCache(MemoryCache):
    def __init__(self, *, clock=lambda: 1_000.0) -> None:
        super().__init__(clock=clock)
        self.writes: list[tuple[str, object]] = []
        self.reads: list[str] = []

    async def get(self, key: str):
        self.reads = [*self.reads, key]
        return await super().get(key)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int,
        stale_seconds: int = 0,
    ) -> None:
        self.writes = [*self.writes, (key, value)]
        await super().set(
            key,
            value,
            ttl_seconds=ttl_seconds,
            stale_seconds=stale_seconds,
        )


class CountingSource:
    def __init__(self, directory: SymbolDirectory, *, failure: Exception | None = None) -> None:
        self.directory = directory
        self.failure = failure
        self.calls = 0

    async def load(self) -> SymbolDirectory:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.directory


def directory(*records: SymbolRecord, hour: int = 0) -> SymbolDirectory:
    return SymbolDirectory(
        records=tuple(records),
        source="sec-company-tickers-exchange",
        as_of=datetime(2026, 8, 9, hour, tzinfo=UTC),
    )


def test_search_never_calls_source_and_requires_explicit_refresh() -> None:
    source = CountingSource(
        directory(SymbolRecord(symbol="AAPL", name="Apple Inc.", exchange="Nasdaq"))
    )
    service = SymbolSearchService(source, MemoryCache())

    with pytest.raises(CacheUnavailableError, match="unavailable") as captured:
        run(service.search_symbols("app"))

    assert source.calls == 0
    assert "app" not in str(captured.value).lower()


def test_runtime_search_service_can_be_constructed_without_sec_source() -> None:
    cache = MemoryCache()
    refresher = SymbolSearchService(
        CountingSource(
            directory(SymbolRecord(symbol="AAPL", name="Apple Inc.", exchange="Nasdaq"))
        ),
        cache,
    )
    run(refresher.refresh_directory())
    runtime = SymbolSearchService(None, cache)

    result = run(runtime.search_symbols("app"))

    assert result.items[0].symbol == "AAPL"
    with pytest.raises(CacheUnavailableError, match="refresh failed"):
        run(runtime.refresh_directory())


def test_refresh_writes_generation_before_manifest_then_searches_cached_index() -> None:
    source = CountingSource(
        directory(
            SymbolRecord(symbol="AAPL", name="Apple Inc.", exchange="Nasdaq"),
            SymbolRecord(symbol="APP", name="AppLovin Corporation", exchange="Nasdaq"),
        )
    )
    cache = RecordingCache()
    service = SymbolSearchService(source, cache)

    refreshed = run(service.refresh_directory())
    result = run(service.search_symbols("app", limit=8))

    assert refreshed.records[0].symbol == "AAPL"
    assert source.calls == 1
    assert len(cache.writes) > 2
    assert all("generation_bucket" in key for key, _ in cache.writes[:-1])
    assert "manifest" in cache.writes[-1][0]
    assert [item.symbol for item in result.items] == ["APP", "AAPL"]
    assert result.source == "sec-company-tickers-exchange"
    assert result.as_of == datetime(2026, 8, 9, tzinfo=UTC)
    assert result.stale is False

    run(service.search_symbols("apple"))
    assert source.calls == 1


def test_search_reads_only_manifest_and_relevant_two_character_bucket() -> None:
    source = CountingSource(
        directory(
            SymbolRecord(symbol="AAPL", name="Apple Inc.", exchange="Nasdaq"),
            SymbolRecord(symbol="MSFT", name="Microsoft Corporation", exchange="Nasdaq"),
            SymbolRecord(symbol="APP", name="AppLovin Corporation", exchange="Nasdaq"),
        )
    )
    cache = RecordingCache()
    service = SymbolSearchService(source, cache)
    run(service.refresh_directory())
    bucket_keys = {
        tuple(item["symbol"] for item in value["records"]): key
        for key, value in cache.writes[:-1]
        if isinstance(value, dict)
    }
    unrelated_key = bucket_keys[("MSFT",)]
    cache.reads = []

    result = run(service.search_symbols("app"))

    assert [item.symbol for item in result.items] == ["APP", "AAPL"]
    assert len(cache.reads) == 2
    assert cache.reads[0] == service.manifest_cache_key
    assert unrelated_key not in cache.reads


def test_ticker_prefix_precedes_company_token_prefix_with_deterministic_order() -> None:
    source = CountingSource(
        directory(
            SymbolRecord(symbol="ZAPP", name="Zed Holdings", exchange="NYSE"),
            SymbolRecord(symbol="APP", name="AppLovin Corporation", exchange="Nasdaq"),
            SymbolRecord(symbol="AAPL", name="The Apple Company", exchange="Nasdaq"),
            SymbolRecord(symbol="APPC", name="Example Corporation", exchange="NYSE"),
            SymbolRecord(symbol="XAPL", name="Global Apple Partners", exchange="NYSE"),
        )
    )
    service = SymbolSearchService(source, MemoryCache())
    run(service.refresh_directory())

    result = run(service.search_symbols("app", limit=8))

    assert [item.symbol for item in result.items] == ["APP", "APPC", "AAPL", "XAPL"]


def test_query_is_normalized_and_result_count_is_bounded_to_eight() -> None:
    records = tuple(
        SymbolRecord(symbol=f"AA{index}", name=f"Issuer {index}", exchange="NYSE")
        for index in range(12)
    )
    service = SymbolSearchService(CountingSource(directory(*records)), MemoryCache())
    run(service.refresh_directory())

    result = run(service.search_symbols("  aa  ", limit=8))

    assert len(result.items) == 8
    assert result.total == 8
    with pytest.raises(ValueError, match="at least two"):
        run(service.search_symbols("a"))
    with pytest.raises(ValueError, match="between 1 and 8"):
        run(service.search_symbols("aa", limit=9))


def test_stale_metadata_is_true_when_manifest_or_generation_is_stale() -> None:
    now = [1_000.0]
    source = CountingSource(
        directory(SymbolRecord(symbol="MSFT", name="Microsoft Corporation", exchange="Nasdaq"))
    )
    service = SymbolSearchService(
        source,
        MemoryCache(clock=lambda: now[0]),
        directory_ttl_seconds=10,
        stale_seconds=30,
    )
    run(service.refresh_directory())
    now[0] = 1_011.0

    result = run(service.search_symbols("micro"))

    assert result.items[0].symbol == "MSFT"
    assert result.stale is True
    assert source.calls == 1


def test_failed_refresh_preserves_previous_generation() -> None:
    source = CountingSource(
        directory(SymbolRecord(symbol="MSFT", name="Microsoft Corporation", exchange="Nasdaq"))
    )
    cache = RecordingCache()
    service = SymbolSearchService(source, cache)
    run(service.refresh_directory())
    writes_before_failure = tuple(cache.writes)
    source.failure = RuntimeError("upstream payload marker secret-query=APPLE")

    with pytest.raises(CacheUnavailableError, match="refresh failed") as captured:
        run(service.refresh_directory())

    assert "secret-query" not in str(captured.value)
    assert tuple(cache.writes) == writes_before_failure
    result = run(service.search_symbols("micro"))
    assert result.items[0].symbol == "MSFT"


def test_failed_bucket_write_does_not_switch_manifest() -> None:
    class FailingWriteCache(RecordingCache):
        fail_after: int | None = None

        async def set(self, key: str, value: Any, **kwargs: Any) -> None:
            if self.fail_after is not None and len(self.writes) >= self.fail_after:
                raise CacheUnavailableError("write payload marker")
            await super().set(key, value, **kwargs)

    cache = FailingWriteCache()
    original_source = CountingSource(
        directory(SymbolRecord(symbol="MSFT", name="Microsoft Corporation", exchange="Nasdaq"))
    )
    service = SymbolSearchService(original_source, cache)
    run(service.refresh_directory())
    manifest_entry = run(cache.get(service.manifest_cache_key))
    assert manifest_entry is not None
    manifest_before = manifest_entry.value
    original_source.directory = directory(
        SymbolRecord(symbol="AAPL", name="Apple Inc.", exchange="Nasdaq"), hour=1
    )
    cache.fail_after = len(cache.writes) + 1

    with pytest.raises(CacheUnavailableError, match="refresh failed"):
        run(service.refresh_directory())

    manifest_entry = run(cache.get(service.manifest_cache_key))
    assert manifest_entry is not None
    assert manifest_entry.value == manifest_before
    assert run(service.search_symbols("micro")).items[0].symbol == "MSFT"


def test_invalid_cached_generation_is_rejected_with_sanitized_error() -> None:
    cache = RecordingCache()
    service = SymbolSearchService(CountingSource(directory()), cache)
    awaitable = cache.set(
        service.manifest_cache_key,
        {"generation": "../../payload-marker"},
        ttl_seconds=60,
        stale_seconds=60,
    )
    run(awaitable)

    with pytest.raises(CacheUnavailableError, match="invalid") as captured:
        run(service.search_symbols("aa"))

    assert "payload-marker" not in str(captured.value)


def test_cache_transport_error_does_not_retain_payload_or_query() -> None:
    class LeakyCache(MemoryCache):
        async def get(self, key: str):
            raise CacheUnavailableError("secret-cache-payload query=APPLE")

    service = SymbolSearchService(CountingSource(directory()), LeakyCache())

    with pytest.raises(CacheUnavailableError, match="unavailable") as captured:
        run(service.search_symbols("apple"))

    assert "secret-cache-payload" not in str(captured.value)
    assert "apple" not in str(captured.value).lower()


def test_symbol_directory_and_records_are_immutable() -> None:
    record = SymbolRecord(symbol="AAPL", name="Apple Inc.", exchange="Nasdaq")
    value = directory(record)

    with pytest.raises(FrozenInstanceError):
        value.source = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        record.symbol = "MSFT"  # type: ignore[misc]
