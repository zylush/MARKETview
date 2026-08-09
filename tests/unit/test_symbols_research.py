from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.cache.memory import MemoryCache
from app.errors import CacheUnavailableError
from app.models import SymbolRecord
from app.services.symbols import StaticSymbolDirectorySource, SymbolSearchService
from app.validation import validate_symbol_query


def run(coro):
    return asyncio.run(coro)


def test_symbol_query_validation_normalizes_and_bounds() -> None:
    assert validate_symbol_query(" app ") == "APP"
    with pytest.raises(ValueError, match="at least two"):
        validate_symbol_query("a")
    with pytest.raises(ValueError, match="letters"):
        validate_symbol_query("<script>")


def test_symbol_search_ranks_prefix_before_company_name_and_limits() -> None:
    service = SymbolSearchService(
        StaticSymbolDirectorySource(
            records=(
                SymbolRecord(symbol="AAPL", name="Apple Inc.", exchange="Nasdaq"),
                SymbolRecord(symbol="APP", name="Applovin Corporation", exchange="Nasdaq"),
                SymbolRecord(symbol="XPEL", name="Apparel Holding", exchange="NYSE"),
            ),
            as_of=datetime(2026, 8, 9, tzinfo=UTC),
        ),
        MemoryCache(),
    )

    run(service.refresh_directory())
    result = run(service.search_symbols("app", limit=2))

    assert [item.symbol for item in result.items] == ["APP", "AAPL"]
    assert result.total == 2
    assert result.source == "static"
    assert result.as_of == datetime(2026, 8, 9, tzinfo=UTC)


def test_symbol_directory_uses_safe_stale_cache_when_refresh_fails() -> None:
    class FailingSource(StaticSymbolDirectorySource):
        async def load(self):
            raise RuntimeError("SEC unavailable")

    cache = MemoryCache(clock=lambda: 1000.0)
    healthy = SymbolSearchService(
        StaticSymbolDirectorySource(
            records=(SymbolRecord(symbol="MSFT", name="Microsoft Corporation", exchange="Nasdaq"),),
            as_of=datetime(2026, 8, 9, tzinfo=UTC),
        ),
        cache,
        directory_ttl_seconds=1,
        stale_seconds=60,
    )
    run(healthy.refresh_directory())
    assert run(healthy.search_symbols("ms", limit=8)).items[0].symbol == "MSFT"

    failing = SymbolSearchService(
        FailingSource(records=(), as_of=datetime(2026, 8, 9, tzinfo=UTC)),
        cache,
        directory_ttl_seconds=1,
        stale_seconds=60,
    )
    with pytest.raises(CacheUnavailableError, match="refresh failed"):
        run(failing.refresh_directory())
    assert run(failing.search_symbols("micro", limit=8)).items[0].symbol == "MSFT"


def test_symbol_record_rejects_xss_payloads() -> None:
    with pytest.raises(ValidationError):
        SymbolRecord(symbol="BAD", name="<img src=x onerror=alert(1)>", exchange="NYSE")
