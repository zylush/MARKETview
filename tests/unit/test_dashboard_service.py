from __future__ import annotations

from datetime import date

import pytest

from app.services.dashboard import DashboardService


class FacadeStub:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.last_metadata = {"source": "cache"}

    async def latest_eod(self, symbol: str) -> str:
        self.calls.append(("latest", symbol))
        return "latest"

    async def history(self, symbol: str, start: date, end: date, **params: object) -> str:
        self.calls.append(("history", symbol, start, end, params))
        return "history"

    async def eod_history(self, symbol: str, **params: object) -> str:
        self.calls.append(("eod_history", symbol, params))
        return "eod_history"

    async def usage(self) -> str:
        self.calls.append(("usage",))
        return "usage"

    async def search_symbols(self, query: str, *, limit: int) -> str:
        self.calls.append(("symbols", query, limit))
        return "symbols"

    async def query_research(self, symbol: str, question: str, **kwargs: object) -> str:
        del kwargs
        self.calls.append(("query", symbol, question))
        return "analysis"

    async def aclose(self) -> None:
        self.calls.append(("close",))

    def manages(self, resource: object) -> bool:
        return resource is self


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dashboard_keeps_market_symbol_and_query_facade() -> None:
    market = FacadeStub()
    symbols = FacadeStub()
    analysis = FacadeStub()
    dashboard = DashboardService(market, symbols, analysis)

    assert dashboard.last_metadata == {"source": "cache"}
    assert await dashboard.latest_eod("AAPL") == "latest"
    assert await dashboard.history("AAPL", date(2026, 1, 1), date(2026, 1, 2), limit=5) == "history"
    assert (
        await dashboard.eod_history("AAPL", date_from=date(2026, 1, 1), date_to=date(2026, 1, 2))
        == "eod_history"
    )
    assert await dashboard.usage() == "usage"
    assert await dashboard.search_symbols("apple", limit=3) == "symbols"
    assert await dashboard.query_research("aapl", "latest close") == "analysis"
    assert analysis.calls == [("query", "aapl", "latest close")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dashboard_closes_each_unique_component_once_and_reports_ownership() -> None:
    shared = FacadeStub()
    analysis = FacadeStub()
    dashboard = DashboardService(shared, shared, analysis)

    assert dashboard.manages(shared) is True
    assert dashboard.manages(object()) is False
    await dashboard.aclose()

    assert shared.calls == [("close",)]
    assert analysis.calls == [("close",)]
