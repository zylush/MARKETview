from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, Protocol

from app.market_analysis.domain import MarketAnalysisAnswer


class MarketDataDashboardProtocol(Protocol):
    @property
    def last_metadata(self) -> object | None: ...

    async def latest_eod(self, symbol: str) -> Any: ...
    async def history(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        limit: int = 100,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> Any: ...

    async def eod_history(
        self,
        symbol: str,
        *,
        date_from: date,
        date_to: date,
        limit: int = 100,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> Any: ...
    async def usage(self) -> Any: ...


class SymbolSearchProtocol(Protocol):
    async def search_symbols(self, query: str, *, limit: int = 8) -> Any: ...


class MarketAnalysisProtocol(Protocol):
    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        before_generation: Callable[[], Awaitable[None]] | None = None,
    ) -> MarketAnalysisAnswer: ...


class DashboardService:
    """Expose market data, symbol lookup, and AI market analysis as one route facade."""

    def __init__(
        self,
        market_data: MarketDataDashboardProtocol,
        symbol_search: SymbolSearchProtocol,
        market_analysis: MarketAnalysisProtocol,
    ) -> None:
        self._market_data = market_data
        self._symbol_search = symbol_search
        self._market_analysis = market_analysis

    @property
    def last_metadata(self) -> object | None:
        return getattr(self._market_data, "last_metadata", None)

    async def latest_eod(self, symbol: str) -> Any:
        return await self._market_data.latest_eod(symbol)

    async def history(self, symbol: str, start: date, end: date, **params: Any) -> Any:
        return await self._market_data.history(symbol, start, end, **params)

    async def eod_history(self, symbol: str, **params: Any) -> Any:
        return await self._market_data.eod_history(symbol, **params)

    async def usage(self) -> Any:
        return await self._market_data.usage()

    async def search_symbols(self, query: str, *, limit: int = 8) -> Any:
        return await self._symbol_search.search_symbols(query, limit=limit)

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        before_generation: Callable[[], Awaitable[None]] | None = None,
    ) -> Any:
        return await self._market_analysis.query_research(
            symbol,
            question,
            before_generation=before_generation,
        )

    async def aclose(self) -> None:
        configured = (self._market_data, self._symbol_search, self._market_analysis)
        components = tuple(
            component
            for index, component in enumerate(configured)
            if all(component is not previous for previous in configured[:index])
        )
        for component in components:
            close = getattr(component, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result

    def manages(self, resource: object) -> bool:
        components = (self._market_data, self._symbol_search, self._market_analysis)
        return any(
            bool(manages and manages(resource))
            for component in components
            if (manages := getattr(component, "manages", None)) is not None
        )
