from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from app.research.control import Reservation
from app.research.deadline import RequestDeadline


class MarketDataDashboardProtocol(Protocol):
    @property
    def last_metadata(self) -> object | None: ...

    async def latest_eod(self, symbol: str) -> Any: ...

    async def history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
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


class ResearchQueryProtocol(Protocol):
    async def authorize_research_reservation(
        self,
        principal_digest: str,
        *,
        daily_limit: int,
        window_seconds: int,
        deadline: RequestDeadline | None = None,
    ) -> Reservation | None: ...

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        reservation: Reservation | None = None,
        deadline: RequestDeadline | None = None,
    ) -> Any: ...

    async def release_research_reservation(
        self,
        reservation: Reservation,
        *,
        deadline: RequestDeadline | None = None,
    ) -> bool: ...


class DashboardService:
    """Compose market data, symbol lookup, and research services behind one route object."""

    def __init__(
        self,
        market_data: MarketDataDashboardProtocol,
        symbol_search: SymbolSearchProtocol,
        research: ResearchQueryProtocol,
    ) -> None:
        self._market_data = market_data
        self._symbol_search = symbol_search
        self._research = research

    @property
    def last_metadata(self) -> object | None:
        return getattr(self._market_data, "last_metadata", None)

    async def latest_eod(self, symbol: str) -> Any:
        return await self._market_data.latest_eod(symbol)

    async def history(
        self,
        symbol: str,
        start: date,
        end: date,
        **params: Any,
    ) -> Any:
        return await self._market_data.history(symbol, start, end, **params)

    async def eod_history(self, symbol: str, **params: Any) -> Any:
        return await self._market_data.eod_history(symbol, **params)

    async def usage(self) -> Any:
        return await self._market_data.usage()

    async def search_symbols(self, query: str, *, limit: int = 8) -> Any:
        return await self._symbol_search.search_symbols(query, limit=limit)

    async def authorize_research_reservation(
        self,
        principal_digest: str,
        *,
        daily_limit: int,
        window_seconds: int,
        deadline: RequestDeadline | None = None,
    ) -> Reservation | None:
        return await self._research.authorize_research_reservation(
            principal_digest,
            daily_limit=daily_limit,
            window_seconds=window_seconds,
            deadline=deadline,
        )

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        reservation: Reservation | None = None,
        deadline: RequestDeadline | None = None,
    ) -> Any:
        return await self._research.query_research(
            symbol,
            question,
            reservation=reservation,
            deadline=deadline,
        )

    async def release_research_reservation(
        self,
        reservation: Reservation,
        *,
        deadline: RequestDeadline | None = None,
    ) -> bool:
        return await self._research.release_research_reservation(
            reservation,
            deadline=deadline,
        )

    async def aclose(self) -> None:
        configured = (self._market_data, self._symbol_search, self._research)
        components = tuple(
            component
            for index, component in enumerate(configured)
            if all(component is not previous for previous in configured[:index])
        )
        for component in components:
            close = getattr(component, "aclose", None)
            if close is not None:
                await close()

    def manages(self, resource: object) -> bool:
        components = (self._market_data, self._symbol_search, self._research)
        return any(
            bool(manages and manages(resource))
            for component in components
            if (manages := getattr(component, "manages", None)) is not None
        )
