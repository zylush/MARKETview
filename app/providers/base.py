from __future__ import annotations

from datetime import date
from typing import Protocol

from app.models import Dividend, EODBar, Exchange, Page, Split, Ticker, Usage


class MarketDataProvider(Protocol):
    async def list_tickers(
        self, *, search: str | None = None, limit: int = 100, cursor: str | None = None
    ) -> Page[Ticker]: ...
    async def list_exchanges(
        self, *, search: str | None = None, limit: int = 100, cursor: str | None = None
    ) -> Page[Exchange]: ...
    async def latest_eod(self, symbol: str) -> EODBar: ...
    async def eod_history(
        self,
        symbol: str,
        *,
        start_date: date | str,
        end_date: date | str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[EODBar]: ...
    async def splits(
        self,
        symbol: str | None = None,
        *,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[Split]: ...
    async def dividends(
        self,
        symbol: str | None = None,
        *,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Page[Dividend]: ...
    async def usage(self) -> Usage: ...
