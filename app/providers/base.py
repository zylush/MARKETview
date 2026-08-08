from __future__ import annotations

from datetime import date
from typing import Protocol

from app.models import EODBar, Page


class MarketDataProvider(Protocol):
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
