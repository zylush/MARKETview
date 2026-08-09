from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class EODBar(DomainModel):
    symbol: str
    date: date
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    close: Decimal | None = None
    volume: int | None = Field(default=None, ge=0)
    adjusted_open: Decimal | None = None
    adjusted_high: Decimal | None = None
    adjusted_low: Decimal | None = None
    adjusted_close: Decimal | None = None
    adjusted_volume: int | None = Field(default=None, ge=0)

    @field_validator("date", mode="before")
    @classmethod
    def normalize_date(cls, value: object) -> object:
        if isinstance(value, str) and len(value) >= 10:
            return value[:10]
        return value


class Usage(DomainModel):
    requests_used: int = Field(ge=0)
    requests_limit: int | None = Field(default=None, ge=0)
    requests_remaining: int | None = Field(default=None, ge=0)
    reset_at: datetime


class Page[ItemT](DomainModel):
    items: tuple[ItemT, ...] = ()
    next_cursor: str | None = None
    total: int | None = Field(default=None, ge=0)


class SymbolRecord(DomainModel):
    symbol: str = Field(min_length=1, max_length=32, pattern=r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")
    name: str = Field(min_length=1, max_length=200)
    exchange: str = Field(min_length=1, max_length=80)

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("name", "exchange")
    @classmethod
    def reject_markup(cls, value: str) -> str:
        if any(character in value for character in "<>"):
            raise ValueError("symbol directory text must not contain markup")
        return value.strip()


class SymbolSearchPage(Page[SymbolRecord]):
    source: str
    as_of: datetime
    stale: bool = False
    limit: int = Field(ge=1, le=8)
