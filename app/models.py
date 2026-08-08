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
