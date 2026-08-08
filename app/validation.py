from __future__ import annotations

import re
from datetime import date

from app.errors import InputValidationError

_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")
_CURSOR = re.compile(r"^\d{1,12}$")


def validate_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not _SYMBOL.fullmatch(normalized):
        raise InputValidationError("symbol must contain 1-32 market identifier characters")
    return normalized


def validate_limit(limit: int, *, maximum: int = 1000) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
        raise InputValidationError(f"limit must be between 1 and {maximum}")
    return limit


def validate_cursor(cursor: str | int | None) -> str | None:
    if cursor is None:
        return None
    normalized = str(cursor).strip()
    if not _CURSOR.fullmatch(normalized):
        raise InputValidationError("cursor must be a non-negative numeric offset")
    return normalized


def validate_date_range(start_date: date, end_date: date) -> tuple[date, date]:
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise InputValidationError("date range values must be dates")
    if start_date > end_date:
        raise InputValidationError("start date must not be after end date")
    if (end_date - start_date).days > 365:
        raise InputValidationError("date range cannot exceed one year")
    return start_date, end_date
