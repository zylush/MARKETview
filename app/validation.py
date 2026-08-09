from __future__ import annotations

import re
from datetime import date

from app.errors import InputValidationError

_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")
_SYMBOL_QUERY = re.compile(r"^[A-Z0-9.\- ]{2,32}$")
_CURSOR = re.compile(r"^\d{1,12}$")


def validate_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not _SYMBOL.fullmatch(normalized):
        raise InputValidationError("symbol must contain 1-32 market identifier characters")
    return normalized


def validate_symbol_query(query: str) -> str:
    normalized = query.strip().upper() if isinstance(query, str) else ""
    if len(normalized) < 2:
        raise InputValidationError("symbol search query must contain at least two characters")
    if not _SYMBOL_QUERY.fullmatch(normalized):
        raise InputValidationError(
            "symbol search query must use letters, numbers, spaces, periods, or hyphens"
        )
    return normalized


def validate_research_question(question: str) -> str:
    normalized = question.strip() if isinstance(question, str) else ""
    if not 1 <= len(normalized) <= 500:
        raise InputValidationError("research question must contain 1-500 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise InputValidationError("research question must not contain control characters")
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
