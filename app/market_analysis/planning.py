from __future__ import annotations

import re
from datetime import date, timedelta

from app.market_analysis.domain import MarketIntent, MarketPeriod, ResearchPlan

_ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_LAST_DAYS = re.compile(r"\b(?:last|past|previous)\s+(\d{1,3})\s+(?:calendar\s+)?days?\b", re.I)
_SNAPSHOT = re.compile(r"\b(?:latest|current|today(?:'s)?|close|closing|price|quote|eod)\b", re.I)
_TREND = re.compile(
    r"\b(?:trend|perform(?:ed|ance)?|gain(?:ed)?|loss|change(?:d)?|return|over time)\b", re.I
)
_COMPARISON = re.compile(r"\b(?:compare|comparison|versus|vs\.?)\b", re.I)


def _dates(question: str) -> tuple[date, ...]:
    try:
        return tuple(date.fromisoformat(item) for item in _ISO_DATE.findall(question))
    except ValueError:
        raise ValueError("explicit market dates must use valid ISO dates") from None


def plan_market_research(
    question: str,
    symbol: str,
    *,
    current_date: date,
) -> ResearchPlan:
    """Create a bounded immutable market plan without any external calls."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("market research question must not be empty")
    explicit_dates = _dates(question)
    periods: tuple[MarketPeriod, ...]
    if len(explicit_dates) not in {0, 2, 4}:
        raise ValueError("market research requires complete period boundaries")
    if len(explicit_dates) == 4:
        intent = MarketIntent.COMPARISON
        periods = (
            MarketPeriod(start=explicit_dates[0], end=explicit_dates[1]),
            MarketPeriod(start=explicit_dates[2], end=explicit_dates[3]),
        )
    elif len(explicit_dates) == 2:
        intent = MarketIntent.TREND
        periods = (MarketPeriod(start=explicit_dates[0], end=explicit_dates[1]),)
    else:
        relative = _LAST_DAYS.search(question)
        if relative:
            duration = int(relative.group(1))
            if not 1 <= duration <= 365:
                raise ValueError("market period must be between 1 and 365 calendar days")
            intent = MarketIntent.TREND
            periods = (
                MarketPeriod(start=current_date - timedelta(days=duration), end=current_date),
            )
        elif _COMPARISON.search(question):
            raise ValueError("comparison requires two explicit market periods")
        elif _TREND.search(question):
            intent = MarketIntent.TREND
            periods = (MarketPeriod(start=current_date - timedelta(days=30), end=current_date),)
        elif _SNAPSHOT.search(question):
            intent = MarketIntent.SNAPSHOT
            periods = ()
        else:
            intent = MarketIntent.NONE
            periods = ()
    return ResearchPlan(
        symbol=symbol,
        market_intent=intent,
        current_date=current_date,
        periods=periods,
    )
