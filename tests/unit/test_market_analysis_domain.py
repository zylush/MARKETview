from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.market_analysis.domain import MarketIntent, MarketPeriod
from app.market_analysis.evidence import (
    EvidenceQualityError,
    MarketEvidenceBundle,
    MarketFact,
    build_period_bundle,
    build_snapshot_bundle,
    calculate_period,
    deterministic_market_answer,
    normalize_market_bars,
    require_fresh_snapshot,
    require_period_coverage,
)
from app.market_analysis.planning import plan_market_research
from app.models import EODBar

NOW = date(2026, 8, 10)
OBSERVED = datetime(2026, 8, 10, 12, tzinfo=UTC)


def bar(day: date, close: str, **changes: object) -> EODBar:
    values: dict[str, object] = {
        "symbol": "AAPL",
        "date": day,
        "open": close,
        "high": str(Decimal(close) + 1),
        "low": str(Decimal(close) - 1),
        "close": close,
        "volume": 100,
    }
    values.update(changes)
    return EODBar(**values)


@pytest.mark.unit
def test_plans_snapshot_and_default_thirty_calendar_day_trend() -> None:
    snapshot = plan_market_research("What is the latest close?", "aapl", current_date=NOW)
    trend = plan_market_research("How has the stock performed?", "AAPL", current_date=NOW)

    assert snapshot.market_intent is MarketIntent.SNAPSHOT
    assert snapshot.periods == ()
    assert trend.market_intent is MarketIntent.TREND
    assert trend.periods == (MarketPeriod(start=date(2026, 7, 11), end=NOW),)
    assert trend.history_limit == 366


@pytest.mark.unit
def test_plans_explicit_range_and_comparison() -> None:
    trend = plan_market_research(
        "AAPL trend from 2026-01-01 to 2026-03-31", "AAPL", current_date=NOW
    )
    comparison = plan_market_research(
        "Compare 2026-01-01 to 2026-01-31 vs 2026-02-01 to 2026-02-28",
        "AAPL",
        current_date=NOW,
    )

    assert trend.periods == (MarketPeriod(start=date(2026, 1, 1), end=date(2026, 3, 31)),)
    assert comparison.market_intent is MarketIntent.COMPARISON
    assert comparison.periods == (
        MarketPeriod(start=date(2026, 1, 1), end=date(2026, 1, 31)),
        MarketPeriod(start=date(2026, 2, 1), end=date(2026, 2, 28)),
    )


@pytest.mark.unit
def test_rejects_ranges_over_365_days_or_in_the_future() -> None:
    with pytest.raises(ValueError, match="365"):
        plan_market_research("trend from 2025-01-01 to 2026-01-02", "AAPL", current_date=NOW)
    with pytest.raises(ValueError, match="future"):
        plan_market_research("trend from 2026-08-01 to 2026-08-11", "AAPL", current_date=NOW)


@pytest.mark.unit
def test_normalizes_sorts_and_deduplicates_identical_bars() -> None:
    first = bar(date(2026, 8, 7), "100")
    second = bar(date(2026, 8, 8), "101")

    normalized = normalize_market_bars((second, first, first), "aapl", current_date=NOW)

    assert tuple(item.observation_date for item in normalized) == (
        date(2026, 8, 7),
        date(2026, 8, 8),
    )
    assert normalized[0].close == Decimal("100")


@pytest.mark.unit
@pytest.mark.parametrize(
    "records",
    [
        (bar(date(2026, 8, 7), "100"), bar(date(2026, 8, 7), "101")),
        (bar(date(2026, 8, 11), "100"),),
        (bar(date(2026, 8, 7), "100", symbol="MSFT"),),
        (bar(date(2026, 8, 7), "0"),),
        (bar(date(2026, 8, 7), "100", high="98"),),
    ],
)
def test_rejects_conflicts_future_wrong_symbol_and_invalid_values(
    records: tuple[EODBar, ...],
) -> None:
    with pytest.raises(EvidenceQualityError):
        normalize_market_bars(records, "AAPL", current_date=NOW)


@pytest.mark.unit
def test_calculation_uses_adjusted_prices_only_when_consistent() -> None:
    period = MarketPeriod(start=date(2026, 8, 3), end=date(2026, 8, 7))
    complete_adjusted = tuple(
        bar(
            date(2026, 8, day),
            str(100 + day),
            adjusted_close=str(200 + day),
            adjusted_high=str(201 + day),
            adjusted_low=str(199 + day),
            adjusted_volume=200,
        )
        for day in range(3, 8)
    )
    mixed = tuple(
        item if index else item.model_copy(update={"adjusted_close": None})
        for index, item in enumerate(complete_adjusted)
    )

    adjusted = calculate_period(
        normalize_market_bars(complete_adjusted, "AAPL", current_date=NOW), period
    )
    unadjusted = calculate_period(normalize_market_bars(mixed, "AAPL", current_date=NOW), period)

    assert adjusted.price_basis == "adjusted"
    assert adjusted.start_close == Decimal("203")
    assert unadjusted.price_basis == "unadjusted"
    assert unadjusted.start_close == Decimal("103")


@pytest.mark.unit
def test_calculates_exact_decimal_metrics() -> None:
    period = MarketPeriod(start=date(2026, 8, 3), end=date(2026, 8, 5))
    normalized = normalize_market_bars(
        (
            bar(date(2026, 8, 3), "3", high="4", low="2", volume=100),
            bar(date(2026, 8, 4), "4", high="6", low="3", volume=200),
            bar(date(2026, 8, 5), "5", high="5", low="4", volume=300),
        ),
        "AAPL",
        current_date=NOW,
    )

    result = calculate_period(normalized, period)

    assert result.absolute_change == Decimal("2")
    assert result.percentage_change == Decimal("66.66666666666666666666666667")
    assert result.high == Decimal("6")
    assert result.low == Decimal("2")
    assert result.average_volume == Decimal("200")


@pytest.mark.unit
def test_snapshot_requires_close_within_seven_calendar_days() -> None:
    fresh = normalize_market_bars((bar(date(2026, 8, 3), "100"),), "AAPL", current_date=NOW)
    stale = normalize_market_bars((bar(date(2026, 8, 2), "100"),), "AAPL", current_date=NOW)

    assert require_fresh_snapshot(fresh[0], current_date=NOW).close == Decimal("100")
    with pytest.raises(EvidenceQualityError, match="fresh"):
        require_fresh_snapshot(stale[0], current_date=NOW)


@pytest.mark.unit
def test_period_coverage_requires_boundaries_and_density() -> None:
    period = MarketPeriod(start=date(2026, 8, 3), end=date(2026, 8, 14))  # 10 weekdays; minimum 6
    sufficient = normalize_market_bars(
        tuple(bar(date(2026, 8, day), str(day)) for day in (3, 4, 5, 6, 7, 14)),
        "AAPL",
        current_date=date(2026, 8, 14),
    )
    sparse = sufficient[:-1]

    assert require_period_coverage(sufficient, period) == sufficient
    with pytest.raises(EvidenceQualityError, match="coverage"):
        require_period_coverage(sparse, period)


@pytest.mark.unit
def test_evidence_bundle_and_nested_values_are_immutable() -> None:
    fact = MarketFact.create(
        symbol="aapl",
        observation_date=date(2026, 8, 7),
        provider="Market Data",
        provider_timestamp=OBSERVED,
        field_name="close",
        value=Decimal("100"),
        unit="USD/share",
    )
    bundle = MarketEvidenceBundle(
        symbol="AAPL",
        provider="Market Data",
        as_of=OBSERVED,
        periods=(),
        evidence=(fact,),
        calculations=(),
    )

    with pytest.raises(ValidationError):
        fact.value = Decimal("101")  # type: ignore[misc]
    with pytest.raises(ValidationError):
        bundle.market_facts = ()  # type: ignore[misc]
    assert fact.symbol == "AAPL"
    assert fact.provider_timestamp == OBSERVED


@pytest.mark.unit
def test_snapshot_bundle_builds_locked_deterministic_answer_and_json() -> None:
    normalized = normalize_market_bars((bar(date(2026, 8, 7), "100"),), "AAPL", current_date=NOW)

    bundle = build_snapshot_bundle(
        normalized[0],
        provider="Market Data",
        provider_timestamp=OBSERVED,
        current_date=NOW,
    )
    answer = deterministic_market_answer(bundle)

    assert answer.answer == ("Market Data reports AAPL's latest EOD close as 100 for 2026-08-07.")
    assert answer.evidence == bundle.evidence
    assert answer.period_start == date(2026, 8, 7)
    assert answer.period_end == date(2026, 8, 7)
    assert answer.evidence_count == 1
    assert answer.disclaimer == (
        "AI-assisted analysis is informational only, not investment advice."
    )
    assert bundle.model_dump(mode="json")["evidence"][0]["value"] == "100"


@pytest.mark.unit
def test_period_bundle_cites_and_verifies_all_decimal_calculations() -> None:
    period = MarketPeriod(start=date(2026, 8, 3), end=date(2026, 8, 7))
    normalized = normalize_market_bars(
        tuple(bar(date(2026, 8, day), str(day)) for day in range(3, 8)),
        "AAPL",
        current_date=NOW,
    )

    bundle = build_period_bundle(
        normalized,
        (period,),
        symbol="AAPL",
        provider="Market Data",
        provider_timestamp=OBSERVED,
    )

    assert {item.operation for item in bundle.calculations} == {
        "absolute_change",
        "percentage_change",
        "period_high",
        "period_low",
        "average_volume",
    }
    assert all(item.input_evidence_ids for item in bundle.calculations)
    assert deterministic_market_answer(bundle).calculations == bundle.calculations
