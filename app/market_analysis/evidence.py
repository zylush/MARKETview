from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, cast

from app.market_analysis.domain import (
    LockedMarketStatement,
    MarketAnalysisAnswer,
    MarketAnswerStatus,
    MarketEvidence,
    MarketEvidenceBundle,
    MarketFact,
    MarketPeriod,
    NormalizedMarketBar,
    PeriodMetrics,
    VerifiedCalculation,
)


class EvidenceQualityError(ValueError):
    """A successful provider payload cannot safely support an answer."""


def _value(record: object, field: str) -> Any:
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _decimal(value: object, *, required: bool, positive: bool = True) -> Decimal | None:
    if value is None and not required:
        return None
    if isinstance(value, bool):
        raise EvidenceQualityError("market evidence contains an invalid value")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception:
        raise EvidenceQualityError("market evidence contains an invalid value") from None
    if not result.is_finite() or (positive and result <= 0):
        raise EvidenceQualityError("market evidence contains an invalid value")
    return result


def _volume(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise EvidenceQualityError("market evidence contains an invalid volume")
    try:
        decimal = Decimal(str(value))
    except Exception:
        raise EvidenceQualityError("market evidence contains an invalid volume") from None
    if not decimal.is_finite() or decimal < 0 or decimal != decimal.to_integral_value():
        raise EvidenceQualityError("market evidence contains an invalid volume")
    return int(decimal)


def _normalize_one(record: object, symbol: str, current_date: date) -> NormalizedMarketBar:
    record_symbol = _value(record, "symbol")
    observation_date = _value(record, "date")
    if not isinstance(record_symbol, str) or record_symbol.strip().upper() != symbol:
        raise EvidenceQualityError("market evidence contains a wrong symbol")
    if type(observation_date) is not date or observation_date > current_date:
        raise EvidenceQualityError("market evidence contains an invalid or future date")
    price_fields = (
        "open",
        "high",
        "low",
        "close",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    )
    values = {
        name: _decimal(_value(record, name), required=name == "close") for name in price_fields
    }
    high = values["high"]
    low = values["low"]
    close = values["close"]
    open_value = values["open"]
    if (
        high is not None
        and low is not None
        and (
            high < low
            or (close is not None and high < close)
            or (open_value is not None and high < open_value)
            or (close is not None and low > close)
            or (open_value is not None and low > open_value)
        )
    ):
        raise EvidenceQualityError("market evidence contains inconsistent OHLC values")
    adjusted_high = values["adjusted_high"]
    adjusted_low = values["adjusted_low"]
    adjusted_close = values["adjusted_close"]
    if (
        adjusted_high is not None
        and adjusted_low is not None
        and (
            adjusted_high < adjusted_low
            or (adjusted_close is not None and adjusted_high < adjusted_close)
            or (adjusted_close is not None and adjusted_low > adjusted_close)
        )
    ):
        raise EvidenceQualityError("market evidence contains inconsistent adjusted values")
    return NormalizedMarketBar(
        symbol=symbol,
        observation_date=observation_date,
        open=values["open"],
        high=values["high"],
        low=values["low"],
        close=cast(Decimal, values["close"]),
        volume=_volume(_value(record, "volume")),
        adjusted_open=values["adjusted_open"],
        adjusted_high=values["adjusted_high"],
        adjusted_low=values["adjusted_low"],
        adjusted_close=values["adjusted_close"],
        adjusted_volume=_volume(_value(record, "adjusted_volume")),
    )


def normalize_market_bars(
    records: Iterable[object], symbol: str, *, current_date: date
) -> tuple[NormalizedMarketBar, ...]:
    checked_symbol = symbol.strip().upper()
    by_date: dict[date, NormalizedMarketBar] = {}
    for record in records:
        normalized = _normalize_one(record, checked_symbol, current_date)
        previous = by_date.get(normalized.observation_date)
        if previous is not None and previous != normalized:
            raise EvidenceQualityError("market evidence contains conflicting duplicates")
        by_date[normalized.observation_date] = normalized
    return tuple(by_date[item] for item in sorted(by_date))


def require_fresh_snapshot(bar: NormalizedMarketBar, *, current_date: date) -> NormalizedMarketBar:
    if not 0 <= (current_date - bar.observation_date).days <= 7:
        raise EvidenceQualityError("market snapshot is not fresh enough")
    return bar


def _weekdays(period: MarketPeriod) -> int:
    return sum(
        1
        for offset in range((period.end - period.start).days + 1)
        if (period.start.fromordinal(period.start.toordinal() + offset)).weekday() < 5
    )


def required_period_closes(period: MarketPeriod) -> int:
    weekdays = _weekdays(period)
    sixty_percent_ceiling = (weekdays * 60 + 99) // 100
    return min(10, max(2, sixty_percent_ceiling))


def require_period_coverage(
    bars: tuple[NormalizedMarketBar, ...], period: MarketPeriod
) -> tuple[NormalizedMarketBar, ...]:
    selected = tuple(bar for bar in bars if period.start <= bar.observation_date <= period.end)
    if (
        len(selected) < required_period_closes(period)
        or not selected
        or (selected[0].observation_date - period.start).days > 7
        or (period.end - selected[-1].observation_date).days > 7
    ):
        raise EvidenceQualityError("historical market evidence has insufficient coverage")
    return selected


def calculate_period(bars: tuple[NormalizedMarketBar, ...], period: MarketPeriod) -> PeriodMetrics:
    selected = tuple(bar for bar in bars if period.start <= bar.observation_date <= period.end)
    if not selected:
        raise EvidenceQualityError("historical market evidence has no usable closes")
    use_adjusted = all(
        bar.adjusted_close is not None
        and bar.adjusted_high is not None
        and bar.adjusted_low is not None
        and (bar.volume is None or bar.adjusted_volume is not None)
        for bar in selected
    )
    if use_adjusted:
        closes = tuple(bar.adjusted_close for bar in selected)
        highs = tuple(bar.adjusted_high for bar in selected)
        lows = tuple(bar.adjusted_low for bar in selected)
        volumes = tuple(bar.adjusted_volume for bar in selected)
    else:
        closes = tuple(bar.close for bar in selected)
        highs = tuple(bar.high for bar in selected)
        lows = tuple(bar.low for bar in selected)
        volumes = tuple(bar.volume for bar in selected)
    if any(value is None for value in (*closes, *highs, *lows)):
        raise EvidenceQualityError("historical market evidence is missing required price values")
    numeric_closes = tuple(value for value in closes if value is not None)
    numeric_highs = tuple(value for value in highs if value is not None)
    numeric_lows = tuple(value for value in lows if value is not None)
    start_close, end_close = numeric_closes[0], numeric_closes[-1]
    absolute_change = end_close - start_close
    percentage_change = absolute_change / start_close * Decimal(100)
    average_volume = None
    if all(value is not None for value in volumes):
        numeric_volumes = tuple(Decimal(value) for value in volumes if value is not None)
        average_volume = sum(numeric_volumes, Decimal(0)) / Decimal(len(numeric_volumes))
    return PeriodMetrics(
        period=period,
        price_basis="adjusted" if use_adjusted else "unadjusted",
        start_date=selected[0].observation_date,
        end_date=selected[-1].observation_date,
        start_close=start_close,
        end_close=end_close,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        high=max(numeric_highs),
        low=min(numeric_lows),
        average_volume=average_volume,
    )


def _calculation(
    operation: str,
    value: Decimal,
    unit: str,
    period: MarketPeriod,
    inputs: tuple[MarketEvidence, ...],
) -> VerifiedCalculation:
    material = "\x1f".join(
        (
            operation,
            str(value),
            unit,
            period.start.isoformat(),
            period.end.isoformat(),
            *(item.evidence_id for item in inputs),
        )
    )
    return VerifiedCalculation(
        calculation_id="calc-" + hashlib.sha256(material.encode()).hexdigest(),
        operation=operation,
        value=value,
        unit=unit,
        period_start=period.start,
        period_end=period.end,
        input_evidence_ids=tuple(item.evidence_id for item in inputs),
        input_values=tuple(item.value for item in inputs),
    )


def _statement(
    text: str,
    *,
    evidence_ids: tuple[str, ...] = (),
    calculation_ids: tuple[str, ...] = (),
) -> LockedMarketStatement:
    material = "\x1f".join((text, *evidence_ids, *calculation_ids))
    return LockedMarketStatement(
        statement_id="statement-" + hashlib.sha256(material.encode()).hexdigest(),
        text=text,
        evidence_ids=evidence_ids,
        calculation_ids=calculation_ids,
    )


def build_snapshot_bundle(
    bar: NormalizedMarketBar,
    *,
    provider: str,
    provider_timestamp: datetime,
    current_date: date,
) -> MarketEvidenceBundle:
    checked = require_fresh_snapshot(bar, current_date=current_date)
    fact = MarketEvidence.create(
        symbol=checked.symbol,
        observation_date=checked.observation_date,
        provider=provider,
        provider_timestamp=provider_timestamp,
        field_name="close",
        value=checked.close,
        unit="USD/share",
    )
    text = (
        f"{provider} reports {checked.symbol}'s latest EOD close as {checked.close} "
        f"for {checked.observation_date.isoformat()}."
    )
    return MarketEvidenceBundle(
        symbol=checked.symbol,
        provider=provider,
        as_of=provider_timestamp.astimezone(UTC),
        periods=(),
        evidence=(fact,),
        calculations=(),
        locked_statements=(_statement(text, evidence_ids=(fact.evidence_id,)),),
    )


def build_period_bundle(
    bars: tuple[NormalizedMarketBar, ...],
    periods: tuple[MarketPeriod, ...],
    *,
    symbol: str,
    provider: str,
    provider_timestamp: datetime,
) -> MarketEvidenceBundle:
    if not periods or len(periods) > 2:
        raise ValueError("historical market bundle requires one or two periods")
    all_evidence: list[MarketEvidence] = []
    all_calculations: list[VerifiedCalculation] = []
    statements: list[LockedMarketStatement] = []
    evidence_by_id: dict[str, MarketEvidence] = {}
    for period in periods:
        selected = require_period_coverage(bars, period)
        metrics = calculate_period(selected, period)
        use_adjusted = metrics.price_basis == "adjusted"

        def fact(
            bar: NormalizedMarketBar,
            field: str,
            value: Decimal,
            unit: str,
            *,
            adjusted: bool = use_adjusted,
        ) -> MarketEvidence:
            item = MarketEvidence.create(
                symbol=symbol,
                observation_date=bar.observation_date,
                provider=provider,
                provider_timestamp=provider_timestamp,
                field_name=("adjusted_" if adjusted else "") + field,
                value=value,
                unit=unit,
            )
            evidence_by_id.setdefault(item.evidence_id, item)
            return item

        start_fact = fact(selected[0], "close", metrics.start_close, "USD/share")
        end_fact = fact(selected[-1], "close", metrics.end_close, "USD/share")
        close_inputs = (start_fact, end_fact)
        absolute = _calculation(
            "absolute_change", metrics.absolute_change, "USD/share", period, close_inputs
        )
        percentage = _calculation(
            "percentage_change", metrics.percentage_change, "percent", period, close_inputs
        )
        high_inputs = tuple(
            fact(
                bar,
                "high",
                (bar.adjusted_high if use_adjusted else bar.high),  # type: ignore[arg-type]
                "USD/share",
            )
            for bar in selected
        )
        low_inputs = tuple(
            fact(
                bar,
                "low",
                (bar.adjusted_low if use_adjusted else bar.low),  # type: ignore[arg-type]
                "USD/share",
            )
            for bar in selected
        )
        calculations = [
            absolute,
            percentage,
            _calculation("period_high", metrics.high, "USD/share", period, high_inputs),
            _calculation("period_low", metrics.low, "USD/share", period, low_inputs),
        ]
        if metrics.average_volume is not None:
            volume_inputs = tuple(
                fact(
                    bar,
                    "volume",
                    Decimal(bar.adjusted_volume if use_adjusted else bar.volume),  # type: ignore[arg-type]
                    "shares",
                )
                for bar in selected
            )
            calculations.append(
                _calculation(
                    "average_volume", metrics.average_volume, "shares", period, volume_inputs
                )
            )
        all_calculations.extend(calculations)
        text = (
            f"From {metrics.start_date.isoformat()} to {metrics.end_date.isoformat()}, "
            f"{symbol} changed by {metrics.absolute_change} USD/share "
            f"({metrics.percentage_change}%), from {metrics.start_close} to "
            f"{metrics.end_close}, using {metrics.price_basis} prices."
        )
        statements.append(
            _statement(
                text,
                evidence_ids=tuple(item.evidence_id for item in close_inputs),
                calculation_ids=(absolute.calculation_id, percentage.calculation_id),
            )
        )
    all_evidence.extend(evidence_by_id.values())
    return MarketEvidenceBundle(
        symbol=symbol,
        provider=provider,
        as_of=provider_timestamp.astimezone(UTC),
        periods=periods,
        evidence=tuple(all_evidence),
        calculations=tuple(all_calculations),
        locked_statements=tuple(statements),
    )


def deterministic_market_answer(bundle: MarketEvidenceBundle) -> MarketAnalysisAnswer:
    if not bundle.locked_statements:
        raise EvidenceQualityError("market evidence bundle has no grounded statements")
    return MarketAnalysisAnswer(
        symbol=bundle.symbol,
        status=MarketAnswerStatus.ANSWERED,
        answer=" ".join(item.text for item in bundle.locked_statements),
        provider=bundle.provider,
        as_of=bundle.as_of,
        periods=bundle.periods,
        evidence=bundle.evidence,
        calculations=bundle.calculations,
    )


__all__ = [
    "EvidenceQualityError",
    "MarketEvidence",
    "MarketEvidenceBundle",
    "MarketFact",
    "VerifiedCalculation",
    "build_period_bundle",
    "build_snapshot_bundle",
    "calculate_period",
    "deterministic_market_answer",
    "normalize_market_bars",
    "require_fresh_snapshot",
    "require_period_coverage",
    "required_period_closes",
]
