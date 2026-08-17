from __future__ import annotations

import hashlib
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")
_IDENTIFIER = re.compile(r"^(?:market|calc|statement)-[0-9a-f]{64}$")


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MarketIntent(StrEnum):
    NONE = "none"
    SNAPSHOT = "snapshot"
    TREND = "trend"
    COMPARISON = "comparison"


class MarketAnswerStatus(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNAVAILABLE = "unavailable"


class MarketPeriod(FrozenModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_range(self) -> MarketPeriod:
        if self.start > self.end:
            raise ValueError("market period start must not be after end")
        if (self.end - self.start).days > 365:
            raise ValueError("market period cannot exceed 365 calendar days")
        return self


class ResearchPlan(FrozenModel):
    symbol: str
    market_intent: MarketIntent
    current_date: date
    periods: tuple[MarketPeriod, ...] = ()

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_shape(self) -> ResearchPlan:
        if _SYMBOL.fullmatch(self.symbol) is None:
            raise ValueError("research plan symbol is malformed")
        expected_periods = {
            MarketIntent.NONE: 0,
            MarketIntent.SNAPSHOT: 0,
            MarketIntent.TREND: 1,
            MarketIntent.COMPARISON: 2,
        }[self.market_intent]
        if len(self.periods) != expected_periods:
            raise ValueError("research plan periods do not match market intent")
        if any(period.end > self.current_date for period in self.periods):
            raise ValueError("market period cannot end in the future")
        return self

    @property
    def history_limit(self) -> int | None:
        return 366 if self.market_intent in {MarketIntent.TREND, MarketIntent.COMPARISON} else None


class NormalizedMarketBar(FrozenModel):
    symbol: str
    observation_date: date
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    close: Decimal
    volume: int | None = Field(default=None, ge=0)
    adjusted_open: Decimal | None = None
    adjusted_high: Decimal | None = None
    adjusted_low: Decimal | None = None
    adjusted_close: Decimal | None = None
    adjusted_volume: int | None = Field(default=None, ge=0)


class PeriodMetrics(FrozenModel):
    period: MarketPeriod
    price_basis: str
    start_date: date
    end_date: date
    start_close: Decimal
    end_close: Decimal
    absolute_change: Decimal
    percentage_change: Decimal
    high: Decimal
    low: Decimal
    average_volume: Decimal | None

    @field_validator("price_basis")
    @classmethod
    def validate_basis(cls, value: str) -> str:
        if value not in {"adjusted", "unadjusted"}:
            raise ValueError("price basis is invalid")
        return value


class MarketEvidence(FrozenModel):
    evidence_id: str
    symbol: str
    observation_date: date
    provider: str
    provider_timestamp: datetime
    field_name: str
    value: Decimal
    unit: str

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_evidence(self) -> MarketEvidence:
        if re.fullmatch(r"market-[0-9a-f]{64}", self.evidence_id) is None:
            raise ValueError("market evidence ID is malformed")
        if _SYMBOL.fullmatch(self.symbol) is None:
            raise ValueError("market evidence symbol is malformed")
        if not self.provider.strip() or not self.field_name.strip() or not self.unit.strip():
            raise ValueError("market evidence metadata must not be empty")
        if self.provider_timestamp.tzinfo is None:
            raise ValueError("provider timestamp must be timezone-aware")
        if not self.value.is_finite():
            raise ValueError("market evidence value must be finite")
        return self

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        observation_date: date,
        provider: str,
        provider_timestamp: datetime,
        field_name: str,
        value: Decimal,
        unit: str,
    ) -> MarketEvidence:
        checked_symbol = symbol.strip().upper()
        if provider_timestamp.tzinfo is None:
            raise ValueError("provider timestamp must be timezone-aware")
        observed_at = provider_timestamp.astimezone(UTC)
        material = "\x1f".join(
            (
                checked_symbol,
                observation_date.isoformat(),
                provider,
                observed_at.isoformat(),
                field_name,
                str(value),
                unit,
            )
        )
        return cls(
            evidence_id="market-" + hashlib.sha256(material.encode()).hexdigest(),
            symbol=checked_symbol,
            observation_date=observation_date,
            provider=provider,
            provider_timestamp=observed_at,
            field_name=field_name,
            value=value,
            unit=unit,
        )


MarketFact = MarketEvidence


class VerifiedCalculation(FrozenModel):
    calculation_id: str
    operation: str
    value: Decimal
    unit: str
    period_start: date
    period_end: date
    input_evidence_ids: tuple[str, ...]
    input_values: tuple[Decimal, ...]

    @model_validator(mode="after")
    def validate_calculation(self) -> VerifiedCalculation:
        if re.fullmatch(r"calc-[0-9a-f]{64}", self.calculation_id) is None:
            raise ValueError("calculation ID is malformed")
        if self.operation not in {
            "absolute_change",
            "percentage_change",
            "period_high",
            "period_low",
            "average_volume",
        }:
            raise ValueError("calculation operation is unsupported")
        if self.period_start > self.period_end:
            raise ValueError("calculation period is invalid")
        if not self.input_evidence_ids or len(self.input_evidence_ids) != len(self.input_values):
            raise ValueError("calculation inputs are invalid")
        if len(set(self.input_evidence_ids)) != len(self.input_evidence_ids):
            raise ValueError("calculation inputs must be unique")
        if any(not item.is_finite() for item in (*self.input_values, self.value)):
            raise ValueError("calculation values must be finite")
        expected = {
            "absolute_change": lambda: self.input_values[-1] - self.input_values[0],
            "percentage_change": lambda: (
                (self.input_values[-1] - self.input_values[0]) / self.input_values[0] * Decimal(100)
            ),
            "period_high": lambda: max(self.input_values),
            "period_low": lambda: min(self.input_values),
            "average_volume": lambda: (
                sum(self.input_values, Decimal(0)) / Decimal(len(self.input_values))
            ),
        }[self.operation]()
        if expected != self.value:
            raise ValueError("calculation value does not match its cited numeric inputs")
        return self


class LockedMarketStatement(FrozenModel):
    statement_id: str
    text: str
    evidence_ids: tuple[str, ...] = ()
    calculation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_statement(self) -> LockedMarketStatement:
        if re.fullmatch(r"statement-[0-9a-f]{64}", self.statement_id) is None:
            raise ValueError("statement ID is malformed")
        if not self.text.strip() or (not self.evidence_ids and not self.calculation_ids):
            raise ValueError("locked statement must be grounded")
        return self


class MarketEvidenceBundle(FrozenModel):
    symbol: str
    provider: str
    as_of: datetime
    periods: tuple[MarketPeriod, ...]
    evidence: tuple[MarketEvidence, ...]
    calculations: tuple[VerifiedCalculation, ...]
    locked_statements: tuple[LockedMarketStatement, ...] = ()

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_bundle(self) -> MarketEvidenceBundle:
        if _SYMBOL.fullmatch(self.symbol) is None or not self.provider.strip():
            raise ValueError("market evidence bundle identity is invalid")
        if self.as_of.tzinfo is None:
            raise ValueError("market evidence bundle as-of must be timezone-aware")
        if any(
            item.symbol != self.symbol or item.provider != self.provider for item in self.evidence
        ):
            raise ValueError("market evidence bundle contains foreign evidence")
        evidence_ids = {item.evidence_id for item in self.evidence}
        calculation_ids = {item.calculation_id for item in self.calculations}
        if len(evidence_ids) != len(self.evidence) or len(calculation_ids) != len(
            self.calculations
        ):
            raise ValueError("market evidence bundle IDs must be unique")
        if any(
            not set(item.input_evidence_ids).issubset(evidence_ids) for item in self.calculations
        ):
            raise ValueError("calculation cites unknown market evidence")
        for statement in self.locked_statements:
            if not set(statement.evidence_ids).issubset(evidence_ids) or not set(
                statement.calculation_ids
            ).issubset(calculation_ids):
                raise ValueError("locked statement cites unknown evidence")
        return self

    @property
    def market_facts(self) -> tuple[MarketEvidence, ...]:
        """Compatibility alias for callers using the descriptive evidence name."""
        return self.evidence

    @property
    def statements(self) -> tuple[LockedMarketStatement, ...]:
        return self.locked_statements


class GeneratedMarketAnalysis(FrozenModel):
    status: MarketAnswerStatus
    answer: str
    evidence_ids: tuple[str, ...] = ()
    calculation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_generation(self) -> GeneratedMarketAnalysis:
        if not self.answer.strip():
            raise ValueError("generated market answer must not be empty")
        if self.status is MarketAnswerStatus.ANSWERED and not (
            self.evidence_ids or self.calculation_ids
        ):
            raise ValueError("answered market analysis must cite evidence")
        return self


class MarketAnalysisAnswer(FrozenModel):
    symbol: str
    status: MarketAnswerStatus
    answer: str
    provider: str | None = None
    as_of: datetime | None = None
    periods: tuple[MarketPeriod, ...] = ()
    evidence: tuple[MarketEvidence, ...] = ()
    calculations: tuple[VerifiedCalculation, ...] = ()

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def period_start(self) -> date | None:
        if self.periods:
            return min(period.start for period in self.periods)
        if self.evidence:
            return min(item.observation_date for item in self.evidence)
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def period_end(self) -> date | None:
        if self.periods:
            return max(period.end for period in self.periods)
        if self.evidence:
            return max(item.observation_date for item in self.evidence)
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def disclaimer(self) -> str:
        return "AI-assisted analysis is informational only, not investment advice."
