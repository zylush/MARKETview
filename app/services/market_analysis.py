from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from typing import Any, Protocol

from app.errors import MarketDataError
from app.market_analysis.domain import (
    GeneratedMarketAnalysis,
    MarketAnalysisAnswer,
    MarketAnswerStatus,
    MarketEvidenceBundle,
    MarketIntent,
)
from app.market_analysis.evidence import (
    EvidenceQualityError,
    build_period_bundle,
    build_snapshot_bundle,
    normalize_market_bars,
)
from app.market_analysis.planning import plan_market_research
from app.providers.openai_market_analysis import OpenAIMarketAnalysisError


class MarketAnalysisUnavailableError(RuntimeError):
    """Sanitized failure raised by a configured market-analysis runtime."""


class MarketDataProtocol(Protocol):
    provider_name: str

    @property
    def last_metadata(self) -> object | None: ...

    async def latest_eod(self, symbol: str) -> Any: ...

    async def history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        limit: int,
    ) -> Any: ...


class MarketAnalysisGeneratorProtocol(Protocol):
    async def generate(
        self,
        question: str,
        evidence: MarketEvidenceBundle,
        *,
        timeout_seconds: float,
    ) -> GeneratedMarketAnalysis: ...


class MarketAnalysisService:
    """Collect one bounded market dataset and synthesize one grounded LLM response."""

    def __init__(
        self,
        market_data: MarketDataProtocol,
        generator: MarketAnalysisGeneratorProtocol,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        timeout_seconds: float = 7.5,
    ) -> None:
        if not 0 < timeout_seconds <= 30:
            raise ValueError("market analysis timeout is invalid")
        self._market_data = market_data
        self._generator = generator
        self._now = now
        self._timeout_seconds = float(timeout_seconds)

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        before_generation: Callable[[], Awaitable[None]] | None = None,
    ) -> MarketAnalysisAnswer:
        current = self._utc_now()
        try:
            plan = plan_market_research(question, symbol, current_date=current.date())
        except ValueError:
            return _insufficient(symbol)
        if plan.market_intent is MarketIntent.NONE:
            return _insufficient(plan.symbol)

        try:
            if plan.market_intent is MarketIntent.SNAPSHOT:
                raw = await self._market_data.latest_eod(plan.symbol)
                normalized = normalize_market_bars(
                    (raw,), plan.symbol, current_date=plan.current_date
                )
                if len(normalized) != 1:
                    raise EvidenceQualityError("market snapshot is missing")
                bundle = build_snapshot_bundle(
                    normalized[0],
                    provider=self._provider_name(),
                    provider_timestamp=self._provider_timestamp(current),
                    current_date=plan.current_date,
                )
            else:
                start = min(period.start for period in plan.periods)
                end = max(period.end for period in plan.periods)
                page = await self._market_data.history(
                    plan.symbol,
                    start,
                    end,
                    limit=plan.history_limit or 366,
                )
                records = getattr(page, "items", None)
                if records is None:
                    raise EvidenceQualityError("historical market evidence is malformed")
                normalized = normalize_market_bars(
                    records, plan.symbol, current_date=plan.current_date
                )
                bundle = build_period_bundle(
                    normalized,
                    plan.periods,
                    symbol=plan.symbol,
                    provider=self._provider_name(),
                    provider_timestamp=self._provider_timestamp(current),
                )
        except EvidenceQualityError:
            return _insufficient(
                plan.symbol,
                provider=self._provider_name(),
                as_of=self._optional_provider_timestamp(),
                periods=plan.periods,
            )
        except (MarketDataError, TimeoutError):
            return _unavailable(plan.symbol, periods=plan.periods)

        try:
            if before_generation is not None:
                await before_generation()
            generated = await self._generator.generate(
                question,
                bundle,
                timeout_seconds=self._timeout_seconds,
            )
            return _answer_from_generation(bundle, generated)
        except (OpenAIMarketAnalysisError, TimeoutError):
            return _unavailable(
                plan.symbol,
                provider=bundle.provider,
                as_of=bundle.as_of,
                periods=bundle.periods,
                evidence=bundle.evidence,
                calculations=bundle.calculations,
            )

    async def aclose(self) -> None:
        close = getattr(self._generator, "aclose", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    def manages(self, resource: object) -> bool:
        return resource is self._generator

    def _utc_now(self) -> datetime:
        current = self._now()
        if not isinstance(current, datetime) or current.tzinfo is None:
            raise ValueError("market analysis clock must be timezone-aware")
        return current.astimezone(UTC)

    def _provider_name(self) -> str:
        value = getattr(self._market_data, "provider_name", "marketdata.app")
        return value.strip() if isinstance(value, str) and value.strip() else "marketdata.app"

    def _provider_timestamp(self, fallback: datetime) -> datetime:
        return self._optional_provider_timestamp() or fallback

    def _optional_provider_timestamp(self) -> datetime | None:
        metadata = getattr(self._market_data, "last_metadata", None)
        value = (
            metadata.get("as_of")
            if isinstance(metadata, dict)
            else getattr(metadata, "as_of", None)
        )
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return value.astimezone(UTC) if isinstance(value, datetime) and value.tzinfo else None


class DisabledMarketAnalysisService:
    """Fail-closed market-analysis implementation used without OpenAI configuration."""

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        before_generation: Callable[[], Awaitable[None]] | None = None,
    ) -> MarketAnalysisAnswer:
        del before_generation
        del question
        return _unavailable(symbol)

    async def aclose(self) -> None:
        return None

    def manages(self, resource: object) -> bool:
        del resource
        return False


def _answer_from_generation(
    bundle: MarketEvidenceBundle,
    generated: GeneratedMarketAnalysis,
) -> MarketAnalysisAnswer:
    if not isinstance(generated, GeneratedMarketAnalysis):
        raise OpenAIMarketAnalysisError("generated analysis was invalid")
    known_evidence = frozenset(item.evidence_id for item in bundle.evidence)
    known_calculations = frozenset(item.calculation_id for item in bundle.calculations)
    if (
        not set(generated.evidence_ids) <= known_evidence
        or not set(generated.calculation_ids) <= known_calculations
    ):
        raise OpenAIMarketAnalysisError("generated analysis was invalid")
    answer = generated.answer
    if generated.status is MarketAnswerStatus.ANSWERED:
        cited_evidence = frozenset(generated.evidence_ids)
        cited_calculations = frozenset(generated.calculation_ids)
        statements = tuple(
            statement.text
            for statement in bundle.locked_statements
            if (
                bool(statement.evidence_ids)
                and set(statement.evidence_ids).issubset(cited_evidence)
            )
            or (
                bool(statement.calculation_ids)
                and set(statement.calculation_ids).issubset(cited_calculations)
            )
        )
        if not statements:
            raise OpenAIMarketAnalysisError("generated analysis was invalid")
        answer = " ".join(statements)
    return MarketAnalysisAnswer(
        symbol=bundle.symbol,
        status=generated.status,
        answer=answer,
        provider=bundle.provider,
        as_of=bundle.as_of,
        periods=bundle.periods,
        evidence=bundle.evidence,
        calculations=bundle.calculations,
    )


def _insufficient(
    symbol: str,
    *,
    provider: str | None = None,
    as_of: datetime | None = None,
    periods: tuple[Any, ...] = (),
) -> MarketAnalysisAnswer:
    return MarketAnalysisAnswer(
        symbol=symbol,
        status=MarketAnswerStatus.INSUFFICIENT_EVIDENCE,
        answer="There is insufficient market evidence to answer this question.",
        provider=provider,
        as_of=as_of,
        periods=periods,
    )


def _unavailable(
    symbol: str,
    *,
    provider: str | None = None,
    as_of: datetime | None = None,
    periods: tuple[Any, ...] = (),
    evidence: tuple[Any, ...] = (),
    calculations: tuple[Any, ...] = (),
) -> MarketAnalysisAnswer:
    return MarketAnalysisAnswer(
        symbol=symbol,
        status=MarketAnswerStatus.UNAVAILABLE,
        answer="Market analysis is temporarily unavailable.",
        provider=provider,
        as_of=as_of,
        periods=periods,
        evidence=evidence,
        calculations=calculations,
    )


__all__ = [
    "DisabledMarketAnalysisService",
    "MarketAnalysisService",
    "MarketAnalysisUnavailableError",
]
