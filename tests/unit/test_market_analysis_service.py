from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from app.errors import MarketDataError
from app.market_analysis.domain import (
    GeneratedMarketAnalysis,
    MarketAnswerStatus,
)
from app.models import EODBar, Page
from app.providers.openai_market_analysis import OpenAIMarketAnalysisError
from app.services.market_analysis import MarketAnalysisService

NOW = datetime(2026, 8, 10, 12, tzinfo=UTC)


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


class MarketDataStub:
    provider_name = "Market Data"

    def __init__(
        self,
        *,
        latest: EODBar | Exception | None = None,
        history: Page[EODBar] | Exception | None = None,
    ) -> None:
        self.latest = latest
        self.history_result = history
        self.calls: list[tuple[object, ...]] = []
        self.last_metadata = {"as_of": NOW}

    async def latest_eod(self, symbol: str) -> EODBar:
        self.calls.append(("latest", symbol))
        if isinstance(self.latest, Exception):
            raise self.latest
        assert self.latest is not None
        return self.latest

    async def history(
        self, symbol: str, start_date: date, end_date: date, *, limit: int
    ) -> Page[EODBar]:
        self.calls.append(("history", symbol, start_date, end_date, limit))
        if isinstance(self.history_result, Exception):
            raise self.history_result
        assert self.history_result is not None
        return self.history_result


class GeneratorStub:
    def __init__(self, result: GeneratedMarketAnalysis | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, object, float]] = []

    async def generate(
        self, question: str, evidence: object, *, timeout_seconds: float
    ) -> GeneratedMarketAnalysis:
        self.calls.append((question, evidence, timeout_seconds))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_snapshot_uses_one_latest_call_and_one_grounded_generation() -> None:
    market = MarketDataStub(latest=bar(date(2026, 8, 7), "101.25"))
    generator = GeneratorStub(
        GeneratedMarketAnalysis(
            status=MarketAnswerStatus.ANSWERED,
            answer="AAPL's observed close was 101.25 USD per share.",
            evidence_ids=(),
            calculation_ids=("placeholder",),
        )
    )
    service = MarketAnalysisService(market, generator, now=lambda: NOW)

    # The generator stub needs to cite the immutable IDs it receives, so derive its
    # response at call time without weakening the production validation contract.
    async def generate(
        question: str, evidence: object, *, timeout_seconds: float
    ) -> GeneratedMarketAnalysis:
        generator.calls.append((question, evidence, timeout_seconds))
        fact_id = evidence.evidence[0].evidence_id  # type: ignore[attr-defined]
        return GeneratedMarketAnalysis(
            status=MarketAnswerStatus.ANSWERED,
            answer="AAPL's observed close was 101.25 USD per share.",
            evidence_ids=(fact_id,),
        )

    generator.generate = generate  # type: ignore[method-assign]
    authorizations = 0

    async def authorize() -> None:
        nonlocal authorizations
        authorizations += 1

    answer = await service.query_research(
        "aapl",
        "What is the latest close?",
        before_generation=authorize,
    )

    assert answer.status is MarketAnswerStatus.ANSWERED
    assert answer.provider == "Market Data"
    assert answer.as_of == NOW
    assert answer.evidence[0].value == Decimal("101.25")
    assert answer.answer != "AAPL's observed close was 101.25 USD per share."
    assert "latest EOD close" in answer.answer
    assert market.calls == [("latest", "AAPL")]
    assert len(generator.calls) == 1
    assert authorizations == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_default_trend_uses_one_366_bar_history_call() -> None:
    items = tuple(
        bar(date(2026, 7, 11) + timedelta(days=offset), str(100 + offset)) for offset in range(31)
    )
    market = MarketDataStub(history=Page(items=items))
    generator = GeneratorStub(
        GeneratedMarketAnalysis(
            status=MarketAnswerStatus.INSUFFICIENT_EVIDENCE,
            answer="Market evidence was insufficient for a grounded analysis.",
        )
    )
    service = MarketAnalysisService(market, generator, now=lambda: NOW)

    authorizations = 0

    async def authorize() -> None:
        nonlocal authorizations
        authorizations += 1

    answer = await service.query_research(
        "AAPL",
        "How has AAPL performed?",
        before_generation=authorize,
    )

    assert answer.status is MarketAnswerStatus.INSUFFICIENT_EVIDENCE
    assert market.calls == [("history", "AAPL", date(2026, 7, 11), date(2026, 8, 10), 366)]
    assert len(generator.calls) == 1
    assert authorizations == 1
    bundle = generator.calls[0][1]
    assert bundle.periods[0].start == date(2026, 7, 11)  # type: ignore[attr-defined]
    assert bundle.calculations  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_comparison_fetches_union_once_and_builds_both_periods() -> None:
    items = tuple(
        bar(date(2026, 1, 1) + timedelta(days=offset), str(100 + offset)) for offset in range(59)
    )
    market = MarketDataStub(history=Page(items=items))
    generator = GeneratorStub(
        GeneratedMarketAnalysis(
            status=MarketAnswerStatus.INSUFFICIENT_EVIDENCE,
            answer="Market evidence was insufficient for a grounded analysis.",
        )
    )
    service = MarketAnalysisService(market, generator, now=lambda: NOW)

    answer = await service.query_research(
        "AAPL",
        "Compare 2026-01-01 to 2026-01-31 vs 2026-02-01 to 2026-02-28",
    )

    assert answer.status is MarketAnswerStatus.INSUFFICIENT_EVIDENCE
    assert market.calls == [("history", "AAPL", date(2026, 1, 1), date(2026, 2, 28), 366)]
    assert len(generator.calls[0][1].periods) == 2  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sparse_or_malformed_evidence_is_insufficient_without_openai_call() -> None:
    market = MarketDataStub(history=Page(items=(bar(date(2026, 8, 9), "100"),)))
    generator = GeneratorStub(RuntimeError("must not run"))
    service = MarketAnalysisService(market, generator, now=lambda: NOW)

    authorizations = 0

    async def authorize() -> None:
        nonlocal authorizations
        authorizations += 1

    answer = await service.query_research(
        "AAPL",
        "How has AAPL performed?",
        before_generation=authorize,
    )

    assert answer.status is MarketAnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.provider == "Market Data"
    assert generator.calls == []
    assert authorizations == 0


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [MarketDataError("private"), TimeoutError("private")])
async def test_market_provider_failure_is_unavailable(failure: Exception) -> None:
    market = MarketDataStub(latest=failure)
    generator = GeneratorStub(RuntimeError("must not run"))
    answer = await MarketAnalysisService(market, generator, now=lambda: NOW).query_research(
        "AAPL", "latest price"
    )

    assert answer.status is MarketAnswerStatus.UNAVAILABLE
    assert "private" not in answer.answer
    assert generator.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_failure_is_unavailable_after_one_call() -> None:
    market = MarketDataStub(latest=bar(date(2026, 8, 7), "101"))
    generator = GeneratorStub(OpenAIMarketAnalysisError("private provider detail"))

    answer = await MarketAnalysisService(market, generator, now=lambda: NOW).query_research(
        "AAPL", "latest price"
    )

    assert answer.status is MarketAnswerStatus.UNAVAILABLE
    assert "private" not in answer.answer
    assert len(generator.calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unsupported_question_is_insufficient_without_external_calls() -> None:
    market = MarketDataStub()
    generator = GeneratorStub(RuntimeError("must not run"))

    answer = await MarketAnalysisService(market, generator, now=lambda: NOW).query_research(
        "AAPL", "Tell me about its corporate culture"
    )

    assert answer.status is MarketAnswerStatus.INSUFFICIENT_EVIDENCE
    assert market.calls == []
    assert generator.calls == []
