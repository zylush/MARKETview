from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    EmbeddingDescriptor,
    EmbeddingVector,
    EvidenceChunk,
    EvidenceQuote,
    FilingDocument,
    GeneratedAnswer,
    GeneratedAnswerStatus,
    GeneratedClaim,
    GenerationManifest,
    ResearchAnswer,
    ResearchOutcome,
    SearchHit,
)
from app.research.ports import AnswerGenerator, Embedder, ResearchControlPlane, VectorStore
from app.research.service import ResearchCoreService, ResearchPolicy

_CATEGORIES = frozenset(
    {
        "answerable",
        "citation",
        "conflicting_evidence",
        "injection",
        "isolation",
        "malformed_id",
        "refusal",
        "unanswerable",
        "unknown_id",
        "weak_score",
    }
)
_CASE_KEYS = frozenset(
    {
        "id",
        "category",
        "symbol",
        "question",
        "evidence",
        "retrieved",
        "generated_claims",
        "expected",
    }
)
_EVIDENCE_KEYS = frozenset({"id", "symbol", "text", "score"})
_CLAIM_KEYS = frozenset({"text", "supporting_evidence_ids", "quotes"})
_QUOTE_KEYS = frozenset({"evidence_id", "text"})
_EXPECTED_KEYS = frozenset({"outcome", "citation_evidence_ids", "claim_count"})
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,31}$")

_EMBEDDING = EmbeddingDescriptor(
    provider="local-evaluation",
    model="deterministic",
    version="v1",
    dimensions=2,
)
_CORPUS = CorpusDescriptor(
    corpus_version="sec-evaluation-v1",
    chunker_version="fixed-fixture-v1",
    embedding=_EMBEDDING,
)


def _checked_text(value: object, name: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"evaluation fixture {name} must be text")
    checked = value.strip()
    if not checked or len(checked) > maximum or any(ord(character) < 32 for character in checked):
        raise ValueError(f"evaluation fixture {name} is invalid")
    return checked


def _checked_identifier(value: object, name: str) -> str:
    checked = _checked_text(value, name, maximum=80)
    if not _IDENTIFIER.fullmatch(checked):
        raise ValueError(f"evaluation fixture {name} is malformed")
    return checked


def _checked_object(value: object, keys: frozenset[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise ValueError(f"evaluation fixture {name} has an invalid schema")
    return cast(dict[str, object], value)


def _checked_list(value: object, name: str, *, maximum: int = 100) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"evaluation fixture {name} must be a bounded array")
    return cast(list[object], value)


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    evidence_id: str
    symbol: str
    text: str
    score: float

    def __post_init__(self) -> None:
        evidence_id = _checked_identifier(self.evidence_id, "evidence ID")
        symbol = _checked_text(self.symbol, "evidence symbol", maximum=32).upper()
        text = _checked_text(self.text, "evidence text", maximum=20_000)
        if not _SYMBOL.fullmatch(symbol):
            raise ValueError("evaluation fixture evidence symbol is malformed")
        if (
            type(self.score) is not float
            or not math.isfinite(self.score)
            or not 0 <= self.score <= 1
        ):
            raise ValueError("evaluation fixture evidence score must be between zero and one")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "text", text)


@dataclass(frozen=True, slots=True)
class EvaluationQuote:
    evidence_id: str
    text: str

    def __post_init__(self) -> None:
        evidence_id = _checked_text(self.evidence_id, "quote evidence ID", maximum=80)
        text = _checked_text(self.text, "quote text", maximum=500)
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "text", text)


@dataclass(frozen=True, slots=True)
class EvaluationGeneratedClaim:
    text: str
    supporting_evidence_ids: tuple[str, ...]
    quotes: tuple[EvaluationQuote, ...]

    def __post_init__(self) -> None:
        text = _checked_text(self.text, "generated claim", maximum=1000)
        supporting = tuple(
            _checked_text(item, "supporting evidence ID", maximum=80)
            for item in self.supporting_evidence_ids
        )
        quotes = tuple(self.quotes)
        if not supporting or len(set(supporting)) != len(supporting):
            raise ValueError("evaluation fixture supporting evidence IDs are invalid")
        if any(not isinstance(item, EvaluationQuote) for item in quotes):
            raise ValueError("evaluation fixture generated claim quotes are invalid")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "supporting_evidence_ids", supporting)
        object.__setattr__(self, "quotes", quotes)


@dataclass(frozen=True, slots=True)
class ResearchEvaluationCase:
    case_id: str
    category: str
    symbol: str
    question: str
    evidence: tuple[EvaluationEvidence, ...]
    retrieved_evidence_ids: tuple[str, ...]
    generated_claims: tuple[EvaluationGeneratedClaim, ...]
    expected_outcome: ResearchOutcome
    expected_citation_evidence_ids: tuple[str, ...]
    expected_claim_count: int

    def __post_init__(self) -> None:
        case_id = _checked_identifier(self.case_id, "case ID")
        category = _checked_identifier(self.category, "category")
        symbol = _checked_text(self.symbol, "case symbol", maximum=32).upper()
        question = _checked_text(self.question, "question", maximum=500)
        evidence = tuple(self.evidence)
        retrieved = tuple(self.retrieved_evidence_ids)
        claims = tuple(self.generated_claims)
        citations = tuple(self.expected_citation_evidence_ids)
        if category not in _CATEGORIES:
            raise ValueError("evaluation fixture category is unsupported")
        if not _SYMBOL.fullmatch(symbol):
            raise ValueError("evaluation fixture case symbol is malformed")
        if any(not isinstance(item, EvaluationEvidence) for item in evidence):
            raise ValueError("evaluation fixture evidence is invalid")
        evidence_ids = tuple(item.evidence_id for item in evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evaluation fixture evidence IDs must be unique")
        if any(item not in evidence_ids for item in retrieved):
            raise ValueError("evaluation fixture retrieved IDs must reference evidence")
        if any(not isinstance(item, EvaluationGeneratedClaim) for item in claims):
            raise ValueError("evaluation fixture generated claims are invalid")
        if not isinstance(self.expected_outcome, ResearchOutcome):
            raise ValueError("evaluation fixture expected outcome is invalid")
        if any(item not in evidence_ids for item in citations):
            raise ValueError("evaluation fixture expected citations must reference evidence")
        if type(self.expected_claim_count) is not int or self.expected_claim_count < 0:
            raise ValueError("evaluation fixture expected claim count is invalid")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "retrieved_evidence_ids", retrieved)
        object.__setattr__(self, "generated_claims", claims)
        object.__setattr__(self, "expected_citation_evidence_ids", citations)


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    case_id: str
    category: str
    passed: bool
    expected_outcome: ResearchOutcome
    actual_outcome: ResearchOutcome | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCategoryMetrics:
    category: str
    total_count: int
    passed_count: int
    failed_count: int
    pass_rate: float


@dataclass(frozen=True, slots=True)
class ResearchEvaluationReport:
    results: tuple[EvaluationCaseResult, ...]
    categories: tuple[EvaluationCategoryMetrics, ...]
    total_count: int
    passed_count: int
    failed_count: int
    pass_rate: float


def load_evaluation_cases(path: Path) -> tuple[ResearchEvaluationCase, ...]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise ValueError("evaluation fixture could not be loaded") from None
    rows = _checked_list(decoded, "root", maximum=1000)
    if not rows:
        raise ValueError("evaluation fixture must contain at least one case")
    cases = tuple(_parse_case(row) for row in rows)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("evaluation fixture case IDs must be unique")
    return cases


def _parse_case(raw: object) -> ResearchEvaluationCase:
    value = _checked_object(raw, _CASE_KEYS, "case")
    evidence = tuple(_parse_evidence(item) for item in _checked_list(value["evidence"], "evidence"))
    retrieved = tuple(
        _checked_text(item, "retrieved evidence ID", maximum=80)
        for item in _checked_list(value["retrieved"], "retrieved IDs")
    )
    claims = tuple(
        _parse_claim(item) for item in _checked_list(value["generated_claims"], "claims")
    )
    expected = _checked_object(value["expected"], _EXPECTED_KEYS, "expectation")
    expected_citations = tuple(
        _checked_text(item, "expected citation evidence ID", maximum=80)
        for item in _checked_list(expected["citation_evidence_ids"], "expected citations")
    )
    claim_count = expected["claim_count"]
    if type(claim_count) is not int:
        raise ValueError("evaluation fixture expected claim count must be an integer")
    raw_outcome = expected["outcome"]
    if not isinstance(raw_outcome, str):
        raise ValueError("evaluation fixture expected outcome is invalid")
    try:
        outcome = ResearchOutcome(raw_outcome)
    except (TypeError, ValueError):
        raise ValueError("evaluation fixture expected outcome is invalid") from None
    return ResearchEvaluationCase(
        case_id=_checked_identifier(value["id"], "case ID"),
        category=_checked_identifier(value["category"], "category"),
        symbol=_checked_text(value["symbol"], "case symbol", maximum=32),
        question=_checked_text(value["question"], "question", maximum=500),
        evidence=evidence,
        retrieved_evidence_ids=retrieved,
        generated_claims=claims,
        expected_outcome=outcome,
        expected_citation_evidence_ids=expected_citations,
        expected_claim_count=claim_count,
    )


def _parse_evidence(raw: object) -> EvaluationEvidence:
    value = _checked_object(raw, _EVIDENCE_KEYS, "evidence item")
    score = value["score"]
    if type(score) not in {int, float} or isinstance(score, bool):
        raise ValueError("evaluation fixture evidence score must be numeric")
    numeric_score = cast(int | float, score)
    return EvaluationEvidence(
        evidence_id=_checked_identifier(value["id"], "evidence ID"),
        symbol=_checked_text(value["symbol"], "evidence symbol", maximum=32),
        text=_checked_text(value["text"], "evidence text", maximum=20_000),
        score=float(numeric_score),
    )


def _parse_claim(raw: object) -> EvaluationGeneratedClaim:
    value = _checked_object(raw, _CLAIM_KEYS, "generated claim")
    supporting = tuple(
        _checked_text(item, "supporting evidence ID", maximum=80)
        for item in _checked_list(value["supporting_evidence_ids"], "supporting IDs")
    )
    quotes = tuple(_parse_quote(item) for item in _checked_list(value["quotes"], "quotes"))
    return EvaluationGeneratedClaim(
        text=_checked_text(value["text"], "generated claim text", maximum=1000),
        supporting_evidence_ids=supporting,
        quotes=quotes,
    )


def _parse_quote(raw: object) -> EvaluationQuote:
    value = _checked_object(raw, _QUOTE_KEYS, "quote")
    return EvaluationQuote(
        evidence_id=_checked_text(value["evidence_id"], "quote evidence ID", maximum=80),
        text=_checked_text(value["text"], "quote text", maximum=500),
    )


class LocalResearchEvaluationHarness:
    """Run fixed RAG cases through the real core using deterministic local fakes only."""

    def __init__(self, *, minimum_score: float = 0.70) -> None:
        self._policy = ResearchPolicy(minimum_score=minimum_score, max_results=5)

    async def evaluate(
        self,
        cases: tuple[ResearchEvaluationCase, ...],
    ) -> ResearchEvaluationReport:
        checked_cases = tuple(cases)
        if not checked_cases or any(
            not isinstance(case, ResearchEvaluationCase) for case in checked_cases
        ):
            raise ValueError("evaluation cases must be a nonempty immutable case tuple")
        if len({case.case_id for case in checked_cases}) != len(checked_cases):
            raise ValueError("evaluation case IDs must be unique")
        results = tuple([await self._evaluate_case(case) for case in checked_cases])
        passed_count = sum(result.passed for result in results)
        categories = tuple(
            self._category_metrics(category, results)
            for category in sorted({result.category for result in results})
        )
        return ResearchEvaluationReport(
            results=results,
            categories=categories,
            total_count=len(results),
            passed_count=passed_count,
            failed_count=len(results) - passed_count,
            pass_rate=passed_count / len(results),
        )

    async def _evaluate_case(self, case: ResearchEvaluationCase) -> EvaluationCaseResult:
        evidence_by_id = {item.evidence_id: self._build_evidence(item) for item in case.evidence}
        manifests = tuple(
            self._manifest(item) for item in evidence_by_id.values() if item.symbol == case.symbol
        )
        hits = tuple(
            SearchHit(
                evidence=evidence_by_id[evidence_id],
                score=self._fixture_evidence(case, evidence_id).score,
                embedding_descriptor=_EMBEDDING,
                active_generation_id=evidence_by_id[evidence_id].generation_id,
            )
            for evidence_id in case.retrieved_evidence_ids
        )
        generator = _LocalGenerator(case.generated_claims, evidence_by_id)
        core = ResearchCoreService(
            corpus=_CORPUS,
            chunker=None,
            embedder=cast(Embedder, _LocalEmbedder()),
            store=cast(VectorStore, _LocalStore(hits)),
            control_plane=cast(ResearchControlPlane, _LocalControlPlane(manifests)),
            generator=cast(AnswerGenerator, generator),
            policy=self._policy,
        )
        try:
            answer = await core.query_research(case.symbol, case.question)
        except Exception as error:
            return EvaluationCaseResult(
                case_id=case.case_id,
                category=case.category,
                passed=False,
                expected_outcome=case.expected_outcome,
                actual_outcome=None,
                failures=(f"evaluation case raised {type(error).__name__}",),
            )
        return self._result(case, answer, evidence_by_id)

    @staticmethod
    def _result(
        case: ResearchEvaluationCase,
        answer: ResearchAnswer,
        evidence_by_id: dict[str, EvidenceChunk],
    ) -> EvaluationCaseResult:
        failures: list[str] = []
        if answer.outcome is not case.expected_outcome:
            expected = case.expected_outcome.value
            actual = answer.outcome.value
            failures.append(f"expected outcome {expected} but received {actual}")
        if len(answer.claims) != case.expected_claim_count:
            failures.append(
                f"expected {case.expected_claim_count} claims but received {len(answer.claims)}"
            )
        reverse_ids = {item.chunk_id: evidence_id for evidence_id, item in evidence_by_id.items()}
        actual_citations = tuple(
            reverse_ids.get(citation.chunk_id, "unknown") for citation in answer.citations
        )
        if actual_citations != case.expected_citation_evidence_ids:
            failures.append(
                "expected citations "
                f"{case.expected_citation_evidence_ids!r} but received {actual_citations!r}"
            )
        if answer.outcome is ResearchOutcome.ANSWERED and any(
            not claim.evidence_quotes for claim in answer.claims
        ):
            failures.append("answered claim was missing a verified evidence quote")
        return EvaluationCaseResult(
            case_id=case.case_id,
            category=case.category,
            passed=not failures,
            expected_outcome=case.expected_outcome,
            actual_outcome=answer.outcome,
            failures=tuple(failures),
        )

    @staticmethod
    def _category_metrics(
        category: str,
        results: tuple[EvaluationCaseResult, ...],
    ) -> EvaluationCategoryMetrics:
        selected = tuple(result for result in results if result.category == category)
        passed = sum(result.passed for result in selected)
        return EvaluationCategoryMetrics(
            category=category,
            total_count=len(selected),
            passed_count=passed,
            failed_count=len(selected) - passed,
            pass_rate=passed / len(selected),
        )

    @staticmethod
    def _build_evidence(item: EvaluationEvidence) -> EvidenceChunk:
        cik = "0000320193" if item.symbol == "AAPL" else "0000789019"
        suffix = int(hashlib.sha256(item.evidence_id.encode()).hexdigest()[:8], 16) % 999_999 + 1
        accession = f"{cik}-25-{suffix:06d}"
        accession_path = accession.replace("-", "")
        document = FilingDocument(
            symbol=item.symbol,
            cik=cik,
            accession_number=accession,
            filing_type="10-K",
            title=f"{item.symbol} evaluation filing",
            filed_date=date(2025, 10, 31),
            source_url=(
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession_path}/{item.symbol.lower()}-evaluation.htm"
            ),
            text=item.text,
        )
        return EvidenceChunk.from_document(document, corpus=_CORPUS, ordinal=0, text=item.text)

    @staticmethod
    def _manifest(item: EvidenceChunk) -> GenerationManifest:
        return GenerationManifest(
            corpus=item.corpus,
            symbol=item.symbol,
            accession_number=item.accession_number,
            generation_id=item.generation_id,
            content_hash=item.content_hash,
            chunk_ids=(item.chunk_id,),
        )

    @staticmethod
    def _fixture_evidence(
        case: ResearchEvaluationCase,
        evidence_id: str,
    ) -> EvaluationEvidence:
        return next(item for item in case.evidence if item.evidence_id == evidence_id)


class _LocalEmbedder:
    descriptor = _EMBEDDING

    async def embed_query(
        self,
        text: str,
        *,
        deadline: RequestDeadline,
    ) -> EmbeddingVector:
        del text
        deadline.raise_if_expired()
        return EmbeddingVector(descriptor=_EMBEDDING, values=(1.0, 0.0))


class _LocalStore:
    def __init__(self, hits: tuple[SearchHit, ...]) -> None:
        self._hits = hits

    async def search(self, **kwargs: object) -> tuple[SearchHit, ...]:
        deadline = cast(RequestDeadline, kwargs["deadline"])
        limit = cast(int, kwargs["limit"])
        deadline.raise_if_expired()
        return self._hits[:limit]


class _LocalControlPlane:
    def __init__(self, manifests: tuple[GenerationManifest, ...]) -> None:
        self._manifests = manifests

    async def list_active_generations(self, **kwargs: object) -> tuple[GenerationManifest, ...]:
        deadline = cast(RequestDeadline, kwargs["deadline"])
        symbol = cast(str, kwargs["symbol"])
        corpus = cast(CorpusDescriptor, kwargs["corpus"])
        deadline.raise_if_expired()
        return tuple(
            item for item in self._manifests if item.symbol == symbol and item.corpus == corpus
        )


class _LocalGenerator:
    def __init__(
        self,
        claims: tuple[EvaluationGeneratedClaim, ...],
        evidence_by_id: dict[str, EvidenceChunk],
    ) -> None:
        self._claims = claims
        self._evidence_by_id = evidence_by_id

    async def generate(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        *,
        deadline: RequestDeadline,
    ) -> GeneratedAnswer:
        del question, evidence
        deadline.raise_if_expired()
        status = (
            GeneratedAnswerStatus.ANSWERED if self._claims else GeneratedAnswerStatus.INSUFFICIENT
        )
        return GeneratedAnswer(
            status=status,
            claims=tuple(
                GeneratedClaim(
                    text=claim.text,
                    supporting_chunk_ids=tuple(
                        self._resolve_id(item) for item in claim.supporting_evidence_ids
                    ),
                    evidence_quotes=tuple(
                        EvidenceQuote(chunk_id=self._resolve_id(item.evidence_id), quote=item.text)
                        for item in claim.quotes
                    ),
                )
                for claim in self._claims
            ),
        )

    def _resolve_id(self, evidence_id: str) -> str:
        if evidence_id.startswith("malformed:"):
            return evidence_id.removeprefix("malformed:")
        evidence = self._evidence_by_id.get(evidence_id)
        if evidence is not None:
            return evidence.chunk_id
        return "chunk-" + hashlib.sha256(f"unknown:{evidence_id}".encode()).hexdigest()
