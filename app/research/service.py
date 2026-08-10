from __future__ import annotations

import math
import re
import secrets
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.research.control import (
    AccessionLease,
    GenerationState,
    IngestionFailureStage,
    IngestionStageError,
)
from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    EmbeddedChunk,
    EvidenceChunk,
    FilingDocument,
    GeneratedAnswerStatus,
    GeneratedClaim,
    GenerationManifest,
    GenerationVerificationOutcome,
    GenerationVerificationReason,
    IngestionResult,
    ResearchAnswer,
    ResearchCitation,
    SearchHit,
)
from app.research.ports import (
    AnswerGenerator,
    DocumentChunker,
    Embedder,
    GenerationInspectionState,
    ResearchControlPlane,
    VectorStore,
)
from app.research.retrieval import classify_safe_hits, contains_prompt_injection
from app.validation import validate_research_question, validate_symbol

_PERSONAL_CONTEXT = re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b", re.IGNORECASE)
_INVESTMENT_ACTION = re.compile(
    r"\b(?:buy|buying|purchase|purchasing|acquire|acquiring|invest|investing|"
    r"sell|selling|unload|unloading|divest|divesting|dispose|disposing|hold|holding|own)\b",
    re.IGNORECASE,
)
_INVESTMENT_SUITABILITY = re.compile(
    r"\b(?:recommend|suit|suits|fit|fits|appropriate|right|wise|belong|belongs)\b",
    re.IGNORECASE,
)
_LEXICAL_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "did",
        "does",
        "during",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "may",
        "might",
        "of",
        "on",
        "period",
        "reported",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "would",
    }
)
_LEXICAL_EQUIVALENTS = {
    "constrain": "limit",
    "constrained": "limit",
    "constrains": "limit",
    "limited": "limit",
    "limits": "limit",
    "scarcity": "shortage",
    "scarce": "shortage",
    "shortages": "shortage",
    "decline": "decrease",
    "declined": "decrease",
    "declines": "decrease",
    "decreased": "decrease",
    "decreases": "decrease",
    "fall": "decrease",
    "fell": "decrease",
    "falls": "decrease",
    "grew": "increase",
    "grow": "increase",
    "growing": "increase",
    "increased": "increase",
    "increases": "increase",
    "not": "negation",
    "no": "negation",
    "never": "negation",
    "without": "negation",
    "rise": "increase",
    "rises": "increase",
    "rose": "increase",
}
_LEXICAL_TOKEN = re.compile(r"[a-z0-9]+")
_POLARITY_PAIRS = (
    frozenset({"increase", "decrease"}),
    frozenset({"gain", "loss"}),
    frozenset({"profit", "loss"}),
)
_POLARITY_TOKENS = frozenset(token for pair in _POLARITY_PAIRS for token in pair)
_NEGATION_MARKER = "negation"


class ResearchCorpusUnavailableError(RuntimeError):
    """The configured corpus has no valid active manifest for the requested symbol."""


@dataclass(frozen=True, slots=True)
class _IngestionOutcome:
    result: IngestionResult | None = None
    failure_stage: IngestionFailureStage | None = None
    verification: GenerationVerificationOutcome | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.failure_stage is None):
            raise ValueError("ingestion outcome must contain exactly one result")
        if self.verification is not None and (
            self.failure_stage is not IngestionFailureStage.VECTOR_VERIFICATION
            or self.result is not None
            or self.verification.reason is GenerationVerificationReason.VERIFIED
            or self.verification.verification is not None
        ):
            raise ValueError("ingestion outcome verification diagnostic is invalid")


@dataclass(frozen=True, slots=True)
class ResearchPolicy:
    minimum_score: float = 0.70
    max_results: int = 5
    overfetch_factor: int = 3
    request_timeout_seconds: float = 7.5
    ingestion_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if (
            type(self.minimum_score) is not float
            or not math.isfinite(self.minimum_score)
            or not 0.0 <= self.minimum_score <= 1.0
        ):
            raise ValueError("minimum research score must be between zero and one")
        if type(self.max_results) is not int or not 1 <= self.max_results <= 20:
            raise ValueError("maximum research results must be between 1 and 20")
        if type(self.overfetch_factor) is not int or not 1 <= self.overfetch_factor <= 10:
            raise ValueError("research overfetch factor must be between 1 and 10")
        if not 0 < self.request_timeout_seconds < 8:
            raise ValueError(
                "research request timeout must be greater than zero and below 8 seconds"
            )
        if not 0 < self.ingestion_timeout_seconds <= 3600:
            raise ValueError("research ingestion timeout must be between zero and 3600 seconds")


class ResearchCoreService:
    """Orchestrate provider-neutral ingestion and citation-grounded retrieval."""

    def __init__(
        self,
        *,
        corpus: CorpusDescriptor,
        chunker: DocumentChunker | None = None,
        embedder: Embedder,
        store: VectorStore,
        control_plane: ResearchControlPlane,
        generator: AnswerGenerator,
        policy: ResearchPolicy | None = None,
    ) -> None:
        if embedder.descriptor != corpus.embedding:
            raise ValueError("embedder descriptor does not match the configured corpus")
        self._corpus = corpus
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._control_plane = control_plane
        self._generator = generator
        self._policy = policy or ResearchPolicy()

    async def ingest(
        self,
        document: FilingDocument,
        *,
        deadline: RequestDeadline | None = None,
        retry_failed: bool = False,
    ) -> IngestionResult:
        outcome = await self._ingestion_outcome(
            document,
            deadline=deadline,
            retry_failed=retry_failed,
        )
        document = None  # type: ignore[assignment]
        if outcome.failure_stage is not None:
            raise IngestionStageError(
                outcome.failure_stage,
                verification=outcome.verification,
            ) from None
        if outcome.result is None:  # pragma: no cover - guarded by the outcome type
            raise IngestionStageError(IngestionFailureStage.PARSING_CHUNKING) from None
        return outcome.result

    async def _ingestion_outcome(
        self,
        document: FilingDocument,
        *,
        deadline: RequestDeadline | None,
        retry_failed: bool,
    ) -> _IngestionOutcome:
        try:
            result = await self._ingest_impl(
                document,
                deadline=deadline,
                retry_failed=retry_failed,
            )
            return _IngestionOutcome(result=result)
        except IngestionStageError as error:
            return _IngestionOutcome(
                failure_stage=error.stage,
                verification=error.verification,
            )
        except Exception:
            return _IngestionOutcome(failure_stage=IngestionFailureStage.CLEANUP)

    async def _ingest_impl(
        self,
        document: FilingDocument,
        *,
        deadline: RequestDeadline | None = None,
        retry_failed: bool = False,
    ) -> IngestionResult:
        operation_deadline = deadline or RequestDeadline.after(
            self._policy.ingestion_timeout_seconds
        )
        operation_deadline.raise_if_expired()
        try:
            if self._chunker is None:
                raise ValueError("research ingestion chunker is not configured")
            texts = tuple(text.strip() for text in self._chunker.chunk(document) if text.strip())
            if not texts:
                raise ValueError("filing chunker produced no usable content")
            evidence = tuple(
                EvidenceChunk.from_document(
                    document,
                    corpus=self._corpus,
                    ordinal=ordinal,
                    text=text,
                )
                for ordinal, text in enumerate(texts)
            )
            manifest = self._manifest(evidence)
        except Exception:
            raise IngestionStageError(IngestionFailureStage.PARSING_CHUNKING) from None
        try:
            lease = await self._control_plane.acquire_generation_lease(
                manifest=manifest,
                owner_digest=secrets.token_hex(32),
                ttl_seconds=max(1, min(3600, math.ceil(operation_deadline.remaining_seconds()))),
                deadline=operation_deadline,
            )
        except Exception:
            raise IngestionStageError(IngestionFailureStage.REDIS_PUBLICATION) from None
        if lease is None:
            raise IngestionStageError(IngestionFailureStage.REDIS_PUBLICATION) from None
        generation_stage = None
        published = False
        current: GenerationManifest | None = None
        failure_stage = IngestionFailureStage.REDIS_PUBLICATION
        try:
            current = await self._control_plane.get_active_generation(
                corpus=self._corpus,
                symbol=document.symbol,
                accession_number=document.accession_number,
                deadline=operation_deadline,
            )
            if retry_failed and current is not None:
                raise RuntimeError("failed-checkpoint recovery state is inconsistent")
            if retry_failed:
                pending = await self._control_plane.list_pending_cleanups(
                    corpus=self._corpus,
                    symbol=document.symbol,
                    deadline=operation_deadline,
                )
                if any(
                    item.superseded_manifest.accession_number == document.accession_number
                    for item in pending
                ):
                    raise RuntimeError("failed-checkpoint recovery state is inconsistent")
                recovered_stage = await self._control_plane.get_generation_stage(
                    manifest=manifest,
                    deadline=operation_deadline,
                )
                if recovered_stage is None or recovered_stage.state is not GenerationState.CLEANED:
                    raise RuntimeError("failed-checkpoint recovery state is inconsistent")
                failure_stage = IngestionFailureStage.VECTOR_VERIFICATION
                inspection = await self._store.inspect_generation(
                    manifest,
                    deadline=operation_deadline,
                )
                if inspection.state is not GenerationInspectionState.ABSENT:
                    raise RuntimeError("failed-checkpoint recovery state is inconsistent")
                failure_stage = IngestionFailureStage.REDIS_PUBLICATION
            if current == manifest:
                cleanup_pending_count = await self._cleanup_superseded_generations(
                    lease=lease,
                    active_manifest=manifest,
                    deadline=operation_deadline,
                )
                return IngestionResult(
                    "unchanged",
                    inserted_count=0,
                    removed_count=0,
                    cleanup_pending_count=cleanup_pending_count,
                )
            generation_stage = await self._control_plane.stage_generation(
                lease=lease,
                manifest=manifest,
                deadline=operation_deadline,
            )
            if generation_stage is None:
                raise RuntimeError("research generation could not be staged")
            failure_stage = IngestionFailureStage.EMBEDDING
            embeddings = await self._embedder.embed_documents(
                texts,
                deadline=operation_deadline,
            )
            if len(embeddings) != len(texts):
                raise ValueError("embedding result count did not match filing chunks")
            chunks = tuple(
                EmbeddedChunk(evidence=item, embedding=embedding)
                for item, embedding in zip(evidence, embeddings, strict=True)
            )
            failure_stage = IngestionFailureStage.VECTOR_STAGING
            await self._store.stage_generation(manifest, chunks, deadline=operation_deadline)
            failure_stage = IngestionFailureStage.VECTOR_VERIFICATION
            verification_outcome = await self._store.verify_generation(
                manifest,
                deadline=operation_deadline,
            )
            if not verification_outcome.proves(manifest):
                raise IngestionStageError(
                    IngestionFailureStage.VECTOR_VERIFICATION,
                    verification=verification_outcome,
                )
            verification = verification_outcome.verification
            if verification is None:  # pragma: no cover - enforced by outcome invariants
                raise IngestionStageError(IngestionFailureStage.VECTOR_VERIFICATION)
            failure_stage = IngestionFailureStage.REDIS_PUBLICATION
            verified_stage = await self._control_plane.mark_generation_verified(
                lease=lease,
                staged=generation_stage,
                verification=verification,
                deadline=operation_deadline,
            )
            if verified_stage is None:
                raise RuntimeError("research generation verification was not recorded")
            generation_stage = verified_stage
            published = await self._control_plane.publish_generation(
                lease=lease,
                verified_stage=verified_stage,
                expected_previous_generation_id=(
                    current.generation_id if current is not None else None
                ),
                manifest=manifest,
                superseded_manifest=(
                    current if current is not None and current != manifest else None
                ),
                deadline=operation_deadline,
            )
            if not published:
                latest = await self._control_plane.get_active_generation(
                    corpus=self._corpus,
                    symbol=document.symbol,
                    accession_number=document.accession_number,
                    deadline=operation_deadline,
                )
                if latest != manifest:
                    raise RuntimeError("research generation publication conflicted")
            cleanup_pending_count = await self._cleanup_superseded_generations(
                lease=lease,
                active_manifest=manifest,
                deadline=operation_deadline,
            )
            return IngestionResult(
                "unchanged"
                if current == manifest
                else "replaced"
                if current is not None
                else "created",
                inserted_count=0 if current == manifest else len(chunks),
                removed_count=(
                    0 if current is None or current == manifest else len(current.chunk_ids)
                ),
                cleanup_pending_count=cleanup_pending_count,
            )
        except Exception as error:
            original_stage = failure_stage
            verification_diagnostic = (
                error.verification if isinstance(error, IngestionStageError) else None
            )
            if (
                generation_stage is not None
                and not published
                and operation_deadline.remaining_seconds() > 0
            ):
                try:
                    aborted = await self._control_plane.abort_generation(
                        lease=lease,
                        stage=generation_stage,
                        deadline=operation_deadline,
                    )
                    if aborted is not None:
                        if current is None or current.generation_id != manifest.generation_id:
                            await self._store.abort_generation(
                                manifest,
                                deadline=operation_deadline,
                            )
                        await self._control_plane.clean_generation(
                            lease=lease,
                            aborted_stage=aborted,
                            marker_ttl_seconds=86_400,
                            deadline=operation_deadline,
                        )
                except Exception:
                    original_stage = IngestionFailureStage.CLEANUP
                    verification_diagnostic = None
            raise IngestionStageError(
                original_stage,
                verification=(
                    verification_diagnostic
                    if original_stage is IngestionFailureStage.VECTOR_VERIFICATION
                    else None
                ),
            ) from None
        finally:
            if operation_deadline.remaining_seconds() > 0:
                active_error = sys.exception()
                release_failed = False
                try:
                    await self._control_plane.release_generation_lease(
                        lease=lease,
                        deadline=operation_deadline,
                    )
                except Exception:
                    release_failed = True
                if release_failed and active_error is None:
                    raise IngestionStageError(IngestionFailureStage.CLEANUP) from None

    async def _cleanup_superseded_generations(
        self,
        *,
        lease: AccessionLease,
        active_manifest: GenerationManifest,
        deadline: RequestDeadline,
    ) -> int:
        try:
            pending_records = await self._control_plane.list_pending_cleanups(
                corpus=self._corpus,
                symbol=active_manifest.symbol,
                deadline=deadline,
            )
        except Exception:
            return 1
        selected = tuple(
            item
            for item in pending_records
            if item.superseded_manifest.accession_number == active_manifest.accession_number
        )
        pending_count = 0
        for pending in selected:
            try:
                await self._store.delete_generation(
                    pending.superseded_manifest,
                    deadline=deadline,
                )
                cleaned = await self._control_plane.mark_superseded_cleaned(
                    lease=lease,
                    pending=pending,
                    marker_ttl_seconds=86_400,
                    deadline=deadline,
                )
            except Exception:
                pending_count += 1
                continue
            if cleaned is None:
                pending_count += 1
        return pending_count

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        deadline: RequestDeadline | None = None,
        before_paid_call: Callable[[], Awaitable[None]] | None = None,
    ) -> ResearchAnswer:
        operation_deadline = deadline or RequestDeadline.after(self._policy.request_timeout_seconds)
        checked_symbol = validate_symbol(symbol)
        checked_question = validate_research_question(question)
        if self._is_personalized_trade_request(checked_question):
            return ResearchAnswer.refusal(checked_symbol)
        active = await self._control_plane.list_active_generations(
            corpus=self._corpus,
            symbol=checked_symbol,
            deadline=operation_deadline,
        )
        active = self._safe_manifests(checked_symbol, active)
        if not active:
            raise ResearchCorpusUnavailableError("research corpus is unavailable")
        if before_paid_call is not None:
            await before_paid_call()
        vector = await self._embedder.embed_query(
            checked_question,
            deadline=operation_deadline,
        )
        if vector.descriptor != self._corpus.embedding:
            return ResearchAnswer.insufficient(checked_symbol)
        raw_hits = await self._store.search(
            corpus=self._corpus,
            symbol=checked_symbol,
            vector=vector,
            active_generations=active,
            limit=self._policy.max_results * self._policy.overfetch_factor,
            deadline=operation_deadline,
        )
        safe_hits = self._safe_hits(checked_symbol, active, raw_hits)
        if not safe_hits:
            return ResearchAnswer.insufficient(checked_symbol)
        evidence = tuple(hit.evidence for hit in safe_hits)
        if self._evidence_has_direct_conflict(evidence):
            return ResearchAnswer.insufficient(checked_symbol)
        generated = await self._generator.generate(
            checked_question,
            evidence,
            deadline=operation_deadline,
        )
        if generated.status is GeneratedAnswerStatus.REFUSED:
            return ResearchAnswer.refusal(checked_symbol)
        if generated.status is GeneratedAnswerStatus.INSUFFICIENT:
            return ResearchAnswer.insufficient(checked_symbol)
        if not self._claims_are_grounded(generated.claims, evidence):
            return ResearchAnswer.insufficient(checked_symbol)
        selected = {item.chunk_id: item for item in evidence}
        referenced = tuple(
            chunk_id for claim in generated.claims for chunk_id in claim.supporting_chunk_ids
        )
        citations = self._citations(referenced, selected)
        return ResearchAnswer.answered(checked_symbol, generated.claims, citations)

    def _safe_manifests(
        self,
        symbol: str,
        manifests: tuple[GenerationManifest, ...],
    ) -> tuple[GenerationManifest, ...]:
        safe: list[GenerationManifest] = []
        seen_accessions: set[str] = set()
        seen_generations: set[str] = set()
        for manifest in manifests:
            if (
                manifest.corpus != self._corpus
                or manifest.symbol != symbol
                or manifest.accession_number in seen_accessions
                or manifest.generation_id in seen_generations
            ):
                continue
            safe.append(manifest)
            seen_accessions.add(manifest.accession_number)
            seen_generations.add(manifest.generation_id)
        return tuple(safe)

    def _safe_hits(
        self,
        symbol: str,
        active: tuple[GenerationManifest, ...],
        hits: tuple[SearchHit, ...],
    ) -> tuple[SearchHit, ...]:
        return classify_safe_hits(
            symbol,
            self._corpus,
            active,
            hits,
            minimum_score=self._policy.minimum_score,
            max_results=self._policy.max_results,
        ).accepted_hits

    def _claims_are_grounded(
        self,
        claims: tuple[GeneratedClaim, ...],
        evidence: tuple[EvidenceChunk, ...],
    ) -> bool:
        if not claims:
            return False
        selected = {item.chunk_id: item for item in evidence}
        for claim in claims:
            if not claim.evidence_quotes or self._is_personalized_trade_request(claim.text):
                return False
            quoted_text: list[str] = []
            for quote in claim.evidence_quotes:
                chunk = selected.get(quote.chunk_id)
                if chunk is None or quote.quote not in chunk.text:
                    return False
                quoted_text.append(quote.quote)
            if any(chunk_id not in selected for chunk_id in claim.supporting_chunk_ids):
                return False
            if not self._claim_has_lexical_support(claim.text, tuple(quoted_text)):
                return False
        return True

    def _manifest(self, evidence: tuple[EvidenceChunk, ...]) -> GenerationManifest:
        first = evidence[0]
        if any(
            item.generation_id != first.generation_id
            or item.accession_number != first.accession_number
            or item.content_hash != first.content_hash
            or item.corpus != self._corpus
            for item in evidence
        ):
            raise ValueError("research evidence does not form one immutable generation")
        return GenerationManifest(
            corpus=self._corpus,
            symbol=first.symbol,
            accession_number=first.accession_number,
            generation_id=first.generation_id,
            content_hash=first.content_hash,
            chunk_ids=tuple(item.chunk_id for item in evidence),
        )

    @staticmethod
    def _citations(
        referenced: tuple[str, ...],
        selected: dict[str, EvidenceChunk],
    ) -> tuple[ResearchCitation, ...]:
        ordered_unique = tuple(dict.fromkeys(referenced))
        return tuple(ResearchCitation.from_chunk(selected[chunk_id]) for chunk_id in ordered_unique)

    @staticmethod
    def _contains_prompt_injection(text: str) -> bool:
        return contains_prompt_injection(text)

    @staticmethod
    def _is_personalized_trade_request(text: str) -> bool:
        return bool(
            _PERSONAL_CONTEXT.search(text)
            and (_INVESTMENT_ACTION.search(text) or _INVESTMENT_SUITABILITY.search(text))
        )

    @staticmethod
    def _claim_has_lexical_support(claim: str, quotes: tuple[str, ...]) -> bool:
        claim_tokens = ResearchCoreService._support_tokens(claim)
        quote_tokens = frozenset(
            token for quote in quotes for token in ResearchCoreService._support_tokens(quote)
        )
        if len(claim_tokens) < 2 or not quote_tokens:
            return False
        numeric_tokens = frozenset(token for token in claim_tokens if token.isdecimal())
        if not numeric_tokens.issubset(quote_tokens):
            return False
        if (claim_tokens.count(_NEGATION_MARKER) > 0) != (_NEGATION_MARKER in quote_tokens):
            return False
        if any(
            frozenset(claim_tokens).intersection(pair) != quote_tokens.intersection(pair)
            for pair in _POLARITY_PAIRS
        ):
            return False
        supported_count = sum(token in quote_tokens for token in claim_tokens)
        return supported_count * 4 >= len(claim_tokens) * 3

    @staticmethod
    def _evidence_has_direct_conflict(evidence: tuple[EvidenceChunk, ...]) -> bool:
        token_sets = tuple(
            frozenset(ResearchCoreService._support_tokens(item.text)) for item in evidence
        )
        non_subject_tokens = _POLARITY_TOKENS | {_NEGATION_MARKER}
        for index, left in enumerate(token_sets):
            for right in token_sets[index + 1 :]:
                shared_subject = (left & right) - non_subject_tokens
                if len(shared_subject) < 2:
                    continue
                if (_NEGATION_MARKER in left) != (_NEGATION_MARKER in right):
                    return True
                if any(
                    left.intersection(pair)
                    and right.intersection(pair)
                    and left.intersection(pair) != right.intersection(pair)
                    for pair in _POLARITY_PAIRS
                ):
                    return True
        return False

    @staticmethod
    def _support_tokens(text: str) -> tuple[str, ...]:
        raw_tokens = _LEXICAL_TOKEN.findall(text.casefold())
        normalized = tuple(
            _LEXICAL_EQUIVALENTS.get(token, token[:-1] if token.endswith("s") else token)
            for token in raw_tokens
            if token not in _LEXICAL_STOP_WORDS
        )
        return tuple(dict.fromkeys(normalized))
