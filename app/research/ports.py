from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.research.control import (
    AccessionLease,
    GenerationStageRecord,
    SupersededCleanupRecord,
)
from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    EmbeddedChunk,
    EmbeddingDescriptor,
    EmbeddingVector,
    EvidenceChunk,
    FilingDiscoveryPage,
    FilingDiscoveryRequest,
    FilingDocument,
    FilingReference,
    GeneratedAnswer,
    GenerationManifest,
    GenerationVerification,
    GenerationVerificationOutcome,
    RawFiling,
    SearchHit,
)


class GenerationInspectionState(StrEnum):
    """Safe aggregate state observed for one immutable generation manifest."""

    ABSENT = "absent"
    EXACT = "exact"
    PARTIAL = "partial"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True, slots=True)
class GenerationInspection:
    """Vector preflight result that deliberately excludes IDs, text, and vectors."""

    state: GenerationInspectionState
    expected_point_count: int
    observed_point_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, GenerationInspectionState):
            raise ValueError("generation inspection state is invalid")
        if type(self.expected_point_count) is not int or self.expected_point_count < 1:
            raise ValueError("generation inspection expected count is invalid")
        if type(self.observed_point_count) is not int or self.observed_point_count < 0:
            raise ValueError("generation inspection observed count is invalid")
        if self.state is GenerationInspectionState.ABSENT and self.observed_point_count != 0:
            raise ValueError("absent generation inspection must observe no points")
        if (
            self.state is GenerationInspectionState.EXACT
            and self.observed_point_count != self.expected_point_count
        ):
            raise ValueError("exact generation inspection must observe every point")
        if self.state is GenerationInspectionState.PARTIAL and not (
            0 < self.observed_point_count < self.expected_point_count
        ):
            raise ValueError("partial generation inspection count is invalid")


class FilingParser(Protocol):
    def parse(self, filing: RawFiling) -> FilingDocument: ...


class FilingSource(Protocol):
    async def discover(
        self,
        request: FilingDiscoveryRequest,
        *,
        deadline: RequestDeadline,
    ) -> FilingDiscoveryPage: ...

    async def fetch(
        self,
        reference: FilingReference,
        *,
        deadline: RequestDeadline,
    ) -> RawFiling: ...


class DocumentChunker(Protocol):
    def chunk(self, document: FilingDocument) -> tuple[str, ...]: ...


class Embedder(Protocol):
    @property
    def descriptor(self) -> EmbeddingDescriptor: ...

    async def embed_documents(
        self,
        texts: tuple[str, ...],
        *,
        deadline: RequestDeadline,
    ) -> tuple[EmbeddingVector, ...]: ...

    async def embed_query(
        self,
        text: str,
        *,
        deadline: RequestDeadline,
    ) -> EmbeddingVector: ...


class VectorStore(Protocol):
    async def inspect_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> GenerationInspection: ...

    async def stage_generation(
        self,
        manifest: GenerationManifest,
        chunks: tuple[EmbeddedChunk, ...],
        *,
        deadline: RequestDeadline,
    ) -> None: ...

    async def verify_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> GenerationVerificationOutcome: ...

    async def abort_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> None: ...

    async def delete_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> None: ...

    async def search(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        vector: EmbeddingVector,
        active_generations: tuple[GenerationManifest, ...],
        limit: int,
        deadline: RequestDeadline,
    ) -> tuple[SearchHit, ...]: ...


class ResearchControlPlane(Protocol):
    async def acquire_generation_lease(
        self,
        *,
        manifest: GenerationManifest,
        owner_digest: str,
        ttl_seconds: int,
        deadline: RequestDeadline,
    ) -> AccessionLease | None: ...

    async def stage_generation(
        self,
        *,
        lease: AccessionLease,
        manifest: GenerationManifest,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None: ...

    async def mark_generation_verified(
        self,
        *,
        lease: AccessionLease,
        staged: GenerationStageRecord,
        verification: GenerationVerification,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None: ...

    async def get_generation_stage(
        self,
        *,
        manifest: GenerationManifest,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None: ...

    async def get_active_generation(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        accession_number: str,
        deadline: RequestDeadline,
    ) -> GenerationManifest | None: ...

    async def list_active_generations(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        deadline: RequestDeadline,
    ) -> tuple[GenerationManifest, ...]: ...

    async def publish_generation(
        self,
        *,
        lease: AccessionLease,
        verified_stage: GenerationStageRecord,
        expected_previous_generation_id: str | None,
        manifest: GenerationManifest,
        superseded_manifest: GenerationManifest | None,
        deadline: RequestDeadline,
    ) -> bool: ...

    async def list_pending_cleanups(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        deadline: RequestDeadline,
    ) -> tuple[SupersededCleanupRecord, ...]: ...

    async def mark_superseded_cleaned(
        self,
        *,
        lease: AccessionLease,
        pending: SupersededCleanupRecord,
        marker_ttl_seconds: int,
        deadline: RequestDeadline,
    ) -> SupersededCleanupRecord | None: ...

    async def abort_generation(
        self,
        *,
        lease: AccessionLease,
        stage: GenerationStageRecord,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None: ...

    async def clean_generation(
        self,
        *,
        lease: AccessionLease,
        aborted_stage: GenerationStageRecord,
        marker_ttl_seconds: int,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None: ...

    async def release_generation_lease(
        self,
        *,
        lease: AccessionLease,
        deadline: RequestDeadline,
    ) -> bool: ...


class AnswerGenerator(Protocol):
    async def generate(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        *,
        deadline: RequestDeadline,
    ) -> GeneratedAnswer: ...
