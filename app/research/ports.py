from __future__ import annotations

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
    RawFiling,
    SearchHit,
)


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
    ) -> GenerationVerification | None: ...

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
