from __future__ import annotations

import hashlib
import math
from dataclasses import replace

from app.research.control import (
    AccessionLease,
    GenerationStageRecord,
    GenerationState,
    SupersededCleanupRecord,
    SupersededCleanupState,
    recoverable_generation_records,
)
from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    EmbeddedChunk,
    EmbeddingVector,
    EvidenceChunk,
    GenerationManifest,
    GenerationVerification,
    SearchHit,
)


class InMemoryVectorStore:
    """Immutable-state deterministic store for tests; never a production adapter."""

    def __init__(self) -> None:
        self._generations: tuple[tuple[GenerationManifest, tuple[EmbeddedChunk, ...]], ...] = ()

    @property
    def chunks(self) -> tuple[EvidenceChunk, ...]:
        return tuple(chunk.evidence for _, chunks in self._generations for chunk in chunks)

    @property
    def embedded_chunks(self) -> tuple[EmbeddedChunk, ...]:
        return tuple(chunk for _, chunks in self._generations for chunk in chunks)

    async def stage_generation(
        self,
        manifest: GenerationManifest,
        chunks: tuple[EmbeddedChunk, ...],
        *,
        deadline: RequestDeadline,
    ) -> None:
        deadline.raise_if_expired()
        checked_chunks = tuple(chunks)
        self._validate_generation(manifest, checked_chunks)
        existing = self._find_generation(manifest.generation_id)
        if existing is not None:
            if existing == (manifest, checked_chunks):
                return
            raise ValueError("research generation is immutable once staged")
        retained = tuple(
            entry for entry in self._generations if entry[0].generation_id != manifest.generation_id
        )
        self._generations = (*retained, (manifest, checked_chunks))

    async def verify_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> GenerationVerification | None:
        deadline.raise_if_expired()
        entry = self._find_generation(manifest.generation_id)
        if entry is None or entry[0] != manifest:
            return None
        try:
            self._validate_generation(entry[0], entry[1])
        except ValueError:
            return None
        return GenerationVerification.from_point_ids(
            manifest.generation_id,
            tuple(chunk.evidence.chunk_id for chunk in entry[1]),
        )

    async def abort_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> None:
        await self.delete_generation(manifest, deadline=deadline)

    async def delete_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> None:
        deadline.raise_if_expired()
        self._generations = tuple(
            entry for entry in self._generations if entry[0].generation_id != manifest.generation_id
        )

    async def search(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        vector: EmbeddingVector,
        active_generations: tuple[GenerationManifest, ...],
        limit: int,
        deadline: RequestDeadline,
    ) -> tuple[SearchHit, ...]:
        deadline.raise_if_expired()
        if vector.descriptor != corpus.embedding:
            raise ValueError("query embedding descriptor does not match the corpus")
        active = {
            manifest.generation_id: manifest
            for manifest in active_generations
            if manifest.corpus == corpus and manifest.symbol == symbol
        }
        hits = tuple(
            SearchHit(
                evidence=chunk.evidence,
                score=self._cosine(vector.values, chunk.embedding.values),
                embedding_descriptor=chunk.embedding.descriptor,
                active_generation_id=manifest.generation_id,
            )
            for manifest, chunks in self._generations
            if manifest.generation_id in active and active[manifest.generation_id] == manifest
            for chunk in chunks
        )
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.evidence.chunk_id))[:limit])

    def _find_generation(
        self, generation_id: str
    ) -> tuple[GenerationManifest, tuple[EmbeddedChunk, ...]] | None:
        return next(
            (entry for entry in self._generations if entry[0].generation_id == generation_id),
            None,
        )

    @staticmethod
    def _validate_generation(
        manifest: GenerationManifest,
        chunks: tuple[EmbeddedChunk, ...],
    ) -> None:
        if not chunks or tuple(chunk.evidence.chunk_id for chunk in chunks) != manifest.chunk_ids:
            raise ValueError("staged generation chunks do not match the manifest")
        if any(
            chunk.evidence.corpus != manifest.corpus
            or chunk.evidence.symbol != manifest.symbol
            or chunk.evidence.accession_number != manifest.accession_number
            or chunk.evidence.generation_id != manifest.generation_id
            or chunk.evidence.content_hash != manifest.content_hash
            for chunk in chunks
        ):
            raise ValueError("staged generation metadata does not match the manifest")

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        score = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
        return max(0.0, min(1.0, score))


class InMemoryResearchControlPlane:
    """Deterministic CAS manifest store standing in for Redis in unit tests."""

    def __init__(self) -> None:
        self._active: tuple[GenerationManifest, ...] = ()
        self._leases: tuple[AccessionLease, ...] = ()
        self._stages: tuple[GenerationStageRecord, ...] = ()
        self._pending_cleanups: tuple[SupersededCleanupRecord, ...] = ()

    @property
    def manifests(self) -> tuple[GenerationManifest, ...]:
        return self._active

    async def acquire_generation_lease(
        self,
        *,
        manifest: GenerationManifest,
        owner_digest: str,
        ttl_seconds: int,
        deadline: RequestDeadline,
    ) -> AccessionLease | None:
        deadline.raise_if_expired()
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 3600:
            raise ValueError("generation lease TTL must be between 1 and 3600 seconds")
        accession_digest = hashlib.sha256(
            (
                f"{manifest.corpus.canonical_key}\x1f{manifest.symbol}\x1f"
                f"{manifest.accession_number}"
            ).encode()
        ).hexdigest()
        if any(item.accession_digest == accession_digest for item in self._leases):
            return None
        lease = AccessionLease(accession_digest=accession_digest, owner_digest=owner_digest)
        self._leases = (*self._leases, lease)
        return lease

    async def stage_generation(
        self,
        *,
        lease: AccessionLease,
        manifest: GenerationManifest,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None:
        deadline.raise_if_expired()
        if not self._owns(lease, manifest):
            return None
        recoverable = recoverable_generation_records(manifest)
        record = recoverable[0]
        existing = next(
            (
                item
                for item in self._stages
                if item.manifest.generation_id == manifest.generation_id
            ),
            None,
        )
        if existing is not None and existing not in recoverable:
            return None
        retained = tuple(
            item for item in self._stages if item.manifest.generation_id != manifest.generation_id
        )
        self._stages = (*retained, record)
        return record

    async def mark_generation_verified(
        self,
        *,
        lease: AccessionLease,
        staged: GenerationStageRecord,
        verification: GenerationVerification,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None:
        deadline.raise_if_expired()
        if (
            not self._owns(lease, staged.manifest)
            or staged.state is not GenerationState.STAGED
            or not verification.proves(staged.manifest)
            or staged not in self._stages
        ):
            return None
        verified = replace(
            staged,
            state=GenerationState.VERIFIED,
            verification=verification,
        )
        self._replace_stage(verified)
        return verified

    async def get_active_generation(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        accession_number: str,
        deadline: RequestDeadline,
    ) -> GenerationManifest | None:
        deadline.raise_if_expired()
        return next(
            (
                manifest
                for manifest in self._active
                if manifest.corpus == corpus
                and manifest.symbol == symbol
                and manifest.accession_number == accession_number
            ),
            None,
        )

    async def list_active_generations(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        deadline: RequestDeadline,
    ) -> tuple[GenerationManifest, ...]:
        deadline.raise_if_expired()
        return tuple(
            manifest
            for manifest in self._active
            if manifest.corpus == corpus and manifest.symbol == symbol
        )

    async def publish_generation(
        self,
        *,
        lease: AccessionLease,
        verified_stage: GenerationStageRecord,
        expected_previous_generation_id: str | None,
        manifest: GenerationManifest,
        superseded_manifest: GenerationManifest | None,
        deadline: RequestDeadline,
    ) -> bool:
        deadline.raise_if_expired()
        if (
            not self._owns(lease, manifest)
            or verified_stage.manifest != manifest
            or verified_stage.state is not GenerationState.VERIFIED
            or verified_stage not in self._stages
        ):
            return False
        current = await self.get_active_generation(
            corpus=manifest.corpus,
            symbol=manifest.symbol,
            accession_number=manifest.accession_number,
            deadline=deadline,
        )
        current_id = current.generation_id if current is not None else None
        if current_id != expected_previous_generation_id:
            return False
        if superseded_manifest is not None and superseded_manifest != current:
            return False
        if current is not None and current != manifest and superseded_manifest is None:
            return False
        retained = tuple(
            item
            for item in self._active
            if not (
                item.corpus == manifest.corpus
                and item.symbol == manifest.symbol
                and item.accession_number == manifest.accession_number
            )
        )
        self._active = (*retained, manifest)
        self._replace_stage(verified_stage.with_state(GenerationState.PUBLISHED))
        if superseded_manifest is not None:
            pending = SupersededCleanupRecord(
                superseded_manifest=superseded_manifest,
                active_generation_id=manifest.generation_id,
                state=SupersededCleanupState.PENDING,
            )
            retained_pending = tuple(
                item
                for item in self._pending_cleanups
                if item.superseded_manifest.generation_id != superseded_manifest.generation_id
            )
            self._pending_cleanups = (*retained_pending, pending)
        return True

    async def list_pending_cleanups(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        deadline: RequestDeadline,
    ) -> tuple[SupersededCleanupRecord, ...]:
        deadline.raise_if_expired()
        return tuple(
            item
            for item in self._pending_cleanups
            if item.state is SupersededCleanupState.PENDING
            and item.superseded_manifest.corpus == corpus
            and item.superseded_manifest.symbol == symbol
        )

    async def mark_superseded_cleaned(
        self,
        *,
        lease: AccessionLease,
        pending: SupersededCleanupRecord,
        marker_ttl_seconds: int,
        deadline: RequestDeadline,
    ) -> SupersededCleanupRecord | None:
        deadline.raise_if_expired()
        if type(marker_ttl_seconds) is not int or marker_ttl_seconds < 1:
            raise ValueError("superseded cleanup marker TTL must be positive")
        active = await self.get_active_generation(
            corpus=pending.superseded_manifest.corpus,
            symbol=pending.superseded_manifest.symbol,
            accession_number=pending.superseded_manifest.accession_number,
            deadline=deadline,
        )
        if (
            not self._owns(lease, pending.superseded_manifest)
            or pending not in self._pending_cleanups
            or pending.state is not SupersededCleanupState.PENDING
            or active is None
            or active.generation_id == pending.superseded_manifest.generation_id
        ):
            return None
        cleaned = replace(pending, state=SupersededCleanupState.CLEANED)
        self._pending_cleanups = tuple(item for item in self._pending_cleanups if item != pending)
        return cleaned

    async def abort_generation(
        self,
        *,
        lease: AccessionLease,
        stage: GenerationStageRecord,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None:
        deadline.raise_if_expired()
        if not self._owns(lease, stage.manifest) or stage not in self._stages:
            return None
        aborted = stage.with_state(GenerationState.ABORTED)
        self._replace_stage(aborted)
        return aborted

    async def clean_generation(
        self,
        *,
        lease: AccessionLease,
        aborted_stage: GenerationStageRecord,
        marker_ttl_seconds: int,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None:
        deadline.raise_if_expired()
        if type(marker_ttl_seconds) is not int or marker_ttl_seconds < 1:
            raise ValueError("generation cleanup marker TTL must be positive")
        if (
            not self._owns(lease, aborted_stage.manifest)
            or aborted_stage.state is not GenerationState.ABORTED
            or aborted_stage not in self._stages
        ):
            return None
        cleaned = aborted_stage.with_state(GenerationState.CLEANED)
        self._replace_stage(cleaned)
        return cleaned

    async def release_generation_lease(
        self,
        *,
        lease: AccessionLease,
        deadline: RequestDeadline,
    ) -> bool:
        deadline.raise_if_expired()
        if lease not in self._leases:
            return False
        self._leases = tuple(item for item in self._leases if item != lease)
        return True

    def _owns(self, lease: AccessionLease, manifest: GenerationManifest) -> bool:
        expected = hashlib.sha256(
            (
                f"{manifest.corpus.canonical_key}\x1f{manifest.symbol}\x1f"
                f"{manifest.accession_number}"
            ).encode()
        ).hexdigest()
        return lease in self._leases and lease.accession_digest == expected

    def _replace_stage(self, record: GenerationStageRecord) -> None:
        retained = tuple(
            item
            for item in self._stages
            if item.manifest.generation_id != record.manifest.generation_id
        )
        self._stages = (*retained, record)
