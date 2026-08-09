from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Self

from app.research.domain import (
    CorpusDescriptor,
    EmbeddingDescriptor,
    GenerationManifest,
    GenerationVerification,
    GenerationVerificationOutcome,
    GenerationVerificationReason,
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ResearchControlUnavailableError(RuntimeError):
    """A sanitized, fail-closed control-plane failure."""


class IngestionFailureStage(StrEnum):
    """Fixed, non-sensitive stages allowed in ingestion diagnostics."""

    SEC_FETCH = "sec_fetch"
    PARSING_CHUNKING = "parsing_chunking"
    EMBEDDING = "embedding"
    VECTOR_STAGING = "vector_staging"
    VECTOR_VERIFICATION = "vector_verification"
    REDIS_PUBLICATION = "redis_publication"
    CLEANUP = "cleanup"
    CHECKPOINTING = "checkpointing"


class IngestionStageError(RuntimeError):
    """Sanitized ingestion failure detached from provider exception graphs."""

    def __init__(
        self,
        stage: IngestionFailureStage,
        *,
        verification: GenerationVerificationOutcome | None = None,
    ) -> None:
        if not isinstance(stage, IngestionFailureStage):
            raise ValueError("ingestion failure stage is invalid")
        if verification is not None and (
            stage is not IngestionFailureStage.VECTOR_VERIFICATION
            or not isinstance(verification, GenerationVerificationOutcome)
            or verification.reason is GenerationVerificationReason.VERIFIED
            or verification.verification is not None
        ):
            raise ValueError("ingestion verification diagnostic is invalid")
        self.stage = stage
        self.verification = verification
        super().__init__(f"research ingestion failed during {stage.value}")


def require_public_digest(value: object, name: str = "digest") -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_count(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "a positive" if positive else "an"
        raise ValueError(f"{name} must be {qualifier} integer count")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _load_record(raw: object, expected_keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("control record must be serialized text")
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("control record is malformed") from None
    if not isinstance(decoded, dict) or frozenset(decoded) != expected_keys:
        raise ValueError("control record has an invalid schema")
    return decoded


def _dump_record(value: dict[str, object]) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


_MANIFEST_KEYS = frozenset(
    {
        "corpus_version",
        "chunker_version",
        "embedding_provider",
        "embedding_model",
        "embedding_version",
        "embedding_dimensions",
        "symbol",
        "accession_number",
        "generation_id",
        "content_hash",
        "chunk_ids",
    }
)


def _manifest_value(manifest: GenerationManifest) -> dict[str, object]:
    embedding = manifest.corpus.embedding
    return {
        "accession_number": manifest.accession_number,
        "chunk_ids": list(manifest.chunk_ids),
        "chunker_version": manifest.corpus.chunker_version,
        "content_hash": manifest.content_hash,
        "corpus_version": manifest.corpus.corpus_version,
        "embedding_dimensions": embedding.dimensions,
        "embedding_model": embedding.model,
        "embedding_provider": embedding.provider,
        "embedding_version": embedding.version,
        "generation_id": manifest.generation_id,
        "symbol": manifest.symbol,
    }


def _manifest_from_value(value: object) -> GenerationManifest:
    if not isinstance(value, dict) or frozenset(value) != _MANIFEST_KEYS:
        raise ValueError("control record has an invalid manifest schema")
    embedding = EmbeddingDescriptor(
        provider=value["embedding_provider"],
        model=value["embedding_model"],
        version=value["embedding_version"],
        dimensions=value["embedding_dimensions"],
    )
    corpus = CorpusDescriptor(
        corpus_version=value["corpus_version"],
        chunker_version=value["chunker_version"],
        embedding=embedding,
    )
    chunk_ids = value["chunk_ids"]
    if not isinstance(chunk_ids, list):
        raise ValueError("control record chunk IDs are invalid")
    return GenerationManifest(
        corpus=corpus,
        symbol=value["symbol"],
        accession_number=value["accession_number"],
        generation_id=value["generation_id"],
        content_hash=value["content_hash"],
        chunk_ids=tuple(chunk_ids),
    )


class GenerationState(StrEnum):
    STAGED = "STAGED"
    VERIFIED = "VERIFIED"
    PUBLISHED = "PUBLISHED"
    ABORTED = "ABORTED"
    CLEANED = "CLEANED"


@dataclass(frozen=True, slots=True)
class AccessionLease:
    accession_digest: str
    owner_digest: str

    def __post_init__(self) -> None:
        require_public_digest(self.accession_digest, "accession digest")
        require_public_digest(self.owner_digest, "owner digest")


@dataclass(frozen=True, slots=True)
class GenerationStageRecord:
    """Immutable serialized lifecycle record around the canonical core manifest."""

    manifest: GenerationManifest
    state: GenerationState
    verification: GenerationVerification | None = None

    _KEYS = frozenset({"manifest", "state", "verification"})

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, GenerationManifest):
            raise ValueError("generation record manifest is invalid")
        if not isinstance(self.state, GenerationState):
            raise ValueError("generation record state is invalid")
        if self.verification is not None and not isinstance(
            self.verification, GenerationVerification
        ):
            raise ValueError("generation verification is invalid")
        if self.state in {GenerationState.VERIFIED, GenerationState.PUBLISHED} and (
            self.verification is None or not self.verification.proves(self.manifest)
        ):
            raise ValueError("verified generation state requires an exact point-set proof")
        if self.state is GenerationState.STAGED and self.verification is not None:
            raise ValueError("staged generation must not contain a verification proof")

    def with_state(self, state: GenerationState) -> GenerationStageRecord:
        return replace(self, state=state)

    def to_json(self) -> str:
        return _dump_record(
            {
                "manifest": _manifest_value(self.manifest),
                "state": self.state.value,
                "verification": (
                    None
                    if self.verification is None
                    else {
                        "generation_id": self.verification.generation_id,
                        "point_count": self.verification.point_count,
                        "point_ids_hash": self.verification.point_ids_hash,
                    }
                ),
            }
        )

    @classmethod
    def from_json(cls, raw: object) -> Self:
        value = _load_record(raw, cls._KEYS)
        try:
            manifest = _manifest_from_value(value["manifest"])
            verification_value = value["verification"]
            if verification_value is None:
                verification = None
            elif isinstance(verification_value, dict) and frozenset(verification_value) == {
                "generation_id",
                "point_count",
                "point_ids_hash",
            }:
                verification = GenerationVerification(
                    generation_id=verification_value["generation_id"],
                    point_count=verification_value["point_count"],
                    point_ids_hash=verification_value["point_ids_hash"],
                )
            else:
                raise ValueError
            state = GenerationState(value["state"])
            return cls(manifest=manifest, state=state, verification=verification)
        except (TypeError, ValueError):
            raise ValueError("control record has invalid generation data") from None


def recoverable_generation_records(
    manifest: GenerationManifest,
) -> tuple[GenerationStageRecord, ...]:
    staged = GenerationStageRecord(manifest, GenerationState.STAGED)
    verified = replace(
        staged,
        state=GenerationState.VERIFIED,
        verification=GenerationVerification.from_point_ids(
            manifest.generation_id,
            manifest.chunk_ids,
        ),
    )
    aborted_staged = staged.with_state(GenerationState.ABORTED)
    aborted_verified = verified.with_state(GenerationState.ABORTED)
    return (
        staged,
        verified,
        aborted_staged,
        aborted_verified,
        aborted_staged.with_state(GenerationState.CLEANED),
        aborted_verified.with_state(GenerationState.CLEANED),
    )


class SupersededCleanupState(StrEnum):
    PENDING = "PENDING"
    CLEANED = "CLEANED"


@dataclass(frozen=True, slots=True)
class SupersededCleanupRecord:
    """Public metadata required to replay deletion of superseded vectors."""

    superseded_manifest: GenerationManifest
    active_generation_id: str
    state: SupersededCleanupState

    _KEYS = frozenset({"superseded_manifest", "active_generation_id", "state"})

    def __post_init__(self) -> None:
        if not isinstance(self.superseded_manifest, GenerationManifest):
            raise ValueError("superseded cleanup manifest is invalid")
        if (
            not isinstance(self.active_generation_id, str)
            or not self.active_generation_id.startswith("gen-")
            or not _DIGEST.fullmatch(self.active_generation_id.removeprefix("gen-"))
            or self.active_generation_id == self.superseded_manifest.generation_id
        ):
            raise ValueError("superseded cleanup active generation is invalid")
        if not isinstance(self.state, SupersededCleanupState):
            raise ValueError("superseded cleanup state is invalid")

    def to_json(self) -> str:
        return _dump_record(
            {
                "active_generation_id": self.active_generation_id,
                "state": self.state.value,
                "superseded_manifest": _manifest_value(self.superseded_manifest),
            }
        )

    @classmethod
    def from_json(cls, raw: object) -> Self:
        value = _load_record(raw, cls._KEYS)
        try:
            return cls(
                superseded_manifest=_manifest_from_value(value["superseded_manifest"]),
                active_generation_id=value["active_generation_id"],
                state=SupersededCleanupState(value["state"]),
            )
        except (TypeError, ValueError):
            raise ValueError("control record has invalid superseded cleanup data") from None


@dataclass(frozen=True, slots=True)
class IngestionCheckpoint:
    job_digest: str
    cursor_digest: str | None
    processed_count: int
    failed_count: int
    complete: bool

    _KEYS = frozenset(
        {"job_digest", "cursor_digest", "processed_count", "failed_count", "complete"}
    )

    def __post_init__(self) -> None:
        require_public_digest(self.job_digest, "job digest")
        if self.cursor_digest is not None:
            require_public_digest(self.cursor_digest, "cursor digest")
        _require_count(self.processed_count, "processed count")
        _require_count(self.failed_count, "failed count")
        if type(self.complete) is not bool:
            raise ValueError("checkpoint completion flag must be boolean")

    def to_json(self) -> str:
        return _dump_record(
            {
                "complete": self.complete,
                "cursor_digest": self.cursor_digest,
                "failed_count": self.failed_count,
                "job_digest": self.job_digest,
                "processed_count": self.processed_count,
            }
        )

    @classmethod
    def from_json(cls, raw: object) -> Self:
        value = _load_record(raw, cls._KEYS)
        try:
            return cls(
                job_digest=value["job_digest"],
                cursor_digest=value["cursor_digest"],
                processed_count=value["processed_count"],
                failed_count=value["failed_count"],
                complete=value["complete"],
            )
        except (TypeError, ValueError):
            raise ValueError("control record has invalid checkpoint data") from None


@dataclass(frozen=True, slots=True)
class IngestionRetryClaim:
    job_digest: str
    attempt_digest: str
    checkpoint_digest: str
    cursor_digest: str

    _KEYS = frozenset({"attempt_digest", "checkpoint_digest", "cursor_digest", "job_digest"})

    def __post_init__(self) -> None:
        require_public_digest(self.job_digest, "job digest")
        require_public_digest(self.attempt_digest, "attempt digest")
        require_public_digest(self.checkpoint_digest, "checkpoint digest")
        require_public_digest(self.cursor_digest, "cursor digest")

    def to_json(self) -> str:
        return _dump_record(
            {
                "attempt_digest": self.attempt_digest,
                "checkpoint_digest": self.checkpoint_digest,
                "cursor_digest": self.cursor_digest,
                "job_digest": self.job_digest,
            }
        )

    @classmethod
    def from_json(cls, raw: object) -> Self:
        value = _load_record(raw, cls._KEYS)
        try:
            return cls(
                job_digest=value["job_digest"],
                attempt_digest=value["attempt_digest"],
                checkpoint_digest=value["checkpoint_digest"],
                cursor_digest=value["cursor_digest"],
            )
        except (TypeError, ValueError):
            raise ValueError("control record has invalid retry claim data") from None


class IngestionRetryState(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class IngestionRetryResult:
    job_digest: str
    attempt_digest: str
    state: IngestionRetryState
    failure_stage: IngestionFailureStage | None
    inserted_count: int
    removed_count: int

    _KEYS = frozenset(
        {
            "attempt_digest",
            "failure_stage",
            "inserted_count",
            "job_digest",
            "removed_count",
            "state",
        }
    )

    def __post_init__(self) -> None:
        require_public_digest(self.job_digest, "job digest")
        require_public_digest(self.attempt_digest, "attempt digest")
        if not isinstance(self.state, IngestionRetryState):
            raise ValueError("retry result state is invalid")
        if self.failure_stage is not None and not isinstance(
            self.failure_stage, IngestionFailureStage
        ):
            raise ValueError("retry failure stage is invalid")
        _require_count(self.inserted_count, "inserted count")
        _require_count(self.removed_count, "removed count")
        if (self.state is IngestionRetryState.FAILED) != (self.failure_stage is not None):
            raise ValueError("retry result failure stage is inconsistent")
        if self.state is IngestionRetryState.FAILED and (self.inserted_count or self.removed_count):
            raise ValueError("failed retry result must not report mutations")

    def to_json(self) -> str:
        return _dump_record(
            {
                "attempt_digest": self.attempt_digest,
                "failure_stage": (None if self.failure_stage is None else self.failure_stage.value),
                "inserted_count": self.inserted_count,
                "job_digest": self.job_digest,
                "removed_count": self.removed_count,
                "state": self.state.value,
            }
        )

    @classmethod
    def from_json(cls, raw: object) -> Self:
        value = _load_record(raw, cls._KEYS)
        try:
            return cls(
                job_digest=value["job_digest"],
                attempt_digest=value["attempt_digest"],
                state=IngestionRetryState(value["state"]),
                failure_stage=(
                    None
                    if value["failure_stage"] is None
                    else IngestionFailureStage(value["failure_stage"])
                ),
                inserted_count=value["inserted_count"],
                removed_count=value["removed_count"],
            )
        except (TypeError, ValueError):
            raise ValueError("control record has invalid retry result data") from None


@dataclass(frozen=True, slots=True)
class IngestionRetrySnapshot:
    checkpoint: IngestionCheckpoint | None
    claim: IngestionRetryClaim | None
    result: IngestionRetryResult | None

    def __post_init__(self) -> None:
        if self.result is not None and self.claim is None:
            raise ValueError("retry result requires its immutable claim")
        if self.claim is not None and self.checkpoint is None:
            raise ValueError("retry claim requires its immutable checkpoint")
        records = tuple(
            item for item in (self.checkpoint, self.claim, self.result) if item is not None
        )
        if records and len({item.job_digest for item in records}) != 1:
            raise ValueError("retry snapshot job digests do not match")
        if (
            self.claim is not None
            and self.result is not None
            and self.claim.attempt_digest != self.result.attempt_digest
        ):
            raise ValueError("retry snapshot attempt digests do not match")
        if self.claim is not None and self.checkpoint is not None:
            checkpoint_digest = hashlib.sha256(
                self.checkpoint.to_json().encode("utf-8")
            ).hexdigest()
            if (
                self.claim.checkpoint_digest != checkpoint_digest
                or self.claim.cursor_digest != self.checkpoint.cursor_digest
            ):
                raise ValueError("retry claim does not match its immutable checkpoint")


class ReservationState(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    COMMITTED = "COMMITTED"
    RELEASED = "RELEASED"


@dataclass(frozen=True, slots=True)
class Reservation:
    reservation_digest: str
    budget_digest: str
    principal_digest: str
    units: int
    state: ReservationState

    _KEYS = frozenset({"reservation_digest", "budget_digest", "principal_digest", "units", "state"})

    def __post_init__(self) -> None:
        require_public_digest(self.reservation_digest, "reservation digest")
        require_public_digest(self.budget_digest, "budget digest")
        require_public_digest(self.principal_digest, "principal digest")
        _require_count(self.units, "reservation units", positive=True)
        if not isinstance(self.state, ReservationState):
            raise ValueError("reservation state is invalid")

    def to_json(self) -> str:
        return _dump_record(
            {
                "budget_digest": self.budget_digest,
                "principal_digest": self.principal_digest,
                "reservation_digest": self.reservation_digest,
                "state": self.state.value,
                "units": self.units,
            }
        )

    @classmethod
    def from_json(cls, raw: object) -> Self:
        value = _load_record(raw, cls._KEYS)
        try:
            return cls(
                reservation_digest=value["reservation_digest"],
                budget_digest=value["budget_digest"],
                principal_digest=value["principal_digest"],
                units=value["units"],
                state=ReservationState(value["state"]),
            )
        except (TypeError, ValueError):
            raise ValueError("control record has invalid reservation data") from None
