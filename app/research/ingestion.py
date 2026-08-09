from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from app.research.control import (
    IngestionFailureStage,
    IngestionRetryClaim,
    IngestionRetryResult,
    IngestionRetryState,
    IngestionStageError,
)
from app.research.deadline import RequestDeadline
from app.research.domain import (
    FilingDiscoveryRequest,
    FilingDocument,
    FilingReference,
    GenerationVerificationOutcome,
    GenerationVerificationReason,
)
from app.research.ports import FilingParser, FilingSource


class IngestOutcome(Protocol):
    @property
    def inserted_count(self) -> int: ...

    @property
    def removed_count(self) -> int: ...


class CoreIngestor(Protocol):
    async def ingest(
        self,
        document: FilingDocument,
        *,
        deadline: RequestDeadline | None = None,
        retry_failed: bool = False,
    ) -> IngestOutcome: ...


_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")
_CIK = re.compile(r"^\d{10}$")
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_ALLOWED_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A"})


class ResearchIngestionRetryError(RuntimeError):
    """Sanitized refusal for unavailable, stale, or conflicting retry state."""

    def __init__(self) -> None:
        super().__init__("research ingestion retry state is unavailable")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checkpoint_json(record: IngestionCheckpointStore.Record) -> str:
    return json.dumps(
        {
            "complete": record.complete,
            "cursor_digest": record.cursor_digest,
            "failed_count": record.failed_count,
            "job_digest": record.job_digest,
            "processed_count": record.processed_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_symbol(value: object) -> str:
    normalized = value.strip().upper() if isinstance(value, str) else ""
    if _SYMBOL.fullmatch(normalized) is None:
        raise ValueError("symbol is invalid")
    return normalized


def _validate_cik(value: object) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _CIK.fullmatch(normalized) is None:
        raise ValueError("CIK is invalid")
    return normalized


def _validate_forms(values: tuple[str, ...]) -> tuple[str, ...]:
    forms = tuple(item.strip().upper() for item in values if isinstance(item, str) and item.strip())
    if not forms or len(forms) > len(_ALLOWED_FORMS):
        raise ValueError("filing type allowlist is invalid")
    if len(set(forms)) != len(forms) or any(item not in _ALLOWED_FORMS for item in forms):
        raise ValueError("filing type allowlist is invalid")
    return forms


@dataclass(frozen=True, slots=True)
class ResearchIngestionRequest:
    symbol: str
    cik: str
    filing_types: tuple[str, ...]
    date_from: date
    date_to: date
    limit: int
    apply: bool = False
    retry_failed: bool = False

    def __post_init__(self) -> None:
        symbol = _validate_symbol(self.symbol)
        cik = _validate_cik(self.cik)
        forms = _validate_forms(tuple(self.filing_types))
        if type(self.date_from) is not date or type(self.date_to) is not date:
            raise ValueError("date allowlist is invalid")
        if self.date_from > self.date_to:
            raise ValueError("date allowlist is invalid")
        if type(self.limit) is not int or not 1 <= self.limit <= 100:
            raise ValueError("count allowlist is invalid")
        if type(self.apply) is not bool:
            raise ValueError("apply flag is invalid")
        if type(self.retry_failed) is not bool or (self.retry_failed and not self.apply):
            raise ValueError("failed retry requires apply")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "cik", cik)
        object.__setattr__(self, "filing_types", forms)

    def discovery_request(self) -> FilingDiscoveryRequest:
        return FilingDiscoveryRequest(
            symbol=self.symbol,
            cik=self.cik,
            filing_types=self.filing_types,
            date_from=self.date_from,
            date_to=self.date_to,
            limit=self.limit,
        )


@dataclass(frozen=True, slots=True)
class ResearchIngestionJob:
    request: ResearchIngestionRequest
    references: tuple[FilingReference, ...] = field(repr=False)
    job_digest: str

    @classmethod
    def from_plan(
        cls,
        request: ResearchIngestionRequest,
        references: tuple[FilingReference, ...],
    ) -> ResearchIngestionJob:
        if len(references) > request.limit or any(
            not isinstance(item, FilingReference) for item in references
        ):
            raise ValueError("ingestion plan is invalid")
        material = "|".join(
            (
                request.symbol,
                request.cik,
                ",".join(request.filing_types),
                request.date_from.isoformat(),
                request.date_to.isoformat(),
                str(request.limit),
                ",".join(item.accession_number for item in references),
            )
        )
        return cls(request=request, references=references, job_digest=_digest(material))

    def cursor_for(self, accession_number: str) -> str:
        if not isinstance(accession_number, str) or _ACCESSION.fullmatch(accession_number) is None:
            raise ValueError("checkpoint cursor accession is invalid")
        if all(item.accession_number != accession_number for item in self.references):
            raise ValueError("checkpoint cursor is outside the ingestion plan")
        return _digest(f"{self.job_digest}|{accession_number}")

    def cursor_index(self, cursor_digest: str | None) -> int:
        if cursor_digest is None:
            return 0
        if (
            not isinstance(cursor_digest, str)
            or re.fullmatch(r"^[0-9a-f]{64}$", cursor_digest) is None
        ):
            raise ValueError("checkpoint cursor is invalid")
        for index, reference in enumerate(self.references, start=1):
            if self.cursor_for(reference.accession_number) == cursor_digest:
                return index
        raise ValueError("checkpoint cursor is outside the ingestion plan")


class IngestionCheckpointStore(Protocol):
    @dataclass(frozen=True, slots=True)
    class Record:
        job_digest: str
        cursor_digest: str | None
        processed_count: int
        failed_count: int
        complete: bool

        def __post_init__(self) -> None:
            if not isinstance(self.job_digest, str) or not re.fullmatch(
                r"^[0-9a-f]{64}$", self.job_digest
            ):
                raise ValueError("checkpoint job ID is invalid")
            if self.cursor_digest is not None and (
                not isinstance(self.cursor_digest, str)
                or re.fullmatch(r"^[0-9a-f]{64}$", self.cursor_digest) is None
            ):
                raise ValueError("checkpoint cursor is invalid")
            if type(self.processed_count) is not int or self.processed_count < 0:
                raise ValueError("checkpoint processed count is invalid")
            if type(self.failed_count) is not int or self.failed_count < 0:
                raise ValueError("checkpoint failed count is invalid")
            if type(self.complete) is not bool:
                raise ValueError("checkpoint completion is invalid")

    async def load(self, job_digest: str, *, deadline: RequestDeadline) -> Record | None: ...
    async def save(self, record: Record, *, deadline: RequestDeadline) -> None: ...
    async def load_retry_snapshot(
        self, job_digest: str, *, deadline: RequestDeadline
    ) -> RetryCheckpointSnapshot: ...
    async def claim_failed_retry(
        self,
        checkpoint: Record,
        claim: IngestionRetryClaim,
        *,
        deadline: RequestDeadline,
    ) -> bool: ...
    async def finish_failed_retry(
        self,
        claim: IngestionRetryClaim,
        result: IngestionRetryResult,
        *,
        deadline: RequestDeadline,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class RetryCheckpointSnapshot:
    checkpoint: IngestionCheckpointStore.Record | None
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
        if (
            self.claim is not None
            and self.checkpoint is not None
            and (
                self.claim.checkpoint_digest != _digest(_checkpoint_json(self.checkpoint))
                or self.claim.cursor_digest != self.checkpoint.cursor_digest
            )
        ):
            raise ValueError("retry claim does not match its immutable checkpoint")


@dataclass(frozen=True, slots=True)
class ResearchIngestionResult:
    dry_run: bool
    job: ResearchIngestionJob = field(repr=False)
    planned_count: int
    processed_count: int
    skipped_count: int
    failed_count: int
    inserted_count: int
    removed_count: int
    errors: tuple[str, ...] = ()
    failure_stages: tuple[IngestionFailureStage, ...] = ()
    vector_verification_failures: tuple[GenerationVerificationOutcome, ...] = ()

    def __post_init__(self) -> None:
        diagnostics = tuple(self.vector_verification_failures)
        if any(
            not isinstance(item, GenerationVerificationOutcome)
            or item.reason is GenerationVerificationReason.VERIFIED
            or item.verification is not None
            for item in diagnostics
        ):
            raise ValueError("vector verification diagnostics are invalid")
        if diagnostics and IngestionFailureStage.VECTOR_VERIFICATION not in self.failure_stages:
            raise ValueError("vector verification diagnostics require a matching stage")
        object.__setattr__(self, "vector_verification_failures", diagnostics)

    @property
    def opaque_job_id(self) -> str:
        return self.job.job_digest

    @property
    def opaque_accession_ids(self) -> tuple[str, ...]:
        return tuple(_digest(item.accession_number) for item in self.job.references)


@dataclass(frozen=True, slots=True)
class _RunnerOutcome:
    result: ResearchIngestionResult | None = None
    failure_stage: IngestionFailureStage | None = None
    verification: GenerationVerificationOutcome | None = None
    timeout: bool = False
    retry_conflict: bool = False

    def __post_init__(self) -> None:
        present = sum(
            (
                self.result is not None,
                self.failure_stage is not None,
                self.timeout,
                self.retry_conflict,
            )
        )
        if present != 1:
            raise ValueError("runner outcome must contain exactly one result")
        if self.verification is not None and (
            self.failure_stage is not IngestionFailureStage.VECTOR_VERIFICATION
            or self.result is not None
            or self.verification.reason is GenerationVerificationReason.VERIFIED
            or self.verification.verification is not None
        ):
            raise ValueError("runner verification diagnostic is invalid")


def _resolve_runner_outcome(outcome: _RunnerOutcome) -> ResearchIngestionResult:
    if outcome.result is not None:
        return outcome.result
    if outcome.failure_stage is not None:
        raise IngestionStageError(
            outcome.failure_stage,
            verification=outcome.verification,
        ) from None
    if outcome.timeout:
        raise TimeoutError("research request deadline expired") from None
    raise ResearchIngestionRetryError() from None


class ResearchIngestionRunner:
    def __init__(
        self,
        *,
        source: FilingSource,
        parser: FilingParser,
        core: CoreIngestor,
        checkpoints: IngestionCheckpointStore | None = None,
    ) -> None:
        self._source = source
        self._parser = parser
        self._core = core
        self._checkpoints = checkpoints

    async def run(
        self,
        request: ResearchIngestionRequest,
        *,
        deadline: RequestDeadline,
    ) -> ResearchIngestionResult:
        outcome = await self._run_outcome(request, deadline=deadline)
        request = None  # type: ignore[assignment]
        return _resolve_runner_outcome(outcome)

    async def _run_outcome(
        self,
        request: ResearchIngestionRequest,
        *,
        deadline: RequestDeadline,
    ) -> _RunnerOutcome:
        try:
            result = await self._run_impl(request, deadline=deadline)
            return _RunnerOutcome(result=result)
        except IngestionStageError as error:
            return _RunnerOutcome(
                failure_stage=error.stage,
                verification=error.verification,
            )
        except TimeoutError:
            return _RunnerOutcome(timeout=True)
        except ResearchIngestionRetryError:
            return _RunnerOutcome(retry_conflict=True)
        except RuntimeError:
            return _RunnerOutcome(failure_stage=IngestionFailureStage.CHECKPOINTING)
        except Exception:
            return _RunnerOutcome(failure_stage=IngestionFailureStage.CHECKPOINTING)

    async def _run_impl(
        self,
        request: ResearchIngestionRequest,
        *,
        deadline: RequestDeadline,
    ) -> ResearchIngestionResult:
        if not isinstance(request, ResearchIngestionRequest):
            raise ValueError("ingestion request is invalid")
        deadline.raise_if_expired()
        try:
            page = await self._source.discover(request.discovery_request(), deadline=deadline)
        except TimeoutError:
            raise
        except Exception:
            raise IngestionStageError(IngestionFailureStage.SEC_FETCH) from None
        job = ResearchIngestionJob.from_plan(request, tuple(page.references))
        if not request.apply:
            return ResearchIngestionResult(
                dry_run=True,
                job=job,
                planned_count=len(job.references),
                processed_count=0,
                skipped_count=0,
                failed_count=0,
                inserted_count=0,
                removed_count=0,
            )
        checkpoint = await self._load_checkpoint(job.job_digest, deadline=deadline)
        retry_claim: IngestionRetryClaim | None = None
        if request.retry_failed:
            retry_claim, prior_result = await self._claim_retry(
                job,
                checkpoint,
                deadline=deadline,
            )
            if prior_result is not None:
                return ResearchIngestionResult(
                    dry_run=False,
                    job=job,
                    planned_count=len(job.references),
                    processed_count=0,
                    skipped_count=len(job.references),
                    failed_count=0,
                    inserted_count=0,
                    removed_count=0,
                )
        resume_index = job.cursor_index(checkpoint.cursor_digest) if checkpoint is not None else 0
        if retry_claim is not None:
            resume_index = 0
        failed_count = checkpoint.failed_count if checkpoint is not None else 0
        total_success_count = checkpoint.processed_count if checkpoint is not None else 0
        if retry_claim is not None:
            failed_count = 0
            total_success_count = 0
        processed_count = 0
        inserted_count = 0
        removed_count = 0
        errors: tuple[str, ...] = ()
        failure_stages: tuple[IngestionFailureStage, ...] = ()
        vector_verification_failures: tuple[GenerationVerificationOutcome, ...] = ()
        cursor_digest = checkpoint.cursor_digest if checkpoint is not None else None
        for index, reference in enumerate(job.references):
            if index < resume_index:
                continue
            failure_stage = IngestionFailureStage.SEC_FETCH
            try:
                raw = await self._source.fetch(reference, deadline=deadline)
                failure_stage = IngestionFailureStage.PARSING_CHUNKING
                document = self._parser.parse(raw)
                if retry_claim is None:
                    result = await self._core.ingest(document, deadline=deadline)
                else:
                    result = await self._core.ingest(
                        document,
                        deadline=deadline,
                        retry_failed=True,
                    )
                inserted_count += result.inserted_count
                removed_count += result.removed_count
                processed_count += 1
                total_success_count += 1
            except IngestionStageError as error:
                failure_stage = error.stage
                failed_count += 1
                failure_stages = (*failure_stages, failure_stage)
                if error.verification is not None:
                    vector_verification_failures = (
                        *vector_verification_failures,
                        error.verification,
                    )
                errors = (*errors, f"ingestion failed during {failure_stage.value}")
            except Exception:
                failed_count += 1
                failure_stages = (*failure_stages, failure_stage)
                errors = (*errors, f"ingestion failed during {failure_stage.value}")
            cursor_digest = job.cursor_for(reference.accession_number)
            if retry_claim is None:
                await self._save_checkpoint(
                    IngestionCheckpointStore.Record(
                        job_digest=job.job_digest,
                        cursor_digest=cursor_digest,
                        processed_count=total_success_count,
                        failed_count=failed_count,
                        complete=False,
                    ),
                    deadline=deadline,
                )
            break_if_retry = retry_claim is not None
            if break_if_retry:
                break
        if retry_claim is not None:
            terminal = IngestionRetryResult(
                job_digest=job.job_digest,
                attempt_digest=retry_claim.attempt_digest,
                state=(
                    IngestionRetryState.FAILED if failed_count else IngestionRetryState.SUCCEEDED
                ),
                failure_stage=failure_stages[0] if failure_stages else None,
                inserted_count=0 if failed_count else inserted_count,
                removed_count=0 if failed_count else removed_count,
            )
            await self._finish_retry(retry_claim, terminal, deadline=deadline)
            return ResearchIngestionResult(
                dry_run=False,
                job=job,
                planned_count=len(job.references),
                processed_count=processed_count,
                skipped_count=0,
                failed_count=failed_count,
                inserted_count=inserted_count,
                removed_count=removed_count,
                errors=tuple(dict.fromkeys(errors)),
                failure_stages=tuple(dict.fromkeys(failure_stages)),
                vector_verification_failures=tuple(dict.fromkeys(vector_verification_failures)),
            )
        new_failure_count = failed_count - (
            checkpoint.failed_count if checkpoint is not None else 0
        )
        complete = resume_index + processed_count + new_failure_count >= len(job.references)
        await self._save_checkpoint(
            IngestionCheckpointStore.Record(
                job_digest=job.job_digest,
                cursor_digest=cursor_digest,
                processed_count=total_success_count,
                failed_count=failed_count,
                complete=complete,
            ),
            deadline=deadline,
        )
        return ResearchIngestionResult(
            dry_run=False,
            job=job,
            planned_count=len(job.references),
            processed_count=processed_count,
            skipped_count=resume_index,
            failed_count=failed_count,
            inserted_count=inserted_count,
            removed_count=removed_count,
            errors=tuple(dict.fromkeys(errors)),
            failure_stages=tuple(dict.fromkeys(failure_stages)),
            vector_verification_failures=tuple(dict.fromkeys(vector_verification_failures)),
        )

    async def _claim_retry(
        self,
        job: ResearchIngestionJob,
        checkpoint: IngestionCheckpointStore.Record | None,
        *,
        deadline: RequestDeadline,
    ) -> tuple[IngestionRetryClaim | None, IngestionRetryResult | None]:
        if self._checkpoints is None:
            raise ResearchIngestionRetryError()
        try:
            snapshot = await self._checkpoints.load_retry_snapshot(
                job.job_digest,
                deadline=deadline,
            )
        except Exception:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
        if snapshot.checkpoint != checkpoint:
            raise ResearchIngestionRetryError()
        if snapshot.result is not None:
            if snapshot.result.state is IngestionRetryState.SUCCEEDED:
                return None, snapshot.result
            raise ResearchIngestionRetryError()
        if snapshot.claim is not None:
            raise ResearchIngestionRetryError()
        if (
            checkpoint is None
            or len(job.references) != 1
            or not checkpoint.complete
            or checkpoint.processed_count != 0
            or checkpoint.failed_count != 1
            or checkpoint.cursor_digest != job.cursor_for(job.references[0].accession_number)
        ):
            raise ResearchIngestionRetryError()
        claim = IngestionRetryClaim(
            job_digest=job.job_digest,
            attempt_digest=secrets.token_hex(32),
            checkpoint_digest=_digest(_checkpoint_json(checkpoint)),
            cursor_digest=checkpoint.cursor_digest,
        )
        try:
            claimed = await self._checkpoints.claim_failed_retry(
                checkpoint,
                claim,
                deadline=deadline,
            )
        except Exception:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
        if not claimed:
            raise ResearchIngestionRetryError()
        return claim, None

    async def _finish_retry(
        self,
        claim: IngestionRetryClaim,
        result: IngestionRetryResult,
        *,
        deadline: RequestDeadline,
    ) -> None:
        if self._checkpoints is None:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
        try:
            finished = await self._checkpoints.finish_failed_retry(
                claim,
                result,
                deadline=deadline,
            )
        except Exception:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
        if not finished:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None

    async def _load_checkpoint(
        self,
        job_digest: str,
        *,
        deadline: RequestDeadline,
    ) -> IngestionCheckpointStore.Record | None:
        if self._checkpoints is None:
            return None
        try:
            return await self._checkpoints.load(job_digest, deadline=deadline)
        except Exception:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None

    async def _save_checkpoint(
        self,
        record: IngestionCheckpointStore.Record,
        *,
        deadline: RequestDeadline,
    ) -> None:
        if self._checkpoints is not None:
            try:
                await self._checkpoints.save(record, deadline=deadline)
            except Exception:
                raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
