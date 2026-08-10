from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Protocol

from app.research.control import (
    IngestionFailureStage,
    IngestionRetryAttemptTwoClaim,
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
    retry_failed_attempt_two: str | None = None
    retry_failed_attempt_two_authorization: str | None = field(default=None, repr=False)

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
        if self.retry_failed_attempt_two is not None and (
            not isinstance(self.retry_failed_attempt_two, str)
            or re.fullmatch(r"^[0-9a-f]{64}$", self.retry_failed_attempt_two) is None
            or not self.apply
            or self.retry_failed
            or self.limit != 1
        ):
            raise ValueError("failed retry attempt two requires apply, one filing, and a job ID")
        if (
            self.retry_failed_attempt_two_authorization is not None
            and (
                not isinstance(self.retry_failed_attempt_two_authorization, str)
                or re.fullmatch(
                    r"^[0-9a-f]{64}$",
                    self.retry_failed_attempt_two_authorization,
                )
                is None
                or self.retry_failed_attempt_two is None
            )
        ) or (
            self.retry_failed_attempt_two is not None
            and self.retry_failed_attempt_two_authorization is None
        ):
            raise ValueError("failed retry attempt two authorization is invalid")
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
    async def load_retry_attempt_two_snapshot(
        self, job_digest: str, *, deadline: RequestDeadline
    ) -> RetryAttemptTwoCheckpointSnapshot: ...
    async def claim_failed_retry_attempt_two(
        self,
        checkpoint: Record,
        first_claim: IngestionRetryClaim,
        first_result: IngestionRetryResult,
        claim: IngestionRetryAttemptTwoClaim,
        *,
        deadline: RequestDeadline,
    ) -> bool: ...
    async def finish_failed_retry_attempt_two(
        self,
        checkpoint: Record,
        first_claim: IngestionRetryClaim,
        first_result: IngestionRetryResult,
        claim: IngestionRetryAttemptTwoClaim,
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
class RetryAttemptTwoCheckpointSnapshot:
    checkpoint: IngestionCheckpointStore.Record
    first_claim: IngestionRetryClaim
    first_result: IngestionRetryResult
    claim: IngestionRetryAttemptTwoClaim | None
    result: IngestionRetryResult | None

    def __post_init__(self) -> None:
        RetryCheckpointSnapshot(self.checkpoint, self.first_claim, self.first_result)
        if (
            not self.checkpoint.complete
            or self.checkpoint.processed_count != 0
            or self.checkpoint.failed_count != 1
            or self.checkpoint.cursor_digest is None
            or self.first_result.state is not IngestionRetryState.FAILED
            or self.first_result.failure_stage is not IngestionFailureStage.VECTOR_VERIFICATION
        ):
            raise ValueError("legacy retry is not eligible for attempt two")
        if self.claim is None:
            if self.result is not None:
                raise ValueError("attempt-two result requires its immutable claim")
            return
        expected = (
            _digest(_checkpoint_json(self.checkpoint)),
            _digest(self.first_claim.to_json()),
            _digest(self.first_result.to_json()),
        )
        if (
            self.claim.job_digest != self.checkpoint.job_digest
            or self.claim.attempt_digest == self.first_claim.attempt_digest
            or self.claim.checkpoint_digest != expected[0]
            or self.claim.cursor_digest != self.checkpoint.cursor_digest
            or self.claim.first_claim_digest != expected[1]
            or self.claim.first_result_digest != expected[2]
        ):
            raise ValueError("attempt-two claim does not match immutable legacy records")
        if self.result is not None and (
            self.result.job_digest != self.claim.job_digest
            or self.result.attempt_digest != self.claim.attempt_digest
        ):
            raise ValueError("attempt-two result does not match its immutable claim")


class ResearchIngestionRecoveryState(StrEnum):
    NOT_REQUESTED = "not_requested"
    ATTEMPT_TWO_SUCCEEDED = "attempt_two_succeeded"
    ATTEMPT_TWO_FAILED = "attempt_two_failed"
    ATTEMPT_TWO_ALREADY_SUCCEEDED = "attempt_two_already_succeeded"
    ATTEMPT_TWO_ALREADY_FAILED = "attempt_two_already_failed"


@dataclass(frozen=True, slots=True)
class _AttemptTwoContext:
    checkpoint: IngestionCheckpointStore.Record
    first_claim: IngestionRetryClaim
    first_result: IngestionRetryResult
    claim: IngestionRetryAttemptTwoClaim


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
    recovery_state: ResearchIngestionRecoveryState = ResearchIngestionRecoveryState.NOT_REQUESTED

    def __post_init__(self) -> None:
        diagnostics = tuple(self.vector_verification_failures)
        if not isinstance(self.recovery_state, ResearchIngestionRecoveryState):
            raise ValueError("recovery state is invalid")
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
        attempt_two: _AttemptTwoContext | None = None
        if request.retry_failed_attempt_two is not None:
            authorization_digest = request.retry_failed_attempt_two_authorization
            if authorization_digest is None:  # pragma: no cover - validated by request
                raise ResearchIngestionRetryError()
            attempt_two, prior_result = await self._claim_retry_attempt_two(
                request.retry_failed_attempt_two,
                authorization_digest,
                deadline=deadline,
            )
            if prior_result is not None:
                return self._attempt_two_replay_result(request, prior_result)
        try:
            page = await self._source.discover(request.discovery_request(), deadline=deadline)
        except TimeoutError:
            if attempt_two is not None:
                await self._finish_attempt_two_failure(
                    attempt_two,
                    IngestionFailureStage.SEC_FETCH,
                    deadline=deadline,
                )
            raise
        except Exception:
            if attempt_two is not None:
                await self._finish_attempt_two_failure(
                    attempt_two,
                    IngestionFailureStage.SEC_FETCH,
                    deadline=deadline,
                )
            raise IngestionStageError(IngestionFailureStage.SEC_FETCH) from None
        try:
            job = ResearchIngestionJob.from_plan(request, tuple(page.references))
        except Exception:
            if attempt_two is not None:
                await self._finish_attempt_two_failure(
                    attempt_two,
                    IngestionFailureStage.CHECKPOINTING,
                    deadline=deadline,
                )
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
        if attempt_two is not None and (
            job.job_digest != attempt_two.checkpoint.job_digest
            or len(job.references) != 1
            or attempt_two.checkpoint.cursor_digest
            != job.cursor_for(job.references[0].accession_number)
        ):
            await self._finish_attempt_two_failure(
                attempt_two,
                IngestionFailureStage.CHECKPOINTING,
                deadline=deadline,
            )
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
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
        checkpoint = (
            attempt_two.checkpoint
            if attempt_two is not None
            else await self._load_checkpoint(job.job_digest, deadline=deadline)
        )
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
        if retry_claim is not None or attempt_two is not None:
            resume_index = 0
        failed_count = checkpoint.failed_count if checkpoint is not None else 0
        total_success_count = checkpoint.processed_count if checkpoint is not None else 0
        if retry_claim is not None or attempt_two is not None:
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
                if retry_claim is None and attempt_two is None:
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
            if retry_claim is None and attempt_two is None:
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
            break_if_retry = retry_claim is not None or attempt_two is not None
            if break_if_retry:
                break
        if retry_claim is not None or attempt_two is not None:
            if retry_claim is not None:
                attempt_digest = retry_claim.attempt_digest
            else:
                if attempt_two is None:  # pragma: no cover - guarded by outer condition
                    raise IngestionStageError(IngestionFailureStage.CHECKPOINTING)
                attempt_digest = attempt_two.claim.attempt_digest
            terminal = IngestionRetryResult(
                job_digest=job.job_digest,
                attempt_digest=attempt_digest,
                state=(
                    IngestionRetryState.FAILED if failed_count else IngestionRetryState.SUCCEEDED
                ),
                failure_stage=failure_stages[0] if failure_stages else None,
                inserted_count=0 if failed_count else inserted_count,
                removed_count=0 if failed_count else removed_count,
            )
            if retry_claim is not None:
                await self._finish_retry(retry_claim, terminal, deadline=deadline)
            else:
                if attempt_two is None:  # pragma: no cover - guarded by outer condition
                    raise IngestionStageError(IngestionFailureStage.CHECKPOINTING)
                await self._finish_retry_attempt_two(attempt_two, terminal, deadline=deadline)
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
                recovery_state=(
                    ResearchIngestionRecoveryState.ATTEMPT_TWO_FAILED
                    if attempt_two is not None and failed_count
                    else ResearchIngestionRecoveryState.ATTEMPT_TWO_SUCCEEDED
                    if attempt_two is not None
                    else ResearchIngestionRecoveryState.NOT_REQUESTED
                ),
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

    async def _claim_retry_attempt_two(
        self,
        job_digest: str,
        authorization_digest: str,
        *,
        deadline: RequestDeadline,
    ) -> tuple[_AttemptTwoContext | None, IngestionRetryResult | None]:
        if self._checkpoints is None:
            raise ResearchIngestionRetryError()
        try:
            snapshot = await self._checkpoints.load_retry_attempt_two_snapshot(
                job_digest,
                deadline=deadline,
            )
        except Exception:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
        if not secrets.compare_digest(
            authorization_digest,
            snapshot.first_claim.attempt_digest,
        ):
            raise ResearchIngestionRetryError()
        if snapshot.result is not None:
            return None, snapshot.result
        if snapshot.claim is not None:
            raise ResearchIngestionRetryError()
        cursor_digest = snapshot.checkpoint.cursor_digest
        if cursor_digest is None:
            raise ResearchIngestionRetryError()
        claim = IngestionRetryAttemptTwoClaim(
            job_digest=job_digest,
            attempt_digest=secrets.token_hex(32),
            checkpoint_digest=_digest(_checkpoint_json(snapshot.checkpoint)),
            cursor_digest=cursor_digest,
            first_claim_digest=_digest(snapshot.first_claim.to_json()),
            first_result_digest=_digest(snapshot.first_result.to_json()),
        )
        try:
            claimed = await self._checkpoints.claim_failed_retry_attempt_two(
                snapshot.checkpoint,
                snapshot.first_claim,
                snapshot.first_result,
                claim,
                deadline=deadline,
            )
        except Exception:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
        if not claimed:
            raise ResearchIngestionRetryError()
        return (
            _AttemptTwoContext(
                checkpoint=snapshot.checkpoint,
                first_claim=snapshot.first_claim,
                first_result=snapshot.first_result,
                claim=claim,
            ),
            None,
        )

    @staticmethod
    def _attempt_two_replay_result(
        request: ResearchIngestionRequest,
        terminal: IngestionRetryResult,
    ) -> ResearchIngestionResult:
        failed = terminal.state is IngestionRetryState.FAILED
        failure_stages = () if terminal.failure_stage is None else (terminal.failure_stage,)
        errors = (
            ()
            if terminal.failure_stage is None
            else (f"ingestion failed during {terminal.failure_stage.value}",)
        )
        return ResearchIngestionResult(
            dry_run=False,
            job=ResearchIngestionJob(
                request=request,
                references=(),
                job_digest=terminal.job_digest,
            ),
            planned_count=1,
            processed_count=0,
            skipped_count=1,
            failed_count=int(failed),
            inserted_count=0,
            removed_count=0,
            errors=errors,
            failure_stages=failure_stages,
            recovery_state=(
                ResearchIngestionRecoveryState.ATTEMPT_TWO_ALREADY_FAILED
                if failed
                else ResearchIngestionRecoveryState.ATTEMPT_TWO_ALREADY_SUCCEEDED
            ),
        )

    async def _finish_attempt_two_failure(
        self,
        context: _AttemptTwoContext,
        stage: IngestionFailureStage,
        *,
        deadline: RequestDeadline,
    ) -> None:
        await self._finish_retry_attempt_two(
            context,
            IngestionRetryResult(
                job_digest=context.claim.job_digest,
                attempt_digest=context.claim.attempt_digest,
                state=IngestionRetryState.FAILED,
                failure_stage=stage,
                inserted_count=0,
                removed_count=0,
            ),
            deadline=deadline,
        )

    async def _finish_retry_attempt_two(
        self,
        context: _AttemptTwoContext,
        result: IngestionRetryResult,
        *,
        deadline: RequestDeadline,
    ) -> None:
        if self._checkpoints is None:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
        try:
            finished = await self._checkpoints.finish_failed_retry_attempt_two(
                context.checkpoint,
                context.first_claim,
                context.first_result,
                context.claim,
                result,
                deadline=deadline,
            )
        except Exception:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None
        if not finished:
            raise IngestionStageError(IngestionFailureStage.CHECKPOINTING) from None

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
