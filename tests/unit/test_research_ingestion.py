from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

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
    FilingDiscoveryPage,
    FilingDiscoveryRequest,
    FilingDocument,
    FilingReference,
    GenerationVerification,
    GenerationVerificationOutcome,
    GenerationVerificationReason,
    IngestionResult,
    RawFiling,
)
from app.research.ingestion import (
    IngestionCheckpointStore,
    ResearchIngestionJob,
    ResearchIngestionRequest,
    ResearchIngestionResult,
    ResearchIngestionRetryError,
    ResearchIngestionRunner,
    RetryAttemptTwoCheckpointSnapshot,
    RetryCheckpointSnapshot,
)


def reference(
    accession: str = "0000320193-25-000001",
    *,
    symbol: str = "AAPL",
    filed: date = date(2025, 10, 31),
) -> FilingReference:
    return FilingReference(
        symbol=symbol,
        cik="0000320193",
        accession_number=accession,
        filing_type="10-K",
        title=f"{symbol} Form 10-K",
        filed_date=filed,
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/"
            f"{accession.replace('-', '')}/aapl-20250927.htm"
        ),
    )


class FakeSource:
    def __init__(self, references: tuple[FilingReference, ...]) -> None:
        self.references = references
        self.discoveries: tuple[FilingDiscoveryRequest, ...] = ()
        self.fetches: tuple[str, ...] = ()

    async def discover(
        self,
        request: FilingDiscoveryRequest,
        *,
        deadline: RequestDeadline,
    ) -> FilingDiscoveryPage:
        deadline.raise_if_expired()
        self.discoveries = (*self.discoveries, request)
        return FilingDiscoveryPage(references=self.references[: request.limit], next_cursor=None)

    async def fetch(self, item: FilingReference, *, deadline: RequestDeadline) -> RawFiling:
        deadline.raise_if_expired()
        self.fetches = (*self.fetches, item.accession_number)
        return RawFiling(reference=item, media_type="text/html", body=b"<html>risk text</html>")


class FakeParser:
    def parse(self, filing: RawFiling) -> FilingDocument:
        return FilingDocument(
            symbol=filing.reference.symbol,
            cik=filing.reference.cik,
            accession_number=filing.reference.accession_number,
            filing_type=filing.reference.filing_type,
            title=filing.reference.title,
            filed_date=filing.reference.filed_date,
            source_url=filing.reference.source_url,
            text="risk text",
        )


class FakeCore:
    def __init__(self) -> None:
        self.ingested: tuple[str, ...] = ()

    async def ingest(
        self,
        document: FilingDocument,
        *,
        deadline: RequestDeadline | None = None,
        retry_failed: bool = False,
    ) -> IngestionResult:
        if deadline is not None:
            deadline.raise_if_expired()
        self.ingested = (*self.ingested, document.accession_number)
        return IngestionResult(
            outcome="created",
            inserted_count=2,
            removed_count=0,
        )


@dataclass(frozen=True, slots=True)
class MemoryCheckpoints:
    Record = IngestionCheckpointStore.Record

    values: dict[str, IngestionCheckpointStore.Record]
    claims: dict[str, IngestionRetryClaim] = field(default_factory=dict)
    results: dict[str, IngestionRetryResult] = field(default_factory=dict)
    attempt_two_claims: dict[str, IngestionRetryAttemptTwoClaim] = field(default_factory=dict)
    attempt_two_results: dict[str, IngestionRetryResult] = field(default_factory=dict)

    async def load(
        self,
        job_digest: str,
        *,
        deadline: RequestDeadline,
    ) -> IngestionCheckpointStore.Record | None:
        deadline.raise_if_expired()
        return self.values.get(job_digest)

    async def save(
        self,
        record: IngestionCheckpointStore.Record,
        *,
        deadline: RequestDeadline,
    ) -> None:
        deadline.raise_if_expired()
        self.values[record.job_digest] = record

    async def load_retry_snapshot(
        self,
        job_digest: str,
        *,
        deadline: RequestDeadline,
    ) -> RetryCheckpointSnapshot:
        deadline.raise_if_expired()
        return RetryCheckpointSnapshot(
            checkpoint=self.values.get(job_digest),
            claim=self.claims.get(job_digest),
            result=self.results.get(job_digest),
        )

    async def claim_failed_retry(
        self,
        checkpoint: IngestionCheckpointStore.Record,
        claim: IngestionRetryClaim,
        *,
        deadline: RequestDeadline,
    ) -> bool:
        deadline.raise_if_expired()
        if (
            self.values.get(checkpoint.job_digest) != checkpoint
            or checkpoint.job_digest in self.claims
            or checkpoint.job_digest in self.results
        ):
            return False
        self.claims[checkpoint.job_digest] = claim
        return True

    async def finish_failed_retry(
        self,
        claim: IngestionRetryClaim,
        result: IngestionRetryResult,
        *,
        deadline: RequestDeadline,
    ) -> bool:
        deadline.raise_if_expired()
        if self.claims.get(claim.job_digest) != claim:
            return False
        existing = self.results.get(claim.job_digest)
        if existing is not None:
            return existing == result
        self.results[claim.job_digest] = result
        return True

    async def load_retry_attempt_two_snapshot(
        self,
        job_digest: str,
        *,
        deadline: RequestDeadline,
    ) -> RetryAttemptTwoCheckpointSnapshot:
        deadline.raise_if_expired()
        return RetryAttemptTwoCheckpointSnapshot(
            checkpoint=self.values.get(job_digest),
            first_claim=self.claims.get(job_digest),
            first_result=self.results.get(job_digest),
            claim=self.attempt_two_claims.get(job_digest),
            result=self.attempt_two_results.get(job_digest),
        )

    async def claim_failed_retry_attempt_two(
        self,
        checkpoint: IngestionCheckpointStore.Record,
        first_claim: IngestionRetryClaim,
        first_result: IngestionRetryResult,
        claim: IngestionRetryAttemptTwoClaim,
        *,
        deadline: RequestDeadline,
    ) -> bool:
        deadline.raise_if_expired()
        if (
            self.values.get(checkpoint.job_digest) != checkpoint
            or self.claims.get(checkpoint.job_digest) != first_claim
            or self.results.get(checkpoint.job_digest) != first_result
            or checkpoint.job_digest in self.attempt_two_claims
            or checkpoint.job_digest in self.attempt_two_results
        ):
            return False
        self.attempt_two_claims[checkpoint.job_digest] = claim
        return True

    async def finish_failed_retry_attempt_two(
        self,
        checkpoint: IngestionCheckpointStore.Record,
        first_claim: IngestionRetryClaim,
        first_result: IngestionRetryResult,
        claim: IngestionRetryAttemptTwoClaim,
        result: IngestionRetryResult,
        *,
        deadline: RequestDeadline,
    ) -> bool:
        deadline.raise_if_expired()
        if (
            self.values.get(claim.job_digest) != checkpoint
            or self.claims.get(claim.job_digest) != first_claim
            or self.results.get(claim.job_digest) != first_result
            or self.attempt_two_claims.get(claim.job_digest) != claim
        ):
            return False
        existing = self.attempt_two_results.get(claim.job_digest)
        if existing is not None:
            return existing == result
        self.attempt_two_results[claim.job_digest] = result
        return True


def request(**overrides: Any) -> ResearchIngestionRequest:
    values = {
        "symbol": "AAPL",
        "cik": "0000320193",
        "filing_types": ("10-K", "10-Q"),
        "date_from": date(2025, 1, 1),
        "date_to": date(2025, 12, 31),
        "limit": 2,
        "apply": False,
    }
    return ResearchIngestionRequest(**{**values, **overrides})


def exception_graph_text(error: BaseException) -> str:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    rendered: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered.extend((repr(current), str(current)))
        traceback = current.__traceback__
        while traceback is not None:
            filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
            if filename.endswith("app/research/ingestion.py"):
                rendered.extend(
                    repr(value)
                    for value in traceback.tb_frame.f_locals.values()
                    if not inspect.iscoroutine(value)
                )
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return "".join(rendered)


def seed_attempt_two_legacy(
    checkpoints: MemoryCheckpoints,
    item: FilingReference,
    *,
    request_value: ResearchIngestionRequest | None = None,
    failure_stage: IngestionFailureStage = IngestionFailureStage.VECTOR_VERIFICATION,
) -> tuple[ResearchIngestionJob, IngestionCheckpointStore.Record]:
    selected_request = request_value or request(limit=1, apply=True)
    job = ResearchIngestionJob.from_plan(selected_request, (item,))
    checkpoint = IngestionCheckpointStore.Record(
        job_digest=job.job_digest,
        cursor_digest=job.cursor_for(item.accession_number),
        processed_count=0,
        failed_count=1,
        complete=True,
    )
    first_claim = IngestionRetryClaim(
        job_digest=job.job_digest,
        attempt_digest="1" * 64,
        checkpoint_digest=hashlib.sha256(
            (
                '{"complete":true,"cursor_digest":"'
                + checkpoint.cursor_digest
                + '","failed_count":1,"job_digest":"'
                + checkpoint.job_digest
                + '","processed_count":0}'
            ).encode()
        ).hexdigest(),
        cursor_digest=checkpoint.cursor_digest,
    )
    first_result = IngestionRetryResult(
        job_digest=job.job_digest,
        attempt_digest=first_claim.attempt_digest,
        state=IngestionRetryState.FAILED,
        failure_stage=failure_stage,
        inserted_count=0,
        removed_count=0,
    )
    checkpoints.values[job.job_digest] = checkpoint
    checkpoints.claims[job.job_digest] = first_claim
    checkpoints.results[job.job_digest] = first_result
    return job, checkpoint


@pytest.mark.asyncio
async def test_ingestion_request_validates_allowlists_before_adapter_calls() -> None:
    with pytest.raises(ValueError, match="symbol"):
        ResearchIngestionRequest(
            symbol="bad symbol!",
            cik="0000320193",
            filing_types=("10-K",),
            date_from=date(2025, 1, 1),
            date_to=date(2025, 1, 31),
            limit=1,
        )

    source = FakeSource((reference(),))
    runner = ResearchIngestionRunner(source=source, parser=FakeParser(), core=FakeCore())

    result = await runner.run(request(limit=1), deadline=RequestDeadline.after(30.0))

    assert result.dry_run is True
    assert source.discoveries[0].symbol == "AAPL"
    assert source.discoveries[0].filing_types == ("10-K", "10-Q")
    assert source.fetches == ()


@pytest.mark.asyncio
async def test_dry_run_freezes_bounded_discovery_plan_before_fetch() -> None:
    refs = (reference("0000320193-25-000002"), reference("0000320193-25-000001"))
    source = FakeSource(refs)
    runner = ResearchIngestionRunner(source=source, parser=FakeParser(), core=FakeCore())

    result = await runner.run(request(limit=2), deadline=RequestDeadline.after(30.0))

    assert isinstance(result.job, ResearchIngestionJob)
    assert result.job.references == refs
    assert result.planned_count == 2
    assert result.processed_count == 0
    assert result.opaque_accession_ids == tuple(
        hashlib.sha256(item.accession_number.encode()).hexdigest() for item in refs
    )
    assert source.fetches == ()


@pytest.mark.asyncio
async def test_apply_fetches_parses_and_delegates_atomic_core_ingest_after_plan() -> None:
    refs = (reference("0000320193-25-000002"), reference("0000320193-25-000001"))
    source = FakeSource(refs)
    core = FakeCore()
    runner = ResearchIngestionRunner(
        source=source,
        parser=FakeParser(),
        core=core,
        checkpoints=MemoryCheckpoints({}),
    )

    result = await runner.run(request(limit=2, apply=True), deadline=RequestDeadline.after(30.0))

    assert source.fetches == ("0000320193-25-000002", "0000320193-25-000001")
    assert core.ingested == source.fetches
    assert result.processed_count == 2
    assert result.inserted_count == 4
    assert result.removed_count == 0
    assert result.failed_count == 0


@pytest.mark.asyncio
async def test_checkpoint_resume_skips_processed_accessions_without_refetch() -> None:
    first = reference("0000320193-25-000001")
    second = reference("0000320193-25-000002")
    source = FakeSource((first, second))
    core = FakeCore()
    job_request = request(limit=2, apply=True)
    checkpoints = MemoryCheckpoints({})
    dry_runner = ResearchIngestionRunner(
        source=source,
        parser=FakeParser(),
        core=core,
        checkpoints=checkpoints,
    )
    dry = await dry_runner.run(request(limit=2), deadline=RequestDeadline.after(30.0))
    await checkpoints.save(
        IngestionCheckpointStore.Record(
            job_digest=dry.job.job_digest,
            cursor_digest=dry.job.cursor_for(first.accession_number),
            processed_count=1,
            failed_count=0,
            complete=False,
        ),
        deadline=RequestDeadline.after(30.0),
    )

    result = await dry_runner.run(job_request, deadline=RequestDeadline.after(30.0))

    assert source.fetches == (second.accession_number,)
    assert core.ingested == (second.accession_number,)
    assert result.skipped_count == 1
    assert result.processed_count == 1


@pytest.mark.asyncio
async def test_checkpoint_resume_uses_cursor_without_storing_raw_accessions() -> None:
    first = reference("0000320193-25-000001")
    second = reference("0000320193-25-000002")
    checkpoints = MemoryCheckpoints({})
    runner = ResearchIngestionRunner(
        source=FakeSource((first, second)),
        parser=FakeParser(),
        core=FakeCore(),
        checkpoints=checkpoints,
    )

    result = await runner.run(request(limit=2, apply=True), deadline=RequestDeadline.after(30.0))

    saved = checkpoints.values[result.opaque_job_id]
    assert saved.cursor_digest == result.job.cursor_for(second.accession_number)
    assert saved.processed_count == 2
    assert first.accession_number not in repr(saved)
    assert second.accession_number not in repr(saved)


@pytest.mark.asyncio
async def test_failure_records_checkpoint_without_leaking_bodies_urls_or_traceback_context() -> (
    None
):
    secret_body = "private body and URL https://www.sec.gov/Archives/edgar/data/private"

    class FailingParser(FakeParser):
        def parse(self, filing: RawFiling) -> FilingDocument:
            del filing
            raise RuntimeError(secret_body)

    checkpoints = MemoryCheckpoints({})
    runner = ResearchIngestionRunner(
        source=FakeSource((reference(),)),
        parser=FailingParser(),
        core=FakeCore(),
        checkpoints=checkpoints,
    )

    result = await runner.run(request(limit=1, apply=True), deadline=RequestDeadline.after(30.0))

    assert result.failed_count == 1
    assert result.errors == ("ingestion failed during parsing_chunking",)
    assert result.failure_stages == (IngestionFailureStage.PARSING_CHUNKING,)
    rendered = repr(result) + str(result)
    assert "private body" not in rendered
    assert "sec.gov" not in rendered
    saved = next(iter(checkpoints.values.values()))
    assert saved.failed_count == 1


@pytest.mark.asyncio
async def test_exact_legacy_failure_requires_explicit_retry_and_recovers_once() -> None:
    item = reference()
    checkpoints = MemoryCheckpoints({})
    source = FakeSource((item,))
    core = FakeCore()
    runner = ResearchIngestionRunner(
        source=source,
        parser=FakeParser(),
        core=core,
        checkpoints=checkpoints,
    )
    dry = await runner.run(request(limit=1), deadline=RequestDeadline.after(30.0))
    legacy = IngestionCheckpointStore.Record(
        job_digest=dry.opaque_job_id,
        cursor_digest=dry.job.cursor_for(item.accession_number),
        processed_count=0,
        failed_count=1,
        complete=True,
    )
    checkpoints.values[dry.opaque_job_id] = legacy

    skipped = await runner.run(request(limit=1, apply=True), deadline=RequestDeadline.after(30.0))
    recovered = await runner.run(
        request(limit=1, apply=True, retry_failed=True),
        deadline=RequestDeadline.after(30.0),
    )
    replayed = await runner.run(
        request(limit=1, apply=True, retry_failed=True),
        deadline=RequestDeadline.after(30.0),
    )

    assert skipped.processed_count == 0
    assert skipped.skipped_count == 1
    assert recovered.processed_count == 1
    assert recovered.failed_count == 0
    assert replayed.processed_count == 0
    assert replayed.skipped_count == 1
    assert core.ingested == (item.accession_number,)
    assert checkpoints.values[dry.opaque_job_id] == legacy
    assert checkpoints.results[dry.opaque_job_id].state is IngestionRetryState.SUCCEEDED


def test_attempt_two_request_requires_apply_opaque_job_and_excludes_first_retry() -> None:
    with pytest.raises(ValueError, match="attempt two"):
        request(retry_failed_attempt_two="a" * 64)
    with pytest.raises(ValueError, match="attempt two"):
        request(apply=True, retry_failed_attempt_two="not-a-job")
    with pytest.raises(ValueError, match="attempt two"):
        request(
            apply=True,
            retry_failed=True,
            retry_failed_attempt_two="a" * 64,
        )

    with pytest.raises(ValueError, match="authorization"):
        request(limit=1, apply=True, retry_failed_attempt_two="a" * 64)
    with pytest.raises(ValueError, match="authorization"):
        request(
            limit=1,
            apply=True,
            retry_failed_attempt_two="a" * 64,
            retry_failed_attempt_two_authorization="not-a-digest",
        )
    with pytest.raises(ValueError, match="authorization"):
        request(
            retry_failed_attempt_two_authorization="b" * 64,
        )

    accepted = request(
        limit=1,
        apply=True,
        retry_failed_attempt_two="a" * 64,
        retry_failed_attempt_two_authorization="b" * 64,
    )
    assert accepted.retry_failed_attempt_two == "a" * 64
    assert accepted.retry_failed_attempt_two_authorization == "b" * 64
    assert "b" * 64 not in repr(accepted)


@pytest.mark.asyncio
async def test_attempt_two_wrong_authorization_makes_zero_claim_or_provider_calls() -> None:
    item = reference()
    checkpoints = MemoryCheckpoints({})
    job, _ = seed_attempt_two_legacy(checkpoints, item)
    source = FakeSource((item,))

    with pytest.raises(ResearchIngestionRetryError) as caught:
        await ResearchIngestionRunner(
            source=source,
            parser=FakeParser(),
            core=FakeCore(),
            checkpoints=checkpoints,
        ).run(
            request(
                limit=1,
                apply=True,
                retry_failed_attempt_two=job.job_digest,
                retry_failed_attempt_two_authorization="f" * 64,
            ),
            deadline=RequestDeadline.after(30.0),
        )

    assert checkpoints.attempt_two_claims == {}
    assert source.discoveries == ()
    assert source.fetches == ()
    assert "f" * 64 not in exception_graph_text(caught.value)


@pytest.mark.asyncio
async def test_attempt_two_claims_before_discovery_and_preserves_legacy_bytes() -> None:
    item = reference()
    checkpoints = MemoryCheckpoints({})
    request_value = request(limit=1, apply=True)
    job, checkpoint = seed_attempt_two_legacy(
        checkpoints,
        item,
        request_value=request_value,
    )
    legacy_bytes = (
        repr(checkpoint),
        checkpoints.claims[job.job_digest].to_json(),
        checkpoints.results[job.job_digest].to_json(),
    )

    class ClaimAwareSource(FakeSource):
        async def discover(self, discovery_request, *, deadline):
            assert job.job_digest in checkpoints.attempt_two_claims
            return await super().discover(discovery_request, deadline=deadline)

    core = FakeCore()
    runner = ResearchIngestionRunner(
        source=ClaimAwareSource((item,)),
        parser=FakeParser(),
        core=core,
        checkpoints=checkpoints,
    )
    result = await runner.run(
        request(
            limit=1,
            apply=True,
            retry_failed_attempt_two=job.job_digest,
            retry_failed_attempt_two_authorization=checkpoints.claims[
                job.job_digest
            ].attempt_digest,
        ),
        deadline=RequestDeadline.after(30.0),
    )

    assert result.processed_count == 1
    assert result.failed_count == 0
    assert checkpoints.attempt_two_results[job.job_digest].state is IngestionRetryState.SUCCEEDED
    assert (
        repr(checkpoints.values[job.job_digest]),
        checkpoints.claims[job.job_digest].to_json(),
        checkpoints.results[job.job_digest].to_json(),
    ) == legacy_bytes


@pytest.mark.asyncio
async def test_attempt_two_terminal_success_is_idempotent_with_zero_provider_calls() -> None:
    item = reference()
    checkpoints = MemoryCheckpoints({})
    job, _ = seed_attempt_two_legacy(checkpoints, item)
    source = FakeSource((item,))
    core = FakeCore()
    runner = ResearchIngestionRunner(
        source=source,
        parser=FakeParser(),
        core=core,
        checkpoints=checkpoints,
    )
    selected = request(
        limit=1,
        apply=True,
        retry_failed_attempt_two=job.job_digest,
        retry_failed_attempt_two_authorization=checkpoints.claims[job.job_digest].attempt_digest,
    )
    first = await runner.run(selected, deadline=RequestDeadline.after(30.0))
    discoveries_after_first = source.discoveries
    fetches_after_first = source.fetches
    ingested_after_first = core.ingested

    repeated = await runner.run(selected, deadline=RequestDeadline.after(30.0))

    assert first.processed_count == 1
    assert repeated.opaque_job_id == job.job_digest
    assert repeated.processed_count == 0
    assert repeated.skipped_count == 1
    assert source.discoveries == discoveries_after_first
    assert source.fetches == fetches_after_first
    assert core.ingested == ingested_after_first


@pytest.mark.asyncio
async def test_attempt_two_failed_terminal_and_stale_claim_never_call_providers() -> None:
    sensitive_sentinel = "private vector payload https://vector.invalid secret-token"

    class FailingCore(FakeCore):
        async def ingest(self, document, *, deadline=None, retry_failed=False):
            del document, retry_failed
            if deadline is not None:
                deadline.raise_if_expired()
            try:
                raise RuntimeError(sensitive_sentinel)
            except RuntimeError:
                raise IngestionStageError(IngestionFailureStage.VECTOR_VERIFICATION) from None

    item = reference()
    checkpoints = MemoryCheckpoints({})
    job, _ = seed_attempt_two_legacy(checkpoints, item)
    source = FakeSource((item,))
    runner = ResearchIngestionRunner(
        source=source,
        parser=FakeParser(),
        core=FailingCore(),
        checkpoints=checkpoints,
    )
    selected = request(
        limit=1,
        apply=True,
        retry_failed_attempt_two=job.job_digest,
        retry_failed_attempt_two_authorization=checkpoints.claims[job.job_digest].attempt_digest,
    )

    failed = await runner.run(selected, deadline=RequestDeadline.after(30.0))
    calls_after_failure = (source.discoveries, source.fetches)
    replayed = await runner.run(selected, deadline=RequestDeadline.after(30.0))

    assert failed.failure_stages == (IngestionFailureStage.VECTOR_VERIFICATION,)
    assert replayed.failed_count == 1
    assert replayed.recovery_state.value == "attempt_two_already_failed"
    assert checkpoints.attempt_two_results[job.job_digest].state is IngestionRetryState.FAILED
    assert (source.discoveries, source.fetches) == calls_after_failure
    assert sensitive_sentinel not in repr(replayed)

    checkpoints.attempt_two_results.clear()
    with pytest.raises(ResearchIngestionRetryError):
        await runner.run(selected, deadline=RequestDeadline.after(30.0))
    assert (source.discoveries, source.fetches) == calls_after_failure


@pytest.mark.asyncio
async def test_attempt_two_rejects_concurrent_claim_and_discovery_ambiguity_before_fetch() -> None:
    item = reference()
    checkpoints = MemoryCheckpoints({})
    job, _ = seed_attempt_two_legacy(checkpoints, item)

    class LosingCheckpoints(MemoryCheckpoints):
        async def claim_failed_retry_attempt_two(self, *args, **kwargs):
            del args, kwargs
            return False

    losing = LosingCheckpoints(
        checkpoints.values,
        checkpoints.claims,
        checkpoints.results,
    )
    losing_source = FakeSource((item,))
    selected = request(
        limit=1,
        apply=True,
        retry_failed_attempt_two=job.job_digest,
        retry_failed_attempt_two_authorization=checkpoints.claims[job.job_digest].attempt_digest,
    )
    with pytest.raises(ResearchIngestionRetryError):
        await ResearchIngestionRunner(
            source=losing_source,
            parser=FakeParser(),
            core=FakeCore(),
            checkpoints=losing,
        ).run(selected, deadline=RequestDeadline.after(30.0))
    assert losing_source.discoveries == ()

    ambiguous_source = FakeSource((reference("0000320193-25-000002"),))
    with pytest.raises(IngestionStageError) as caught:
        await ResearchIngestionRunner(
            source=ambiguous_source,
            parser=FakeParser(),
            core=FakeCore(),
            checkpoints=checkpoints,
        ).run(selected, deadline=RequestDeadline.after(30.0))
    assert caught.value.stage is IngestionFailureStage.CHECKPOINTING
    assert len(ambiguous_source.discoveries) == 1
    assert ambiguous_source.fetches == ()


@pytest.mark.asyncio
async def test_attempt_two_discovery_timeout_terminalizes_when_deadline_remains() -> None:
    item = reference()
    checkpoints = MemoryCheckpoints({})
    job, _ = seed_attempt_two_legacy(checkpoints, item)

    class TimingOutSource(FakeSource):
        async def discover(self, discovery_request, *, deadline):
            del discovery_request, deadline
            raise TimeoutError("private upstream timeout detail")

    with pytest.raises(TimeoutError):
        await ResearchIngestionRunner(
            source=TimingOutSource((item,)),
            parser=FakeParser(),
            core=FakeCore(),
            checkpoints=checkpoints,
        ).run(
            request(
                limit=1,
                apply=True,
                retry_failed_attempt_two=job.job_digest,
                retry_failed_attempt_two_authorization=checkpoints.claims[
                    job.job_digest
                ].attempt_digest,
            ),
            deadline=RequestDeadline.after(30.0),
        )

    terminal = checkpoints.attempt_two_results[job.job_digest]
    assert terminal.state is IngestionRetryState.FAILED
    assert terminal.failure_stage is IngestionFailureStage.SEC_FETCH


@pytest.mark.asyncio
async def test_attempt_two_publication_before_finalization_is_not_eligible() -> None:
    item = reference()
    checkpoints = MemoryCheckpoints({})
    job, _ = seed_attempt_two_legacy(
        checkpoints,
        item,
        failure_stage=IngestionFailureStage.REDIS_PUBLICATION,
    )
    source = FakeSource((item,))

    with pytest.raises(IngestionStageError) as caught:
        await ResearchIngestionRunner(
            source=source,
            parser=FakeParser(),
            core=FakeCore(),
            checkpoints=checkpoints,
        ).run(
            request(
                limit=1,
                apply=True,
                retry_failed_attempt_two=job.job_digest,
                retry_failed_attempt_two_authorization=checkpoints.claims[
                    job.job_digest
                ].attempt_digest,
            ),
            deadline=RequestDeadline.after(30.0),
        )

    assert caught.value.stage is IngestionFailureStage.CHECKPOINTING
    assert source.discoveries == ()


@pytest.mark.asyncio
async def test_retry_failed_rejects_ambiguous_or_claimed_checkpoint_before_fetch() -> None:
    refs = (reference("0000320193-25-000001"), reference("0000320193-25-000002"))
    checkpoints = MemoryCheckpoints({})
    source = FakeSource(refs)
    runner = ResearchIngestionRunner(
        source=source,
        parser=FakeParser(),
        core=FakeCore(),
        checkpoints=checkpoints,
    )
    dry = await runner.run(request(limit=2), deadline=RequestDeadline.after(30.0))
    checkpoints.values[dry.opaque_job_id] = IngestionCheckpointStore.Record(
        job_digest=dry.opaque_job_id,
        cursor_digest=dry.job.cursor_for(refs[-1].accession_number),
        processed_count=0,
        failed_count=1,
        complete=True,
    )

    with pytest.raises(ResearchIngestionRetryError) as caught:
        await runner.run(
            request(limit=2, apply=True, retry_failed=True),
            deadline=RequestDeadline.after(30.0),
        )

    assert source.fetches == ()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert refs[0].source_url not in exception_graph_text(caught.value)


@pytest.mark.asyncio
async def test_retry_failure_records_only_fixed_stage_without_sensitive_exception_graph() -> None:
    sensitive_sentinel = "private filing content https://sec.invalid secret-token"

    class FailingCore(FakeCore):
        async def ingest(self, document, *, deadline=None, retry_failed=False):
            del document, retry_failed
            if deadline is not None:
                deadline.raise_if_expired()
            try:
                raise RuntimeError(sensitive_sentinel)
            except RuntimeError:
                raise IngestionStageError(IngestionFailureStage.EMBEDDING) from None

    item = reference()
    checkpoints = MemoryCheckpoints({})
    runner = ResearchIngestionRunner(
        source=FakeSource((item,)),
        parser=FakeParser(),
        core=FailingCore(),
        checkpoints=checkpoints,
    )
    dry = await runner.run(request(limit=1), deadline=RequestDeadline.after(30.0))
    checkpoints.values[dry.opaque_job_id] = IngestionCheckpointStore.Record(
        job_digest=dry.opaque_job_id,
        cursor_digest=dry.job.cursor_for(item.accession_number),
        processed_count=0,
        failed_count=1,
        complete=True,
    )

    failed = await runner.run(
        request(limit=1, apply=True, retry_failed=True),
        deadline=RequestDeadline.after(30.0),
    )

    assert failed.failure_stages == (IngestionFailureStage.EMBEDDING,)
    assert failed.errors == ("ingestion failed during embedding",)
    rendered = repr(failed) + repr(checkpoints.results[dry.opaque_job_id])
    assert sensitive_sentinel not in rendered


@pytest.mark.asyncio
async def test_runner_returns_sanitized_vector_verification_diagnostic_only() -> None:
    diagnostic = GenerationVerificationOutcome.failed(
        reason=GenerationVerificationReason.PARTIAL_VISIBILITY,
        expected_point_count=128,
        observed_point_count=96,
        null_point_count=32,
        attempt_count=4,
    )

    class FailingCore(FakeCore):
        async def ingest(self, document, *, deadline=None, retry_failed=False):
            del document, retry_failed
            if deadline is not None:
                deadline.raise_if_expired()
            raise IngestionStageError(
                IngestionFailureStage.VECTOR_VERIFICATION,
                verification=diagnostic,
            ) from None

    runner = ResearchIngestionRunner(
        source=FakeSource((reference(),)),
        parser=FakeParser(),
        core=FailingCore(),
    )

    result = await runner.run(
        request(limit=1, apply=True),
        deadline=RequestDeadline.after(30.0),
    )

    assert result.failure_stages == (IngestionFailureStage.VECTOR_VERIFICATION,)
    assert result.vector_verification_failures == (diagnostic,)
    assert "AAPL" not in repr(result.vector_verification_failures)


def test_result_rejects_success_proof_as_failure_diagnostic() -> None:
    item = reference()
    job = ResearchIngestionJob.from_plan(request(limit=1, apply=True), (item,))
    verified = GenerationVerificationOutcome.verified(
        verification=GenerationVerification.from_point_ids(
            "gen-" + "a" * 64,
            ("chunk-" + "b" * 64,),
        ),
        expected_point_count=1,
    )

    with pytest.raises(ValueError, match="diagnostics"):
        ResearchIngestionResult(
            dry_run=False,
            job=job,
            planned_count=1,
            processed_count=0,
            skipped_count=0,
            failed_count=1,
            inserted_count=0,
            removed_count=0,
            failure_stages=(IngestionFailureStage.VECTOR_VERIFICATION,),
            vector_verification_failures=(verified,),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["load", "claim", "finish"])
async def test_checkpoint_failures_detach_sensitive_traceback_locals(operation: str) -> None:
    sensitive_sentinel = "private-checkpoint-token-and-filing-text"

    class FailingCheckpoints(MemoryCheckpoints):
        async def load(self, job_digest, *, deadline):
            if operation == "load":
                raise RuntimeError(sensitive_sentinel)
            return await super().load(job_digest, deadline=deadline)

        async def claim_failed_retry(self, checkpoint, claim, *, deadline):
            if operation == "claim":
                raise RuntimeError(sensitive_sentinel)
            return await super().claim_failed_retry(checkpoint, claim, deadline=deadline)

        async def finish_failed_retry(self, claim, result, *, deadline):
            if operation == "finish":
                raise RuntimeError(sensitive_sentinel)
            return await super().finish_failed_retry(claim, result, deadline=deadline)

    item = reference()
    checkpoints = FailingCheckpoints({})
    runner = ResearchIngestionRunner(
        source=FakeSource((item,)),
        parser=FakeParser(),
        core=FakeCore(),
        checkpoints=checkpoints,
    )
    dry = await ResearchIngestionRunner(
        source=FakeSource((item,)),
        parser=FakeParser(),
        core=FakeCore(),
    ).run(request(limit=1), deadline=RequestDeadline.after(30.0))
    checkpoints.values[dry.opaque_job_id] = IngestionCheckpointStore.Record(
        job_digest=dry.opaque_job_id,
        cursor_digest=dry.job.cursor_for(item.accession_number),
        processed_count=0,
        failed_count=1,
        complete=True,
    )

    with pytest.raises(IngestionStageError) as caught:
        await runner.run(
            request(limit=1, apply=True, retry_failed=operation != "load"),
            deadline=RequestDeadline.after(30.0),
        )

    assert caught.value.stage is IngestionFailureStage.CHECKPOINTING
    assert sensitive_sentinel not in exception_graph_text(caught.value)
