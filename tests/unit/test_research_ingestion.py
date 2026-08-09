from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from app.research.control import (
    IngestionFailureStage,
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
    IngestionResult,
    RawFiling,
)
from app.research.ingestion import (
    IngestionCheckpointStore,
    ResearchIngestionJob,
    ResearchIngestionRequest,
    ResearchIngestionRetryError,
    ResearchIngestionRunner,
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
