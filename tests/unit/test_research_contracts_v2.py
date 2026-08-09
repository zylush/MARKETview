from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from datetime import date

import pytest

from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    EmbeddedChunk,
    EmbeddingDescriptor,
    EmbeddingVector,
    EvidenceChunk,
    FilingDiscoveryCursor,
    FilingDiscoveryPage,
    FilingDiscoveryRequest,
    FilingDocument,
    FilingReference,
    GenerationManifest,
    RawFiling,
)
from app.research.memory import InMemoryResearchControlPlane, InMemoryVectorStore
from app.research.ports import GenerationInspection, GenerationInspectionState


def reference() -> FilingReference:
    return FilingReference(
        symbol="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000001",
        filing_type="10-K",
        title="Apple 2025 Form 10-K",
        filed_date=date(2025, 10, 31),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/aapl-20250927.htm"
        ),
    )


def document() -> FilingDocument:
    filing = reference()
    return FilingDocument(
        symbol=filing.symbol,
        cik=filing.cik,
        accession_number=filing.accession_number,
        filing_type=filing.filing_type,
        title=filing.title,
        filed_date=filing.filed_date,
        source_url=filing.source_url,
        text="Apple reports that supply constraints could affect results.",
    )


def corpus() -> CorpusDescriptor:
    return CorpusDescriptor(
        corpus_version="sec-filings-v1",
        chunker_version="paragraph-800-100-v1",
        embedding=EmbeddingDescriptor(
            provider="openai",
            model="text-embedding-3-small",
            version="2024-01",
            dimensions=2,
        ),
    )


def test_discovery_request_and_page_are_bounded_and_immutable() -> None:
    request = FilingDiscoveryRequest(
        symbol="aapl",
        cik="0000320193",
        filing_types=("10-k", "8-k"),
        date_from=date(2024, 1, 1),
        date_to=date(2025, 12, 31),
        limit=25,
        cursor=FilingDiscoveryCursor("page-2"),
    )
    page = FilingDiscoveryPage(references=(reference(),), next_cursor=None)

    assert request.symbol == "AAPL"
    assert request.filing_types == ("10-K", "8-K")
    assert page.references == (reference(),)
    with pytest.raises(FrozenInstanceError):
        request.limit = 5  # type: ignore[misc]

    with pytest.raises(ValueError, match="between 1 and 100"):
        replace(request, limit=101)
    with pytest.raises(ValueError, match="date range"):
        replace(request, date_from=date(2026, 1, 1))
    with pytest.raises(ValueError, match="cursor"):
        FilingDiscoveryCursor("x" * 513)
    with pytest.raises(ValueError, match="page"):
        FilingDiscoveryPage(references=tuple(reference() for _ in range(101)))


def test_raw_filing_is_bound_to_reference_and_hides_sensitive_payload_in_repr() -> None:
    body = b"<html><body>filing</body></html>"
    raw = RawFiling(reference=reference(), media_type="text/html", body=body)

    assert raw.reference == reference()
    assert len(raw.content_hash) == 64
    assert "body" not in repr(raw)
    assert body.decode() not in repr(raw)
    assert raw.content_hash not in repr(raw)
    with pytest.raises(ValueError, match="media type"):
        replace(raw, media_type="application/json")


def test_filing_document_repr_does_not_disclose_filing_text() -> None:
    filing = document()

    assert filing.text not in repr(filing)


@pytest.mark.parametrize(
    "source_url",
    [
        "https://www.sec.gov/Archives/edgar/data/320193/../000032019325000001/aapl.htm",
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/../aapl.htm",
        "https://www.sec.gov/Archives/edgar/data/789019/000032019325000001/aapl.htm",
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325999999/aapl.htm",
    ],
)
def test_filing_reference_rejects_noncanonical_or_mismatched_archive_identity(
    source_url: str,
) -> None:
    with pytest.raises(ValueError, match="SEC Archives HTTPS"):
        FilingReference(
            symbol="AAPL",
            cik="0000320193",
            accession_number="0000320193-25-000001",
            filing_type="10-Q",
            title="Quarterly report",
            filed_date=date(2025, 8, 1),
            source_url=source_url,
        )


@pytest.mark.parametrize(
    "values",
    [(), (1.0,), (1.0, True), (1.0, math.nan), (1.0, math.inf)],
)
def test_embedding_vector_enforces_exact_finite_float_dimensions(
    values: tuple[object, ...],
) -> None:
    descriptor = corpus().embedding

    with pytest.raises(ValueError, match="embedding"):
        EmbeddingVector(descriptor=descriptor, values=values)  # type: ignore[arg-type]


def test_evidence_has_deterministic_canonical_ids_and_never_contains_vectors() -> None:
    first = EvidenceChunk.from_document(document(), corpus=corpus(), ordinal=0, text="Evidence.")
    second = EvidenceChunk.from_document(document(), corpus=corpus(), ordinal=0, text="Evidence.")
    changed = EvidenceChunk.from_document(document(), corpus=corpus(), ordinal=1, text="Evidence.")
    changed_text = EvidenceChunk.from_document(
        document(), corpus=corpus(), ordinal=0, text="Different evidence."
    )

    assert first == second
    assert first.chunk_id == second.chunk_id
    assert first.generation_id == second.generation_id
    assert first.chunk_id != changed.chunk_id
    assert first.chunk_id != changed_text.chunk_id
    assert first.evidence_digest != changed_text.evidence_digest
    assert not hasattr(first, "embedding")
    assert not hasattr(first, "vector")


def test_request_deadline_caps_children_and_fails_when_expired() -> None:
    now = [100.0]
    deadline = RequestDeadline.after(8.0, clock=lambda: now[0])

    assert deadline.remaining_seconds() == pytest.approx(8.0)
    assert deadline.child(2.0).remaining_seconds() == pytest.approx(2.0)
    now[0] = 109.0
    assert deadline.remaining_seconds() == 0.0
    with pytest.raises(TimeoutError, match="deadline"):
        deadline.raise_if_expired()


def test_generation_inspection_is_immutable_and_contains_only_safe_aggregate_state() -> None:
    inspection = GenerationInspection(
        state=GenerationInspectionState.PARTIAL,
        expected_point_count=128,
        observed_point_count=96,
    )

    assert inspection.state is GenerationInspectionState.PARTIAL
    assert inspection.expected_point_count == 128
    assert inspection.observed_point_count == 96
    assert not hasattr(inspection, "point_ids")
    assert not hasattr(inspection, "vectors")
    assert not hasattr(inspection, "data")
    with pytest.raises(FrozenInstanceError):
        inspection.observed_point_count = 128  # type: ignore[misc]


@pytest.mark.parametrize(
    ("state", "expected", "observed"),
    [
        (GenerationInspectionState.ABSENT, 2, 1),
        (GenerationInspectionState.EXACT, 2, 1),
        (GenerationInspectionState.PARTIAL, 2, 0),
        (GenerationInspectionState.PARTIAL, 2, 2),
    ],
)
def test_generation_inspection_rejects_impossible_state_counts(
    state: GenerationInspectionState,
    expected: int,
    observed: int,
) -> None:
    with pytest.raises(ValueError, match="inspection"):
        GenerationInspection(
            state=state,
            expected_point_count=expected,
            observed_point_count=observed,
        )


def _inspection_fixture() -> tuple[GenerationManifest, tuple[EmbeddedChunk, ...]]:
    configured = corpus()
    evidence = tuple(
        EvidenceChunk.from_document(
            document(),
            corpus=configured,
            ordinal=ordinal,
            text=text,
        )
        for ordinal, text in enumerate(("First disclosure.", "Second disclosure."))
    )
    chunks = tuple(
        EmbeddedChunk(
            evidence=item,
            embedding=EmbeddingVector(
                descriptor=configured.embedding,
                values=(1.0, 0.0) if index == 0 else (0.0, 1.0),
            ),
        )
        for index, item in enumerate(evidence)
    )
    first = evidence[0]
    return (
        GenerationManifest(
            corpus=configured,
            symbol=first.symbol,
            accession_number=first.accession_number,
            generation_id=first.generation_id,
            content_hash=first.content_hash,
            chunk_ids=tuple(item.chunk_id for item in evidence),
        ),
        chunks,
    )


@pytest.mark.asyncio
async def test_in_memory_vector_inspection_distinguishes_absent_exact_and_partial() -> None:
    filing_manifest, chunks = _inspection_fixture()
    vector_store = InMemoryVectorStore()
    deadline = RequestDeadline.after(1)

    absent = await vector_store.inspect_generation(filing_manifest, deadline=deadline)
    await vector_store.stage_generation(filing_manifest, chunks, deadline=deadline)
    exact = await vector_store.inspect_generation(filing_manifest, deadline=deadline)
    vector_store._generations = ((filing_manifest, chunks[:1]),)
    partial = await vector_store.inspect_generation(filing_manifest, deadline=deadline)

    assert absent.state is GenerationInspectionState.ABSENT
    assert exact.state is GenerationInspectionState.EXACT
    assert partial.state is GenerationInspectionState.PARTIAL
    assert partial.observed_point_count == 1


@pytest.mark.asyncio
async def test_in_memory_vector_inspection_fails_closed_for_inconsistent_state() -> None:
    filing_manifest, chunks = _inspection_fixture()
    vector_store = InMemoryVectorStore()
    vector_store._generations = ((filing_manifest, (chunks[1], chunks[0])),)

    inspection = await vector_store.inspect_generation(
        filing_manifest,
        deadline=RequestDeadline.after(1),
    )

    assert inspection.state is GenerationInspectionState.INCONSISTENT


@pytest.mark.asyncio
async def test_in_memory_control_reads_only_the_exact_generation_stage() -> None:
    filing_manifest, _ = _inspection_fixture()
    control = InMemoryResearchControlPlane()
    deadline = RequestDeadline.after(1)
    lease = await control.acquire_generation_lease(
        manifest=filing_manifest,
        owner_digest="a" * 64,
        ttl_seconds=60,
        deadline=deadline,
    )
    assert lease is not None
    staged = await control.stage_generation(
        lease=lease,
        manifest=filing_manifest,
        deadline=deadline,
    )

    observed = await control.get_generation_stage(
        manifest=filing_manifest,
        deadline=deadline,
    )

    assert observed == staged
