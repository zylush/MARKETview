from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from datetime import date

import httpx
import pytest
from pydantic import SecretStr

from app.providers.upstash_vector import (
    UpstashVectorConfigurationError,
    UpstashVectorStore,
    UpstashVectorTimeoutError,
    UpstashVectorUnavailableError,
)
from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    EmbeddedChunk,
    EmbeddingDescriptor,
    EmbeddingVector,
    EvidenceChunk,
    FilingDocument,
    GenerationManifest,
    GenerationVerificationReason,
)
from app.research.ports import GenerationInspectionState


def corpus() -> CorpusDescriptor:
    return CorpusDescriptor(
        corpus_version="sec-v1",
        chunker_version="chunks-v1",
        embedding=EmbeddingDescriptor(
            provider="openai",
            model="text-embedding-3-small",
            version="v1",
            dimensions=2,
        ),
    )


def embedded_chunks(
    texts: tuple[str, ...] = (
        "Supply risk disclosure.",
        "Cloud demand disclosure.",
    ),
) -> tuple[EmbeddedChunk, ...]:
    configured = corpus()
    document = FilingDocument(
        symbol="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000001",
        filing_type="10-K",
        title="Apple 2025 Form 10-K",
        filed_date=date(2025, 10, 31),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/aapl-20250927.htm"
        ),
        text=" ".join(texts),
    )
    return tuple(
        EmbeddedChunk(
            evidence=EvidenceChunk.from_document(
                document,
                corpus=configured,
                ordinal=ordinal,
                text=text,
            ),
            embedding=EmbeddingVector(
                descriptor=configured.embedding,
                values=values,
            ),
        )
        for ordinal, (text, values) in enumerate(
            (
                (text, (1.0, 0.0) if ordinal % 2 == 0 else (0.0, 1.0))
                for ordinal, text in enumerate(texts)
            )
        )
    )


def manifest(chunks: tuple[EmbeddedChunk, ...] | None = None) -> GenerationManifest:
    selected = chunks or embedded_chunks()
    first = selected[0].evidence
    return GenerationManifest(
        corpus=first.corpus,
        symbol=first.symbol,
        accession_number=first.accession_number,
        generation_id=first.generation_id,
        content_hash=first.content_hash,
        chunk_ids=tuple(chunk.evidence.chunk_id for chunk in selected),
    )


def second_manifest() -> GenerationManifest:
    configured = corpus()
    document = FilingDocument(
        symbol="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000002",
        filing_type="10-Q",
        title="Apple 2025 Form 10-Q",
        filed_date=date(2025, 8, 1),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000002/aapl-20250628.htm"
        ),
        text="Quarterly risk disclosure.",
    )
    evidence = EvidenceChunk.from_document(
        document,
        corpus=configured,
        ordinal=0,
        text=document.text,
    )
    return GenerationManifest(
        corpus=configured,
        symbol=evidence.symbol,
        accession_number=evidence.accession_number,
        generation_id=evidence.generation_id,
        content_hash=evidence.content_hash,
        chunk_ids=(evidence.chunk_id,),
    )


def metadata(chunk: EmbeddedChunk) -> dict[str, object]:
    evidence = chunk.evidence
    descriptor = evidence.corpus.embedding
    values: dict[str, object] = {
        "symbol": evidence.symbol,
        "cik": evidence.cik,
        "accession_number": evidence.accession_number,
        "filing_type": evidence.filing_type,
        "title": evidence.title,
        "filed_date": evidence.filed_date.isoformat(),
        "source_url": evidence.source_url,
        "chunk_id": evidence.chunk_id,
        "content_hash": evidence.content_hash,
        "ordinal": evidence.ordinal,
        "corpus_id": evidence.corpus.canonical_key,
        "corpus_version": evidence.corpus.corpus_version,
        "chunker_version": evidence.corpus.chunker_version,
        "embedding_key": descriptor.canonical_key,
        "embedding_provider": descriptor.provider,
        "embedding_model": descriptor.model,
        "embedding_version": descriptor.version,
        "embedding_dimensions": descriptor.dimensions,
        "generation_id": evidence.generation_id,
        "evidence_digest": evidence.evidence_digest,
    }
    values["point_digest"] = UpstashVectorStore.point_digest(
        evidence.chunk_id,
        evidence.text,
        values,
    )
    return values


def store(
    handler: httpx.MockTransport,
    *,
    token: str = "vector-secret-token",
) -> tuple[UpstashVectorStore, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=handler, follow_redirects=True)
    return (
        UpstashVectorStore(
            "https://research-vector.upstash.io",
            SecretStr(token),
            namespace="sec-filings-v1",
            client=client,
        ),
        client,
    )


@pytest.mark.parametrize(
    ("url", "namespace"),
    [
        ("http://research-vector.upstash.io", "sec-filings-v1"),
        ("https://research-vector.upstash.io/path", "sec-filings-v1"),
        ("https://research-vector.upstash.io?token=bad", "sec-filings-v1"),
        ("https://research-vector.upstash.io.evil.example", "sec-filings-v1"),
        ("https://user@research-vector.upstash.io", "sec-filings-v1"),
        ("https://research-vector.upstash.io:443", "sec-filings-v1"),
        ("https://research-vector.upstash.io", "other-namespace"),
    ],
)
def test_configuration_rejects_noncanonical_roots_and_unapproved_namespaces(
    url: str,
    namespace: str,
) -> None:
    with pytest.raises(UpstashVectorConfigurationError):
        UpstashVectorStore(url, SecretStr("secret"), namespace=namespace)


def test_configuration_requires_secretstr_without_exposing_value() -> None:
    with pytest.raises(UpstashVectorConfigurationError) as captured:
        UpstashVectorStore(
            "https://research-vector.upstash.io",
            "plain-secret",  # type: ignore[arg-type]
            namespace="sec-filings-v1",
        )

    assert "plain-secret" not in str(captured.value)


@pytest.mark.asyncio
async def test_stage_generation_upserts_immutable_records_with_server_only_auth() -> None:
    calls: list[httpx.Request] = []
    chunks = embedded_chunks()
    expected_snapshot = copy.deepcopy(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"result": "Success"})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        await adapter.stage_generation(
            manifest(chunks),
            chunks,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await adapter.aclose()
        assert not client.is_closed
        await client.aclose()

    assert chunks == expected_snapshot
    assert len(calls) == 1
    request = calls[0]
    assert request.method == "POST"
    assert request.url == httpx.URL("https://research-vector.upstash.io/upsert/sec-filings-v1")
    assert request.headers["Authorization"] == "Bearer vector-secret-token"
    assert "Upstash-Namespace" not in request.headers
    payload = json.loads(request.content)
    assert [point["id"] for point in payload] == list(manifest(chunks).chunk_ids)
    assert payload[0]["vector"] == [1.0, 0.0]
    assert payload[0]["metadata"] == metadata(chunks[0])
    assert payload[0]["data"] == "Supply risk disclosure."


@pytest.mark.asyncio
async def test_stage_rejects_manifest_chunk_mismatch_before_network() -> None:
    calls = 0
    chunks = embedded_chunks()
    mismatched = GenerationManifest(
        corpus=manifest(chunks).corpus,
        symbol="AAPL",
        accession_number=manifest(chunks).accession_number,
        generation_id=manifest(chunks).generation_id,
        content_hash=manifest(chunks).content_hash,
        chunk_ids=(manifest(chunks).chunk_ids[0],),
    )

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"result": "Success"})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError, match="manifest"):
            await adapter.stage_generation(
                mismatched,
                chunks,
                deadline=RequestDeadline.after(1),
            )
    finally:
        await client.aclose()

    assert calls == 0


@pytest.mark.asyncio
async def test_verify_fetches_exact_ids_without_vectors_and_checks_all_metadata() -> None:
    chunks = embedded_chunks()
    filing_manifest = manifest(chunks)
    request_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": chunk.evidence.chunk_id,
                        "metadata": metadata(chunk),
                        "data": chunk.evidence.text,
                    }
                    for chunk in chunks
                ]
            },
        )

    adapter, client = store(httpx.MockTransport(handler))
    try:
        verified = await adapter.verify_generation(
            filing_manifest,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert verified.reason is GenerationVerificationReason.VERIFIED
    assert verified.expected_point_count == 2
    assert verified.observed_point_count == 2
    assert verified.null_point_count == 0
    assert verified.attempt_count == 1
    assert verified.proves(filing_manifest)
    assert request_payloads == [
        {
            "ids": list(filing_manifest.chunk_ids),
            "includeVectors": False,
            "includeMetadata": True,
            "includeData": True,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returned", "expected_state"),
    [
        ((), GenerationInspectionState.ABSENT),
        ((0,), GenerationInspectionState.PARTIAL),
        ((0, 1), GenerationInspectionState.EXACT),
    ],
)
async def test_inspect_generation_returns_only_safe_aggregate_state_without_vectors_or_data(
    returned: tuple[int, ...],
    expected_state: GenerationInspectionState,
) -> None:
    chunks = embedded_chunks()
    filing_manifest = manifest(chunks)
    request_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": chunks[index].evidence.chunk_id,
                        "metadata": metadata(chunks[index]),
                    }
                    for index in returned
                ]
            },
        )

    adapter, client = store(httpx.MockTransport(handler))
    try:
        inspection = await adapter.inspect_generation(
            filing_manifest,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert inspection.state is expected_state
    assert inspection.expected_point_count == 2
    assert inspection.observed_point_count == len(returned)
    assert not hasattr(inspection, "point_ids")
    assert request_payloads == [
        {
            "ids": list(filing_manifest.chunk_ids),
            "includeVectors": False,
            "includeMetadata": True,
            "includeData": False,
        }
    ]


@pytest.mark.asyncio
async def test_inspect_generation_treats_missing_fetch_entries_as_absent() -> None:
    chunks = embedded_chunks()
    filing_manifest = manifest(chunks)

    adapter, client = store(
        httpx.MockTransport(lambda _: httpx.Response(200, json={"result": [None, None]}))
    )
    try:
        inspection = await adapter.inspect_generation(
            filing_manifest,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert inspection.state is GenerationInspectionState.ABSENT
    assert inspection.expected_point_count == 2
    assert inspection.observed_point_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["duplicate", "unexpected", "metadata", "malformed"])
async def test_inspect_generation_fails_closed_on_inconsistent_provider_records(
    fault: str,
) -> None:
    chunks = embedded_chunks()
    filing_manifest = manifest(chunks)
    base_records: list[dict[str, object]] = [
        {"id": chunk.evidence.chunk_id, "metadata": metadata(chunk)} for chunk in chunks
    ]
    records: object = base_records
    if fault == "duplicate":
        records = [base_records[0], base_records[0]]
    elif fault == "unexpected":
        records = [
            base_records[0],
            {"id": "chunk-" + "f" * 64, "metadata": metadata(chunks[1])},
        ]
    elif fault == "metadata":
        records = [
            base_records[0],
            {
                "id": chunks[1].evidence.chunk_id,
                "metadata": {**metadata(chunks[1]), "generation_id": "gen-" + "f" * 64},
            },
        ]
    else:
        records = "private-filing-content-must-not-escape"

    adapter, client = store(
        httpx.MockTransport(lambda _: httpx.Response(200, json={"result": records}))
    )
    try:
        inspection = await adapter.inspect_generation(
            filing_manifest,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert inspection.state is GenerationInspectionState.INCONSISTENT
    assert "private-filing-content" not in repr(inspection)


def manifest_with_point_count(point_count: int) -> GenerationManifest:
    return GenerationManifest(
        corpus=corpus(),
        symbol="AAPL",
        accession_number="0000320193-25-000001",
        generation_id="gen-" + "a" * 64,
        content_hash="b" * 64,
        chunk_ids=tuple(
            "chunk-" + hashlib.sha256(str(index).encode("ascii")).hexdigest()
            for index in range(point_count)
        ),
    )


@pytest.mark.asyncio
async def test_inspect_generation_bounds_expected_ids_and_preserves_absolute_deadline() -> None:
    filing_manifest = manifest_with_point_count(128)
    calls: list[dict[str, object]] = []
    now = [100.0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        now[0] = 102.0
        return httpx.Response(200, json={"result": []})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        with pytest.raises(UpstashVectorTimeoutError, match="timed out"):
            await adapter.inspect_generation(
                filing_manifest,
                deadline=RequestDeadline.after(1, clock=lambda: now[0]),
            )
    finally:
        await client.aclose()

    assert len(calls) == 1
    assert calls[0] == {
        "ids": list(filing_manifest.chunk_ids[:100]),
        "includeVectors": False,
        "includeMetadata": True,
        "includeData": False,
    }


@pytest.mark.asyncio
async def test_inspect_generation_batches_all_expected_ids_without_unbounded_queries() -> None:
    filing_manifest = manifest_with_point_count(128)
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"result": []})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        inspection = await adapter.inspect_generation(
            filing_manifest,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert inspection.state is GenerationInspectionState.ABSENT
    assert [len(call["ids"]) for call in calls] == [100, 28]  # type: ignore[arg-type]
    assert all(call["includeVectors"] is False for call in calls)
    assert all(call["includeData"] is False for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["data", "title", "source_url", "cik", "filed_date"])
async def test_verify_rejects_same_id_point_tampering_without_traceback_disclosure(
    tamper: str,
) -> None:
    chunks = embedded_chunks()
    filing_manifest = manifest(chunks)
    records = [
        {
            "id": chunk.evidence.chunk_id,
            "metadata": metadata(chunk),
            "data": chunk.evidence.text,
        }
        for chunk in chunks
    ]
    secret_marker = "tampered-private-filing-text"
    if tamper == "data":
        records[0]["data"] = secret_marker
    else:
        changed = dict(records[0]["metadata"])
        changed[tamper] = secret_marker
        records[0]["metadata"] = changed

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": records})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        outcome = await adapter.verify_generation(
            filing_manifest,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    expected_reason = (
        GenerationVerificationReason.DATA_MISMATCH
        if tamper == "data"
        else GenerationVerificationReason.METADATA_MISMATCH
    )
    assert outcome.reason is expected_reason
    assert outcome.expected_point_count == 2
    assert outcome.observed_point_count == 2
    assert outcome.null_point_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["data", "title", "source_url", "cik", "filed_date"])
async def test_search_rejects_same_id_point_tampering_without_payload_or_traceback(
    tamper: str,
) -> None:
    chunk = embedded_chunks()[0]
    active = manifest(embedded_chunks())
    record: dict[str, object] = {
        "id": chunk.evidence.chunk_id,
        "score": 0.99,
        "metadata": metadata(chunk),
        "data": chunk.evidence.text,
    }
    secret_marker = "tampered-private-filing-text"
    if tamper == "data":
        record["data"] = secret_marker
    else:
        changed = dict(record["metadata"])  # type: ignore[arg-type]
        changed[tamper] = secret_marker
        record["metadata"] = changed

    adapter, client = store(
        httpx.MockTransport(lambda _: httpx.Response(200, json={"result": [record]}))
    )
    try:
        with pytest.raises(UpstashVectorUnavailableError) as captured:
            await adapter.search(
                corpus=active.corpus,
                symbol="AAPL",
                vector=EmbeddingVector(
                    descriptor=active.corpus.embedding,
                    values=(1.0, 0.0),
                ),
                active_generations=(active,),
                limit=1,
                deadline=RequestDeadline.after(1),
            )
    finally:
        await client.aclose()

    assert secret_marker not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_verify_returns_false_for_missing_or_incompatible_points() -> None:
    chunks = embedded_chunks()
    bad_metadata = {**metadata(chunks[0]), "generation_id": "gen-" + "0" * 64}

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": chunks[0].evidence.chunk_id,
                        "metadata": bad_metadata,
                    }
                ]
            },
        )

    adapter, client = store(httpx.MockTransport(handler))
    try:
        outcome = await adapter.verify_generation(
            manifest(chunks),
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert outcome.reason is GenerationVerificationReason.RESPONSE_SHAPE


@pytest.mark.asyncio
async def test_verify_polls_only_missing_points_then_succeeds_without_reupsert() -> None:
    chunks = embedded_chunks()
    filing_manifest = manifest(chunks)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        visible = len(calls) > 1
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": chunks[0].evidence.chunk_id,
                        "metadata": metadata(chunks[0]),
                        "data": chunks[0].evidence.text,
                    },
                    (
                        {
                            "id": chunks[1].evidence.chunk_id,
                            "metadata": metadata(chunks[1]),
                            "data": chunks[1].evidence.text,
                        }
                        if visible
                        else None
                    ),
                ]
            },
        )

    adapter, client = store(httpx.MockTransport(handler))
    try:
        outcome = await adapter.verify_generation(
            filing_manifest,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert outcome.reason is GenerationVerificationReason.VERIFIED
    assert outcome.proves(filing_manifest)
    assert calls == ["/fetch/sec-filings-v1", "/fetch/sec-filings-v1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_index", [0, 1])
async def test_verify_polls_short_leading_or_internal_visibility_as_partial_then_succeeds(
    missing_index: int,
) -> None:
    chunks = embedded_chunks(("First disclosure.", "Second disclosure.", "Third disclosure."))
    filing_manifest = manifest(chunks)
    calls = 0

    def record(index: int) -> dict[str, object]:
        return {
            "id": chunks[index].evidence.chunk_id,
            "metadata": metadata(chunks[index]),
            "data": chunks[index].evidence.text,
        }

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        records = (
            [record(index) for index in range(3) if index != missing_index]
            if calls == 1
            else [record(0), record(1), record(2)]
        )
        return httpx.Response(200, json={"result": records})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        outcome = await adapter.verify_generation(
            filing_manifest,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert outcome.reason is GenerationVerificationReason.VERIFIED
    assert outcome.attempt_count == 2
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    [
        ("shape", GenerationVerificationReason.RESPONSE_SHAPE),
        ("order", GenerationVerificationReason.ORDERING_MISMATCH),
        ("duplicate", GenerationVerificationReason.ORDERING_MISMATCH),
        ("unrequested", GenerationVerificationReason.ORDERING_MISMATCH),
        ("null_position", GenerationVerificationReason.ORDERING_MISMATCH),
        ("metadata", GenerationVerificationReason.METADATA_MISMATCH),
        ("data", GenerationVerificationReason.DATA_MISMATCH),
        ("integrity", GenerationVerificationReason.INTEGRITY_MISMATCH),
    ],
)
async def test_verify_does_not_poll_nonretryable_failures(
    fault: str,
    expected_reason: GenerationVerificationReason,
) -> None:
    chunks = embedded_chunks()
    records: object = [
        {
            "id": chunk.evidence.chunk_id,
            "metadata": metadata(chunk),
            "data": chunk.evidence.text,
        }
        for chunk in chunks
    ]
    if fault == "shape":
        records = ["private-provider-payload"]
    elif fault == "order":
        records = list(reversed(records))  # type: ignore[arg-type]
    elif fault == "duplicate":
        records = [records[0], records[0]]  # type: ignore[index]
    elif fault == "unrequested":
        records[0]["id"] = "chunk-" + "f" * 64  # type: ignore[index]
    elif fault == "null_position":
        records = [None, records[0]]  # type: ignore[index]
    elif fault == "metadata":
        changed = dict(records[0]["metadata"])  # type: ignore[index]
        changed["symbol"] = "MSFT"
        records[0]["metadata"] = changed  # type: ignore[index]
    elif fault == "data":
        records[0]["data"] = "private-tampered-filing-text"  # type: ignore[index]
    else:
        changed = dict(records[0]["metadata"])  # type: ignore[index]
        changed["point_digest"] = "f" * 64
        records[0]["metadata"] = changed  # type: ignore[index]
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"result": records})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        outcome = await adapter.verify_generation(
            manifest(chunks),
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert outcome.reason is expected_reason
    assert calls == 1
    assert "private" not in repr(outcome)


@pytest.mark.asyncio
async def test_verify_missing_counts_are_sanitized_after_bounded_polling() -> None:
    chunks = embedded_chunks()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": chunks[0].evidence.chunk_id,
                        "metadata": metadata(chunks[0]),
                        "data": chunks[0].evidence.text,
                    },
                    None,
                ]
            },
        )

    adapter, client = store(httpx.MockTransport(handler))
    try:
        outcome = await adapter.verify_generation(
            manifest(chunks),
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert outcome.reason is GenerationVerificationReason.PARTIAL_VISIBILITY
    assert outcome.expected_point_count == 2
    assert outcome.observed_point_count == 1
    assert outcome.null_point_count == 1
    assert outcome.attempt_count == 4
    assert calls == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("visible_count", "expected_reason"),
    [
        (0, GenerationVerificationReason.MISSING_POINTS),
        (1, GenerationVerificationReason.PARTIAL_VISIBILITY),
    ],
)
async def test_verify_short_result_is_pollable_without_inventing_nulls(
    visible_count: int,
    expected_reason: GenerationVerificationReason,
) -> None:
    chunks = embedded_chunks()
    records = [
        {
            "id": chunk.evidence.chunk_id,
            "metadata": metadata(chunk),
            "data": chunk.evidence.text,
        }
        for chunk in chunks[:visible_count]
    ]
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"result": records})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        outcome = await adapter.verify_generation(
            manifest(chunks),
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert outcome.reason is expected_reason
    assert outcome.observed_point_count == visible_count
    assert outcome.null_point_count == 0
    assert outcome.attempt_count == 4
    assert calls == 4


@pytest.mark.asyncio
async def test_verify_polling_never_extends_the_absolute_deadline() -> None:
    chunks = embedded_chunks()
    now = [10.0]
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        now[0] = 12.0
        return httpx.Response(200, json={"result": [None, None]})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        with pytest.raises(UpstashVectorTimeoutError, match="timed out") as caught:
            await adapter.verify_generation(
                manifest(chunks),
                deadline=RequestDeadline.after(1, clock=lambda: now[0]),
            )
    finally:
        await client.aclose()

    assert calls == 1
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_verify_128_points_recomputes_global_counts_for_each_poll_attempt() -> None:
    filing_manifest = manifest_with_point_count(128)
    call_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ids = json.loads(request.content)["ids"]
        call_sizes.append(len(ids))
        return httpx.Response(200, json={"result": [None for _ in ids]})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        outcome = await adapter.verify_generation(
            filing_manifest,
            deadline=RequestDeadline.after(2),
        )
    finally:
        await client.aclose()

    assert outcome.reason is GenerationVerificationReason.MISSING_POINTS
    assert outcome.expected_point_count == 128
    assert outcome.observed_point_count == 0
    assert outcome.null_point_count == 128
    assert outcome.attempt_count == 4
    assert call_sizes == [100, 28, 100, 28, 100, 28, 100, 28]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["abort", "delete"])
async def test_generation_cleanup_uses_exact_server_generated_scope(operation: str) -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request.method, json.loads(request.content)))
        return httpx.Response(200, json={"result": {"deleted": 2}})

    filing_manifest = manifest()
    adapter, client = store(httpx.MockTransport(handler))
    try:
        method = adapter.abort_generation if operation == "abort" else adapter.delete_generation
        await method(filing_manifest, deadline=RequestDeadline.after(1))
    finally:
        await client.aclose()

    assert captured[0][0] == "DELETE"
    filter_text = str(captured[0][1]["filter"])
    assert "symbol = 'AAPL'" in filter_text
    assert f"corpus_id = '{filing_manifest.corpus.canonical_key}'" in filter_text
    assert f"accession_number = '{filing_manifest.accession_number}'" in filter_text
    assert f"generation_id = '{filing_manifest.generation_id}'" in filter_text


@pytest.mark.asyncio
async def test_search_filters_provider_and_defensively_rejects_stale_cross_scope_hits() -> None:
    chunks = embedded_chunks()
    active = manifest(chunks)
    stale_metadata = {
        **metadata(chunks[0]),
        "generation_id": "gen-" + "0" * 64,
    }
    cross_symbol = {**metadata(chunks[1]), "symbol": "MSFT"}
    request_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": chunks[0].evidence.chunk_id,
                        "score": 0.99,
                        "metadata": metadata(chunks[0]),
                        "data": chunks[0].evidence.text,
                    },
                    {
                        "id": chunks[0].evidence.chunk_id,
                        "score": 0.98,
                        "metadata": stale_metadata,
                        "data": chunks[0].evidence.text,
                    },
                    {
                        "id": chunks[1].evidence.chunk_id,
                        "score": 0.97,
                        "metadata": cross_symbol,
                        "data": chunks[1].evidence.text,
                    },
                ]
            },
        )

    adapter, client = store(httpx.MockTransport(handler))
    try:
        hits = await adapter.search(
            corpus=active.corpus,
            symbol="AAPL",
            vector=EmbeddingVector(
                descriptor=active.corpus.embedding,
                values=(1.0, 0.0),
            ),
            active_generations=(active,),
            limit=3,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert tuple(hit.evidence.chunk_id for hit in hits) == (chunks[0].evidence.chunk_id,)
    assert hits[0].active_generation_id == active.generation_id
    assert request_body["vector"] == [1.0, 0.0]
    assert request_body["topK"] == 3
    assert request_body["includeVectors"] is False
    assert request_body["includeMetadata"] is True
    assert request_body["includeData"] is True
    assert "symbol = 'AAPL'" in str(request_body["filter"])
    assert f"corpus_id = '{active.corpus.canonical_key}'" in str(request_body["filter"])
    assert f"embedding_key = '{active.corpus.embedding.canonical_key}'" in str(
        request_body["filter"]
    )
    assert f"generation_id IN ('{active.generation_id}')" in str(request_body["filter"])


@pytest.mark.asyncio
async def test_search_filter_exactly_allowlists_multiple_active_generations() -> None:
    first = manifest()
    second = second_manifest()
    captured_filter = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_filter
        captured_filter = str(json.loads(request.content)["filter"])
        return httpx.Response(200, json={"result": []})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        hits = await adapter.search(
            corpus=first.corpus,
            symbol="AAPL",
            vector=EmbeddingVector(
                descriptor=first.corpus.embedding,
                values=(1.0, 0.0),
            ),
            active_generations=(first, second),
            limit=2,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await client.aclose()

    assert hits == ()
    assert captured_filter == " AND ".join(
        (
            "symbol = 'AAPL'",
            f"corpus_id = '{first.corpus.canonical_key}'",
            f"embedding_key = '{first.corpus.embedding.canonical_key}'",
            (f"generation_id IN ('{first.generation_id}', '{second.generation_id}')"),
        )
    )


@pytest.mark.parametrize("unsafe", ["gen-' OR 1=1", "gen-\\escape"])
def test_search_filter_literal_rejects_quote_and_escape_injection(unsafe: str) -> None:
    with pytest.raises(ValueError, match="filter value"):
        UpstashVectorStore._quoted(unsafe)


@pytest.mark.asyncio
async def test_search_rejects_malformed_provider_record_without_leaking_payload() -> None:
    secret_marker = "provider-raw-secret-marker"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": [{"raw": secret_marker}]})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        with pytest.raises(UpstashVectorUnavailableError) as captured:
            await adapter.search(
                corpus=corpus(),
                symbol="AAPL",
                vector=EmbeddingVector(
                    descriptor=corpus().embedding,
                    values=(1.0, 0.0),
                ),
                active_generations=(manifest(),),
                limit=1,
                deadline=RequestDeadline.after(1),
            )
    finally:
        await client.aclose()

    assert secret_marker not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"Location": "https://evil.example/steal"}),
        httpx.Response(401, json={"error": "vector-secret-token"}),
        httpx.Response(200, content=b"{" + b"x" * 1_048_576),
        httpx.Response(200, content=b"not-json"),
    ],
)
async def test_provider_failures_are_one_attempt_and_sanitized(response: httpx.Response) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response

    adapter, client = store(httpx.MockTransport(handler))
    try:
        with pytest.raises(UpstashVectorUnavailableError) as captured:
            await adapter.stage_generation(
                manifest(),
                embedded_chunks(),
                deadline=RequestDeadline.after(1),
            )
    finally:
        await client.aclose()

    assert calls == 1
    message = str(captured.value)
    assert "vector-secret-token" not in message
    assert "evil.example" not in message
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_expired_deadline_makes_no_request_and_raises_sanitized_timeout() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"result": "Success"})

    adapter, client = store(httpx.MockTransport(handler))
    expired = RequestDeadline(expires_at=0.0, _clock=lambda: 1.0)
    try:
        with pytest.raises(UpstashVectorTimeoutError):
            await adapter.stage_generation(manifest(), embedded_chunks(), deadline=expired)
    finally:
        await client.aclose()

    assert calls == 0


@pytest.mark.asyncio
async def test_adapter_closes_only_internally_owned_client() -> None:
    adapter = UpstashVectorStore(
        "https://research-vector.upstash.io",
        SecretStr("secret"),
        namespace="sec-filings-v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"result": "Success"})),
    )

    await adapter.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await adapter.stage_generation(
            manifest(),
            embedded_chunks(),
            deadline=RequestDeadline.after(1),
        )


def test_point_set_hash_is_deterministic_and_order_sensitive() -> None:
    chunks = embedded_chunks()
    expected = hashlib.sha256("\n".join(manifest(chunks).chunk_ids).encode("ascii")).hexdigest()

    assert UpstashVectorStore.point_set_hash(manifest(chunks).chunk_ids) == expected
    assert (
        UpstashVectorStore.point_set_hash(tuple(reversed(manifest(chunks).chunk_ids))) != expected
    )


@pytest.mark.asyncio
async def test_deadline_cancels_slow_mock_transport_without_retry() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"result": "Success"})

    adapter, client = store(httpx.MockTransport(handler))
    try:
        with pytest.raises(UpstashVectorTimeoutError):
            await adapter.stage_generation(
                manifest(),
                embedded_chunks(),
                deadline=RequestDeadline.after(0.001),
            )
    finally:
        await client.aclose()

    assert calls == 1
