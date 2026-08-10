from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import httpx
import pytest
from pydantic import SecretStr

from app.providers.upstash_vector import UpstashVectorStore
from app.research import diagnostic_runtime
from app.research.control import Reservation, ReservationState
from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    EmbeddingDescriptor,
    EmbeddingVector,
    EvidenceChunk,
    FilingDocument,
    GenerationManifest,
    SearchHit,
)
from app.research.ports import GenerationInspection, GenerationInspectionState
from app.research.retrieval import classify_safe_hits
from app.retrieval_diagnostic import (
    DiagnosticPreflight,
    RetrievalDiagnosticRuntime,
    build_retrieval_diagnostic_runtime,
)

EMBEDDING = EmbeddingDescriptor(
    provider="openai",
    model="text-embedding-3-small",
    version="2024-01",
    dimensions=2,
)
CORPUS = CorpusDescriptor(
    corpus_version="v1",
    chunker_version="tokens-800-100-v1",
    embedding=EMBEDDING,
)
VECTOR = EmbeddingVector(descriptor=EMBEDDING, values=(1.0, 0.0))
QUESTION = "What material risks did Apple disclose in its latest Form 10-K?"


def document(
    *,
    text: str = "Apple reports supply constraints and product demand risks.",
    filing_type: str = "10-K",
):
    return FilingDocument(
        symbol="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000001",
        filing_type=filing_type,
        title=f"AAPL 2025 Form {filing_type}",
        filed_date=date(2025, 10, 31),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/aapl-20250927.htm"
        ),
        text=text,
    )


def evidence(
    *,
    text: str = "Apple reports supply constraints.",
    source: FilingDocument | None = None,
    ordinal: int = 0,
) -> EvidenceChunk:
    return EvidenceChunk.from_document(
        source or document(),
        corpus=CORPUS,
        ordinal=ordinal,
        text=text,
    )


def manifest(*items: EvidenceChunk) -> GenerationManifest:
    first = items[0]
    return GenerationManifest(
        corpus=CORPUS,
        symbol="AAPL",
        accession_number=first.accession_number,
        generation_id=first.generation_id,
        content_hash=first.content_hash,
        chunk_ids=tuple(item.chunk_id for item in items),
    )


class FakeEmbedder:
    descriptor = EMBEDDING

    def __init__(self) -> None:
        self.calls: list[tuple[str, RequestDeadline]] = []

    async def embed_query(
        self,
        text: str,
        *,
        deadline: RequestDeadline,
    ) -> EmbeddingVector:
        self.calls.append((text, deadline))
        return VECTOR


class FakeStore:
    def __init__(
        self,
        *,
        inspection: GenerationInspection | None = None,
        hits: tuple[SearchHit, ...] = (),
    ) -> None:
        self.inspection = inspection or GenerationInspection(
            state=GenerationInspectionState.EXACT,
            expected_point_count=1,
            observed_point_count=1,
        )
        self.hits = hits
        self.events: list[str] = []
        self.inspections: list[tuple[GenerationManifest, RequestDeadline]] = []
        self.searches: list[dict[str, object]] = []

    async def inspect_generation(
        self,
        selected: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> GenerationInspection:
        self.events.append("inspection")
        self.inspections.append((selected, deadline))
        return self.inspection

    async def search(self, **kwargs: object) -> tuple[SearchHit, ...]:
        self.events.append("search")
        self.searches.append(kwargs)
        return self.hits


class FakeControl:
    def __init__(
        self,
        *,
        active: tuple[GenerationManifest, ...],
        pending: tuple[object, ...] = (),
        authorize_result: bool = True,
        authorize_error: BaseException | None = None,
    ) -> None:
        self.active = active
        self.pending = pending
        self.authorize_result = authorize_result
        self.authorize_error = authorize_error
        self.events: list[str] = []
        self.authorization: dict[str, object] | None = None
        self.commits: list[dict[str, object]] = []
        self.releases: list[dict[str, object]] = []

    async def list_active_generations(self, **_kwargs: object):
        self.events.append("active")
        return self.active

    async def list_pending_cleanups(self, **_kwargs: object):
        self.events.append("pending")
        return self.pending

    async def authorize_reservation(self, **kwargs: object) -> bool:
        self.events.append("authorize")
        self.authorization = kwargs
        if self.authorize_error is not None:
            raise self.authorize_error
        return self.authorize_result

    async def commit_reservation(self, **kwargs: object) -> bool:
        self.events.append("commit")
        self.commits.append(kwargs)
        return True

    async def release_reservation(self, **kwargs: object) -> bool:
        self.events.append("release")
        self.releases.append(kwargs)
        return True


class FakeClient:
    async def aclose(self) -> None:
        return None


def runtime(
    *,
    active: tuple[GenerationManifest, ...],
    pending: tuple[object, ...] = (),
    store: FakeStore | None = None,
    embedder: FakeEmbedder | None = None,
    control: FakeControl | None = None,
    maximum_results: int = 2,
    overfetch_factor: int = 3,
    minimum_score: float = 0.70,
    daily_limit: int = 17,
) -> tuple[RetrievalDiagnosticRuntime, FakeControl, FakeStore, FakeEmbedder]:
    selected_control = control or FakeControl(active=active, pending=pending)
    selected_store = store or FakeStore()
    selected_embedder = embedder or FakeEmbedder()
    return (
        RetrievalDiagnosticRuntime(
            corpus=CORPUS,
            embedder=selected_embedder,  # type: ignore[arg-type]
            store=selected_store,  # type: ignore[arg-type]
            control=selected_control,  # type: ignore[arg-type]
            client=FakeClient(),  # type: ignore[arg-type]
            maximum_results=maximum_results,
            overfetch_factor=overfetch_factor,
            minimum_score=minimum_score,
            daily_limit=daily_limit,
        ),
        selected_control,
        selected_store,
        selected_embedder,
    )


@pytest.mark.asyncio
async def test_preflight_calls_active_pending_inspection_and_returns_exact_aggregates() -> None:
    item = evidence()
    active = manifest(item)
    store = FakeStore(
        inspection=GenerationInspection(
            state=GenerationInspectionState.EXACT,
            expected_point_count=3,
            observed_point_count=3,
        )
    )
    subject, control, _, _ = runtime(active=(active,), store=store)
    deadline = RequestDeadline.after(1)

    result = await subject.preflight(symbol="AAPL", filing_type="10-K", deadline=deadline)

    assert control.events == ["active", "pending"]
    assert store.events == ["inspection"]
    assert store.inspections == [(active, deadline)]
    assert result == DiagnosticPreflight(
        active_generation_count=1,
        pending_cleanup_count=0,
        inspection_state="exact",
        expected_point_count=3,
        observed_point_count=3,
        private_manifest=active,
    )
    assert result.is_exact is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("active_count", "pending"),
    [(0, ()), (2, ()), (1, (object(),))],
)
async def test_invalid_active_or_pending_state_short_circuits_inspection(
    active_count: int,
    pending: tuple[object, ...],
) -> None:
    item = evidence()
    active = manifest(item)
    subject, control, store, _ = runtime(active=(active,) * active_count, pending=pending)

    result = await subject.preflight(
        symbol="AAPL",
        filing_type="10-K",
        deadline=RequestDeadline.after(1),
    )

    assert control.events == ["active", "pending"]
    assert store.events == []
    assert result.active_generation_count == active_count
    assert result.pending_cleanup_count == len(pending)
    assert result.inspection_state is None
    assert result.expected_point_count is None
    assert result.observed_point_count is None
    assert result.is_exact is False


@pytest.mark.asyncio
async def test_revalidate_detects_exact_manifest_drift() -> None:
    original_item = evidence()
    original = manifest(original_item)
    subject, control, _, _ = runtime(active=(original,))
    snapshot = await subject.preflight(
        symbol="AAPL",
        filing_type="10-K",
        deadline=RequestDeadline.after(1),
    )
    replacement_item = evidence(
        source=document(text="A changed filing body creates a different generation."),
        text="Changed disclosure.",
    )
    control.active = (manifest(replacement_item),)

    unchanged = await subject.revalidate(
        preflight=snapshot,
        deadline=RequestDeadline.after(1),
    )

    assert unchanged is False
    assert control.events == ["active", "pending", "active", "pending"]


@pytest.mark.asyncio
async def test_authorize_uses_one_unit_daily_limit_and_next_utc_midnight_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_now = datetime(2026, 8, 10, 12, 34, 56, tzinfo=UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is UTC
            return fixed_now

    monkeypatch.setattr(diagnostic_runtime, "datetime", FixedDatetime)
    item = evidence()
    subject, control, _, _ = runtime(active=(manifest(item),), daily_limit=17)
    deadline = RequestDeadline.after(1)

    result = await subject.authorize(deadline=deadline)

    assert isinstance(result, Reservation)
    assert result.units == 1
    assert result.state is ReservationState.AUTHORIZED
    assert control.authorization is not None
    assert control.authorization["reservation"] is result
    assert control.authorization["limit"] == 17
    assert control.authorization["deadline"] is deadline
    window = control.authorization["window_seconds"]
    assert isinstance(window, int)
    assert window == 41_104
    assert 1 <= window <= 86_400


@pytest.mark.asyncio
async def test_indeterminate_authorization_attempts_release_of_exact_reservation() -> None:
    item = evidence()
    control = FakeControl(
        active=(manifest(item),),
        authorize_error=RuntimeError("private-provider-error"),
    )
    subject, _, _, _ = runtime(active=control.active, control=control)

    with pytest.raises(RuntimeError, match="authorization failed") as captured:
        await subject.authorize(deadline=RequestDeadline.after(1))

    assert "private-provider-error" not in str(captured.value)
    assert control.authorization is not None
    assert len(control.releases) == 1
    assert control.releases[0]["reservation"] is control.authorization["reservation"]
    assert isinstance(control.releases[0]["deadline"], RequestDeadline)


@pytest.mark.asyncio
async def test_commit_and_release_pass_the_exact_reservation_and_deadline() -> None:
    item = evidence()
    subject, control, _, _ = runtime(active=(manifest(item),))
    reserved = Reservation(
        reservation_digest="a" * 64,
        budget_digest="b" * 64,
        principal_digest="c" * 64,
        units=1,
        state=ReservationState.AUTHORIZED,
    )
    deadline = RequestDeadline.after(1)

    assert await subject.commit(reservation=reserved, deadline=deadline) is True
    assert await subject.release(reservation=reserved, deadline=deadline) is True

    assert control.commits == [{"reservation": reserved, "deadline": deadline}]
    assert control.releases == [{"reservation": reserved, "deadline": deadline}]
    assert control.commits[0]["reservation"] is reserved
    assert control.releases[0]["reservation"] is reserved


@pytest.mark.asyncio
async def test_embed_enforces_fixed_question_before_one_embed_call() -> None:
    item = evidence()
    subject, _, _, embedder = runtime(active=(manifest(item),))
    deadline = RequestDeadline.after(1)

    with pytest.raises(ValueError, match="question is fixed"):
        await subject.embed(text="What arbitrary question should be sent?", deadline=deadline)
    result = await subject.embed(text=QUESTION, deadline=deadline)

    assert result is VECTOR
    assert embedder.calls == [(QUESTION, deadline)]


@pytest.mark.asyncio
async def test_search_uses_overfetch_limit_and_returns_shared_classifier_aggregates() -> None:
    accepted = evidence(text="Accepted risk disclosure.", ordinal=0)
    weak = evidence(text="Weak risk disclosure.", ordinal=1)
    active = manifest(accepted, weak)
    hits = (
        SearchHit(
            evidence=accepted,
            score=0.91,
            embedding_descriptor=EMBEDDING,
            active_generation_id=active.generation_id,
        ),
        SearchHit(
            evidence=weak,
            score=0.69,
            embedding_descriptor=EMBEDDING,
            active_generation_id=active.generation_id,
        ),
    )
    store = FakeStore(hits=hits)
    subject, _, _, _ = runtime(
        active=(active,),
        store=store,
        maximum_results=2,
        overfetch_factor=3,
        minimum_score=0.70,
    )
    snapshot = DiagnosticPreflight(
        active_generation_count=1,
        pending_cleanup_count=0,
        inspection_state="exact",
        expected_point_count=2,
        observed_point_count=2,
        private_manifest=active,
    )
    deadline = RequestDeadline.after(1)

    result = await subject.search(
        symbol="AAPL",
        filing_type="10-K",
        vector=VECTOR,
        preflight=snapshot,
        deadline=deadline,
    )
    expected = classify_safe_hits(
        "AAPL",
        CORPUS,
        (active,),
        hits,
        minimum_score=0.70,
        max_results=2,
    )

    assert result == expected
    assert result.candidate_count == 2
    assert result.accepted_count == 1
    assert result.rejection_counts.below_threshold == 1
    assert store.searches == [
        {
            "corpus": CORPUS,
            "symbol": "AAPL",
            "vector": VECTOR,
            "active_generations": (active,),
            "limit": 6,
            "deadline": deadline,
        }
    ]


@pytest.mark.asyncio
async def test_search_rejects_non_10k_hit_from_fixed_diagnostic_case() -> None:
    quarterly = evidence(
        text="A quarterly supply constraint disclosure.",
        source=document(filing_type="10-Q"),
    )
    active = manifest(quarterly)
    store = FakeStore(
        hits=(
            SearchHit(
                evidence=quarterly,
                score=0.95,
                embedding_descriptor=EMBEDDING,
                active_generation_id=active.generation_id,
            ),
        )
    )
    subject, _, _, _ = runtime(
        active=(active,),
        store=store,
        maximum_results=5,
        overfetch_factor=4,
    )
    snapshot = DiagnosticPreflight(
        active_generation_count=1,
        pending_cleanup_count=0,
        inspection_state="exact",
        expected_point_count=1,
        observed_point_count=1,
        private_manifest=active,
    )

    result = await subject.search(
        symbol="AAPL",
        filing_type="10-K",
        vector=VECTOR,
        preflight=snapshot,
        deadline=RequestDeadline.after(1),
    )

    assert result.candidate_count == 1
    assert result.accepted_count == 0
    assert result.rejection_counts.metadata_integrity == 1


@pytest.mark.asyncio
async def test_diagnostic_provider_query_breadth_is_exactly_twenty_candidates() -> None:
    request_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        request_body.update(json.loads(request.content))
        return httpx.Response(200, json={"result": []})

    item = evidence()
    active = manifest(item)
    store = UpstashVectorStore(
        "https://research-vector.upstash.io",
        SecretStr("private-vector-token"),
        namespace="sec-filings-v1",
        transport=httpx.MockTransport(handler),
    )
    subject, _, _, _ = runtime(
        active=(active,),
        store=store,  # type: ignore[arg-type]
        maximum_results=5,
        overfetch_factor=4,
    )
    snapshot = DiagnosticPreflight(
        active_generation_count=1,
        pending_cleanup_count=0,
        inspection_state="exact",
        expected_point_count=1,
        observed_point_count=1,
        private_manifest=active,
    )

    try:
        await subject.search(
            symbol="AAPL",
            filing_type="10-K",
            vector=VECTOR,
            preflight=snapshot,
            deadline=RequestDeadline.after(1),
        )
    finally:
        await store.aclose()

    assert request_body["topK"] == 20


@dataclass(frozen=True)
class _Settings:
    openai_api_key: SecretStr = field(default_factory=lambda: SecretStr("private-openai-key"))
    upstash_vector_rest_url: str = "https://research-vector.upstash.io"
    upstash_vector_rest_token: SecretStr = field(
        default_factory=lambda: SecretStr("private-vector-token")
    )
    upstash_redis_rest_url: str = "https://research-redis.upstash.io"
    upstash_redis_rest_token: SecretStr = field(
        default_factory=lambda: SecretStr("private-redis-token")
    )
    research_embedding_provider: str = "openai"
    research_embedding_model: str = "text-embedding-3-small"
    research_embedding_dimensions: int = 1536
    research_vector_provider: str = "upstash"
    research_vector_namespace: str = "sec-filings-v1"
    research_index_schema_version: str = "v1"
    research_chunk_tokens: int = 800
    research_chunk_overlap_tokens: int = 100
    research_max_results: int = 5
    research_vector_overfetch: int = 4
    research_minimum_score: float = 0.70
    research_daily_global_limit: int = 100
    research_enabled: bool = False


def test_builder_constructs_retrieval_only_graph_without_answer_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"answer_generator": 0}
    descriptor = EmbeddingDescriptor(
        provider="openai",
        model="text-embedding-3-small",
        version="2024-01",
        dimensions=1536,
    )

    class BuilderEmbedder:
        def __init__(self, **_kwargs: object) -> None:
            self.descriptor = descriptor

    class BuilderComponent:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

    class BuilderClient:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def aclose(self) -> None:
            return None

    def forbidden_generator(*_args: object, **_kwargs: object) -> object:
        calls["answer_generator"] += 1
        raise AssertionError("retrieval diagnostic must not construct an answer generator")

    monkeypatch.setattr(diagnostic_runtime, "OpenAIEmbedder", BuilderEmbedder)
    monkeypatch.setattr(diagnostic_runtime, "UpstashVectorStore", BuilderComponent)
    monkeypatch.setattr(diagnostic_runtime, "RedisResearchControl", BuilderComponent)
    monkeypatch.setattr(
        "app.research.diagnostic_runtime.httpx.AsyncClient",
        BuilderClient,
    )
    monkeypatch.setattr(
        diagnostic_runtime,
        "OpenAIAnswerGenerator",
        forbidden_generator,
        raising=False,
    )

    built = build_retrieval_diagnostic_runtime(_Settings())

    assert isinstance(built, RetrievalDiagnosticRuntime)
    assert calls["answer_generator"] == 0
    assert not hasattr(built, "_generator")
    assert "OpenAIAnswerGenerator" not in inspect.getsource(build_retrieval_diagnostic_runtime)
