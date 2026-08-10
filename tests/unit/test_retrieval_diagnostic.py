from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError, dataclass, field, replace
from datetime import date
from typing import Any

import pytest
from pydantic import SecretStr

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
from app.research.retrieval import (
    SafeHitClassification,
    SafeHitRejectionCounts,
    classify_safe_hits,
)
from app.retrieval_diagnostic import (
    EXIT_CONFIG,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    DiagnosticPreflight,
    async_main,
)

EMBEDDING = EmbeddingDescriptor(
    provider="openai",
    model="text-embedding-3-small",
    version="2024-01",
    dimensions=2,
)
CORPUS = CorpusDescriptor(
    corpus_version="sec-filings-v1",
    chunker_version="tokens-800-80-v1",
    embedding=EMBEDDING,
)
VECTOR = EmbeddingVector(descriptor=EMBEDDING, values=(1.0, 0.0))


def filing() -> FilingDocument:
    return FilingDocument(
        symbol="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000001",
        filing_type="10-K",
        title="AAPL 2025 Form 10-K",
        filed_date=date(2025, 10, 31),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/aapl-20250927.htm"
        ),
        text="private-content-sentinel Apple reports supply constraints.",
    )


def evidence(
    *,
    score_text: str = "private-content-sentinel Apple reports supply constraints.",
) -> EvidenceChunk:
    return EvidenceChunk.from_document(
        filing(),
        corpus=CORPUS,
        ordinal=0,
        text=score_text,
    )


def manifest(item: EvidenceChunk) -> GenerationManifest:
    return GenerationManifest(
        corpus=CORPUS,
        symbol="AAPL",
        accession_number=item.accession_number,
        generation_id=item.generation_id,
        content_hash=item.content_hash,
        chunk_ids=(item.chunk_id,),
    )


def classification(*, score: float = 0.95) -> SafeHitClassification:
    item = evidence()
    active = manifest(item)
    return classify_safe_hits(
        symbol="AAPL",
        corpus=CORPUS,
        active_manifests=(active,),
        hits=(
            SearchHit(
                evidence=item,
                score=score,
                embedding_descriptor=EMBEDDING,
                active_generation_id=item.generation_id,
            ),
        ),
        minimum_score=0.70,
        max_results=5,
    )


def mixed_unsafe_classification(reason: str) -> SafeHitClassification:
    accepted = classification()
    return replace(
        accepted,
        candidate_count=2,
        rejection_counts=SafeHitRejectionCounts(**{reason: 1}),
    )


def reservation() -> Reservation:
    return Reservation(
        reservation_digest="a" * 64,
        budget_digest="b" * 64,
        principal_digest="c" * 64,
        units=1,
        state=ReservationState.AUTHORIZED,
    )


def preflight(**changes: object) -> DiagnosticPreflight:
    item = evidence()
    values: dict[str, object] = {
        "active_generation_count": 1,
        "pending_cleanup_count": 0,
        "inspection_state": "exact",
        "expected_point_count": 1,
        "observed_point_count": 1,
        "private_manifest": manifest(item),
    }
    values.update(changes)
    return DiagnosticPreflight(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _Settings:
    openai_api_key: SecretStr = field(
        default_factory=lambda: SecretStr("private-openai-key-sentinel")
    )
    upstash_vector_rest_token: SecretStr = field(
        default_factory=lambda: SecretStr("private-vector-token-sentinel")
    )
    upstash_vector_rest_url: str = "https://research-vector.upstash.io"
    upstash_redis_rest_token: SecretStr = field(
        default_factory=lambda: SecretStr("private-redis-token-sentinel")
    )
    upstash_redis_rest_url: str = "https://research-redis.upstash.io"
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


_DEFAULT_AUTHORIZATION = object()


class FakeRuntime:
    def __init__(
        self,
        *,
        snapshot: DiagnosticPreflight | None = None,
        authorized: Reservation | object | None = _DEFAULT_AUTHORIZATION,
        classification_result: SafeHitClassification | object | None = None,
        fail_at: str | None = None,
        timeout_at: str | None = None,
        revalidate_result: bool = True,
        commit_result: bool = True,
    ) -> None:
        self.snapshot = snapshot or preflight()
        self.authorized = reservation() if authorized is _DEFAULT_AUTHORIZATION else authorized
        self.classification_result = classification_result or classification()
        self.fail_at = fail_at
        self.timeout_at = timeout_at
        self.revalidate_result = revalidate_result
        self.commit_result = commit_result
        self.events: list[str] = []
        self.arguments: dict[str, dict[str, object]] = {}
        self.generate_calls = 0

    def _record(self, name: str, values: dict[str, object]) -> None:
        self.events.append(name)
        self.arguments[name] = values
        if self.timeout_at == name:
            raise TimeoutError("private-timeout-exception-sentinel")
        if self.fail_at == name:
            raise RuntimeError("private-provider-exception-sentinel")

    async def preflight(self, **kwargs: object) -> DiagnosticPreflight:
        self._record("preflight", kwargs)
        return self.snapshot

    async def authorize(self, **kwargs: object) -> Reservation | None:
        self._record("authorize", kwargs)
        assert self.authorized is None or isinstance(self.authorized, Reservation)
        return self.authorized

    async def revalidate(self, **kwargs: object) -> bool:
        self._record("revalidate", kwargs)
        return self.revalidate_result

    async def commit(self, **kwargs: object) -> bool:
        self._record("commit", kwargs)
        return self.commit_result

    async def embed(self, **kwargs: object) -> EmbeddingVector:
        self._record("embed", kwargs)
        return VECTOR

    async def search(self, **kwargs: object) -> SafeHitClassification | object:
        self._record("search", kwargs)
        return self.classification_result

    async def release(self, **kwargs: object) -> bool:
        self._record("release", kwargs)
        return True

    async def generate(self, **kwargs: object) -> None:
        self.generate_calls += 1
        self._record("generate", kwargs)

    async def aclose(self) -> None:
        self.events.append("aclose")


LIVE_FLAGS = ["--apply", "--acknowledge-live-retrieval-diagnostic"]


def output_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


@pytest.mark.asyncio
async def test_diagnostic_rejects_five_by_three_candidate_contract_before_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = replace(_Settings(), research_vector_overfetch=3)

    exit_code = await async_main(
        [],
        settings_factory=lambda: settings,
        runtime_factory=lambda _settings: pytest.fail("runtime constructed"),
    )

    output = output_json(capsys)
    assert exit_code == EXIT_CONFIG
    assert output["error_category"] == "configuration"
    assert output["call_counts"]["vector_search"] == 0


@pytest.mark.asyncio
async def test_diagnostic_rejects_nonfixed_result_limit_before_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = replace(_Settings(), research_max_results=4)

    exit_code = await async_main(
        [],
        settings_factory=lambda: settings,
        runtime_factory=lambda _settings: pytest.fail("runtime constructed"),
    )

    output = output_json(capsys)
    assert exit_code == EXIT_CONFIG
    assert output["error_category"] == "configuration"
    assert output["call_counts"]["vector_search"] == 0


@pytest.mark.asyncio
async def test_diagnostic_rejects_non_upstash_redis_endpoint_before_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = replace(_Settings(), upstash_redis_rest_url="https://attacker.example")

    exit_code = await async_main(
        [],
        settings_factory=lambda: settings,
        runtime_factory=lambda _settings: pytest.fail("runtime constructed"),
    )

    output = output_json(capsys)
    assert exit_code == EXIT_CONFIG
    assert output["error_category"] == "configuration"
    assert output["call_counts"]["vector_search"] == 0


@pytest.mark.asyncio
async def test_diagnostic_requires_research_to_remain_disabled_before_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = replace(_Settings(), research_enabled=True)

    exit_code = await async_main(
        [],
        settings_factory=lambda: settings,
        runtime_factory=lambda _settings: pytest.fail("runtime constructed"),
    )

    output = output_json(capsys)
    assert exit_code == EXIT_CONFIG
    assert output["error_category"] == "configuration"
    assert output["call_counts"]["embedding"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("minimum_score", [0.69, 0.71])
async def test_diagnostic_rejects_nonfixed_minimum_score_before_runtime(
    minimum_score: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = replace(_Settings(), research_minimum_score=minimum_score)

    exit_code = await async_main(
        [],
        settings_factory=lambda: settings,
        runtime_factory=lambda _settings: pytest.fail("runtime constructed"),
    )

    output = output_json(capsys)
    assert exit_code == EXIT_CONFIG
    assert output["error_category"] == "configuration"
    assert output["call_counts"]["embedding"] == 0


def test_preflight_is_frozen_and_hides_private_manifest_identifiers() -> None:
    snapshot = preflight()

    with pytest.raises(FrozenInstanceError):
        snapshot.observed_point_count = 0  # type: ignore[misc]
    rendered = repr(snapshot)
    assert snapshot.private_manifest is not None
    assert snapshot.private_manifest.generation_id not in rendered
    assert snapshot.private_manifest.chunk_ids[0] not in rendered


@pytest.mark.asyncio
async def test_cli_defaults_to_dry_run_without_constructing_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    constructed = False

    def runtime_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("runtime must not be constructed during dry-run")

    exit_code = await async_main(
        [],
        settings_factory=_Settings,
        runtime_factory=runtime_factory,
    )

    output = output_json(capsys)
    assert exit_code == EXIT_SUCCESS
    assert output["applied"] is False
    assert output["dry_run"] is True
    assert output["diagnostic_case"] == "aapl_latest_10k_risks_v1"
    assert output["call_counts"] == {
        "answer_generation": 0,
        "embedding": 0,
        "vector_search": 0,
    }
    assert constructed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setting_name", "unsafe_endpoint"),
    [
        (
            "upstash_vector_rest_url",
            "https://private-vector-host-sentinel.example.com",
        ),
        (
            "upstash_vector_rest_url",
            "https://private-vector-sentinel.upstash.io/private-path-sentinel",
        ),
        (
            "upstash_vector_rest_url",
            "https://private-vector-sentinel.upstash.io?private-query-sentinel=1",
        ),
        (
            "upstash_vector_rest_url",
            "https://private-user-sentinel@private-vector-sentinel.upstash.io",
        ),
        (
            "upstash_vector_rest_url",
            "https://private-vector-sentinel.upstash.io:8443",
        ),
        (
            "upstash_redis_rest_url",
            "https://private-redis-sentinel.example.com/private-path-sentinel",
        ),
        (
            "upstash_redis_rest_url",
            "https://private-redis-sentinel.example.com?private-query-sentinel=1",
        ),
        (
            "upstash_redis_rest_url",
            "https://private-user-sentinel@private-redis-sentinel.example.com",
        ),
        (
            "upstash_redis_rest_url",
            "https://private-redis-sentinel.example.com:8443",
        ),
        (
            "upstash_redis_rest_url",
            "https://private-redis-sentinel.example.com#private-fragment-sentinel",
        ),
    ],
)
async def test_dry_run_rejects_unsafe_endpoints_without_echoing_values(
    setting_name: str,
    unsafe_endpoint: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = (
        replace(_Settings(), upstash_vector_rest_url=unsafe_endpoint)
        if setting_name == "upstash_vector_rest_url"
        else replace(_Settings(), upstash_redis_rest_url=unsafe_endpoint)
    )

    exit_code = await async_main([], settings_factory=lambda: settings)

    rendered = json.dumps(output_json(capsys), sort_keys=True)
    assert exit_code == EXIT_CONFIG
    assert "private-" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("flags", [["--apply"], ["--acknowledge-live-retrieval-diagnostic"]])
async def test_cli_requires_both_live_flags_before_settings_or_runtime(
    flags: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings_calls = 0

    def settings_factory() -> object:
        nonlocal settings_calls
        settings_calls += 1
        raise AssertionError("settings must not be read for rejected authorization")

    exit_code = await async_main(
        flags,
        settings_factory=settings_factory,
        runtime_factory=lambda *_args, **_kwargs: pytest.fail("runtime constructed"),
    )

    output = output_json(capsys)
    assert exit_code == EXIT_CONFIG
    assert output["error_category"] == "authorization"
    assert output["call_counts"]["embedding"] == 0
    assert settings_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_value", ["0", "nan", "121", "private-timeout-sentinel"])
async def test_cli_rejects_invalid_timeout_before_settings_or_runtime(
    timeout_value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = await async_main(
        [*LIVE_FLAGS, "--timeout-seconds", timeout_value],
        settings_factory=lambda: pytest.fail("settings read"),
        runtime_factory=lambda *_args, **_kwargs: pytest.fail("runtime constructed"),
    )

    output = output_json(capsys)
    assert exit_code == EXIT_CONFIG
    assert output["error_category"] == "configuration"
    assert output["call_counts"]["embedding"] == 0
    assert "private-timeout-sentinel" not in json.dumps(output)


@pytest.mark.asyncio
async def test_live_cli_runs_fixed_retrieval_in_exact_order_once_without_generation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime()

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    assert exit_code == EXIT_SUCCESS
    assert runtime.events == [
        "preflight",
        "authorize",
        "revalidate",
        "commit",
        "embed",
        "search",
        "aclose",
    ]
    assert runtime.arguments["preflight"]["symbol"] == "AAPL"
    assert runtime.arguments["preflight"]["filing_type"] == "10-K"
    assert runtime.arguments["revalidate"]["preflight"] is runtime.snapshot
    assert runtime.arguments["commit"]["reservation"] is runtime.authorized
    assert runtime.arguments["search"]["preflight"] is runtime.snapshot
    assert runtime.generate_calls == 0
    assert output["call_counts"] == {
        "answer_generation": 0,
        "embedding": 1,
        "vector_search": 1,
    }
    assert output["candidate_count"] == 1
    assert output["accepted_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_exit"),
    [
        ("revalidate_false", EXIT_CONFIG),
        ("commit_false", EXIT_CONFIG),
        ("revalidate_error", EXIT_FAILURE),
    ],
)
async def test_precommit_failures_release_the_exact_authorized_reservation(
    failure: str,
    expected_exit: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime(
        revalidate_result=failure != "revalidate_false",
        commit_result=failure != "commit_false",
        fail_at="revalidate" if failure == "revalidate_error" else None,
    )

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    assert exit_code == expected_exit
    assert runtime.events[-2:] == ["release", "aclose"]
    assert runtime.arguments["release"]["reservation"] is runtime.authorized
    assert "embed" not in runtime.events
    assert "search" not in runtime.events
    assert output["call_counts"] == {
        "answer_generation": 0,
        "embedding": 0,
        "vector_search": 0,
    }


@pytest.mark.asyncio
async def test_cancellation_after_authorization_releases_exact_reservation_then_reraises() -> None:
    reached_revalidation = asyncio.Event()

    class RuntimeCancelledDuringRevalidation(FakeRuntime):
        async def revalidate(self, **kwargs: object) -> bool:
            self._record("revalidate", kwargs)
            reached_revalidation.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    runtime = RuntimeCancelledDuringRevalidation()
    task = asyncio.create_task(
        async_main(
            LIVE_FLAGS,
            settings_factory=_Settings,
            runtime_factory=lambda _settings: runtime,
        )
    )
    await reached_revalidation.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.events == ["preflight", "authorize", "revalidate", "release", "aclose"]
    assert runtime.arguments["release"]["reservation"] is runtime.authorized
    release_deadline = runtime.arguments["release"]["deadline"]
    assert isinstance(release_deadline, RequestDeadline)
    assert release_deadline is not runtime.arguments["revalidate"]["deadline"]
    assert 0 < release_deadline.remaining_seconds() <= 0.25


@pytest.mark.asyncio
async def test_cancellation_once_commit_starts_never_releases_reservation() -> None:
    commit_started = asyncio.Event()

    class RuntimeCancelledDuringCommit(FakeRuntime):
        async def commit(self, **kwargs: object) -> bool:
            self._record("commit", kwargs)
            commit_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    runtime = RuntimeCancelledDuringCommit()
    task = asyncio.create_task(
        async_main(
            LIVE_FLAGS,
            settings_factory=_Settings,
            runtime_factory=lambda _settings: runtime,
        )
    )
    await commit_started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.events == ["preflight", "authorize", "revalidate", "commit", "aclose"]
    assert "release" not in runtime.events


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["embed", "search"])
async def test_postcommit_failures_retain_charge_and_never_release(
    failure: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime(fail_at=failure)

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    assert exit_code == EXIT_FAILURE
    assert "commit" in runtime.events
    assert "release" not in runtime.events
    assert runtime.events[-1] == "aclose"
    assert output["call_counts"]["answer_generation"] == 0
    assert runtime.generate_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot",
    [
        preflight(active_generation_count=0, private_manifest=None),
        preflight(pending_cleanup_count=1),
        preflight(inspection_state="missing", observed_point_count=0),
        preflight(expected_point_count=2, observed_point_count=1),
    ],
)
async def test_failed_preflight_stops_before_budget_or_paid_calls(
    snapshot: DiagnosticPreflight,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime(snapshot=snapshot)

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    assert exit_code == EXIT_CONFIG
    assert runtime.events == ["preflight", "aclose"]
    assert output["call_counts"] == {
        "answer_generation": 0,
        "embedding": 0,
        "vector_search": 0,
    }
    assert output["error_category"] == "preflight"


@pytest.mark.asyncio
async def test_exhausted_budget_stops_before_revalidation_and_paid_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime(authorized=None)

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    assert exit_code == EXIT_CONFIG
    assert runtime.events == ["preflight", "authorize", "aclose"]
    assert output["call_counts"] == {
        "answer_generation": 0,
        "embedding": 0,
        "vector_search": 0,
    }
    assert output["error_category"] == "budget"


def empty_classification() -> SafeHitClassification:
    item = evidence()
    return classify_safe_hits(
        symbol="AAPL",
        corpus=CORPUS,
        active_manifests=(manifest(item),),
        hits=(),
        minimum_score=0.70,
        max_results=5,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "candidate_count", "below_threshold"),
    [
        (empty_classification(), 0, 0),
        (classification(score=0.69), 1, 1),
    ],
)
async def test_no_candidates_and_all_below_threshold_are_aggregate_failures(
    result: SafeHitClassification,
    candidate_count: int,
    below_threshold: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime(classification_result=result)

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    assert exit_code == EXIT_FAILURE
    assert output["passed"] is False
    assert output["candidate_count"] == candidate_count
    assert output["accepted_count"] == 0
    assert output["accepted_score_range"] is None
    assert output["rejection_counts"]["below_threshold"] == below_threshold
    assert output["call_counts"] == {
        "answer_generation": 0,
        "embedding": 1,
        "vector_search": 1,
    }
    assert "release" not in runtime.events


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["metadata_integrity", "prompt_injection"])
async def test_any_integrity_or_injection_rejection_fails_even_with_an_accepted_hit(
    reason: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime(classification_result=mixed_unsafe_classification(reason))

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    assert exit_code == EXIT_FAILURE
    assert output["passed"] is False
    assert output["accepted_count"] == 1
    assert output["rejection_counts"][reason] == 1
    assert output["error_category"] == "retrieval"
    assert output["budget_units_committed"] == 1
    assert "release" not in runtime.events


@pytest.mark.asyncio
async def test_malformed_search_result_fails_closed_after_commit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime(classification_result={"accepted_hits": ["private-content-sentinel"]})

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    assert exit_code == EXIT_FAILURE
    assert output["error_category"] == "search"
    assert "release" not in runtime.events
    assert "private-content-sentinel" not in json.dumps(output)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "release_expected", "embedding_calls", "search_calls"),
    [
        ("preflight", False, 0, 0),
        ("authorize", False, 0, 0),
        ("revalidate", True, 0, 0),
        ("commit", False, 0, 0),
        ("embed", False, 1, 0),
        ("search", False, 1, 1),
    ],
)
@pytest.mark.parametrize("failure_kind", ["provider", "timeout"])
async def test_provider_errors_and_timeouts_are_sanitized_with_commit_safe_release(
    phase: str,
    release_expected: bool,
    embedding_calls: int,
    search_calls: int,
    failure_kind: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime(
        fail_at=phase if failure_kind == "provider" else None,
        timeout_at=phase if failure_kind == "timeout" else None,
    )

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    rendered = json.dumps(output)
    assert exit_code == EXIT_FAILURE
    assert ("release" in runtime.events) is release_expected
    assert output["call_counts"] == {
        "answer_generation": 0,
        "embedding": embedding_calls,
        "vector_search": search_calls,
    }
    assert output["error_category"] in {
        "preflight",
        "authorization",
        "revalidation",
        "commit",
        "embedding",
        "search",
        "timeout",
    }
    assert "private-provider-exception-sentinel" not in rendered
    assert "private-timeout-exception-sentinel" not in rendered


@pytest.mark.asyncio
async def test_aggregate_json_never_exposes_secrets_content_or_provider_identifiers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = classification()
    accepted = result.accepted_hits[0]
    runtime = FakeRuntime(classification_result=result)

    exit_code = await async_main(
        LIVE_FLAGS,
        settings_factory=_Settings,
        runtime_factory=lambda _settings: runtime,
    )

    output = output_json(capsys)
    rendered = json.dumps(output, sort_keys=True)
    assert exit_code == EXIT_SUCCESS
    assert output["rejection_counts"] == {
        "below_threshold": 0,
        "duplicate_chunk": 0,
        "inactive_manifest": 0,
        "metadata_integrity": 0,
        "prompt_injection": 0,
        "result_limit": 0,
    }
    for forbidden in (
        "private-openai-key-sentinel",
        "private-vector-token-sentinel",
        "private-redis-token-sentinel",
        "private-content-sentinel",
        accepted.evidence.text,
        accepted.evidence.chunk_id,
        accepted.evidence.generation_id,
        accepted.evidence.accession_number,
        accepted.evidence.content_hash,
        accepted.evidence.source_url,
    ):
        assert forbidden not in rendered
