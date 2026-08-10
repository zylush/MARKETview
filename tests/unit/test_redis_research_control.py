from __future__ import annotations

import asyncio
import hashlib
import json
import time
import traceback
from dataclasses import FrozenInstanceError, replace
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.providers.redis_research import (
    _ABORT_SCRIPT,
    _AUTHORIZE_RESERVATION_SCRIPT,
    _CHECKPOINT_SCRIPT,
    _CLAIM_RETRY_SCRIPT,
    _CLEAN_SCRIPT,
    _CLEAN_SUPERSEDED_SCRIPT,
    _FINISH_RETRY_SCRIPT,
    _LEASE_RELEASE_SCRIPT,
    _PUBLISH_SCRIPT,
    _RELEASE_RESERVATION_SCRIPT,
    _RESERVATION_TRANSITION_SCRIPT,
    _STAGE_SCRIPT,
    _VERIFY_SCRIPT,
    RedisResearchControl,
    _accession_digest,
    _cleanup_digest,
    _scope_digest,
)
from app.research.control import (
    AccessionLease,
    GenerationStageRecord,
    GenerationState,
    IngestionCheckpoint,
    IngestionFailureStage,
    IngestionRetryClaim,
    IngestionRetryResult,
    IngestionRetryState,
    ResearchControlUnavailableError,
    Reservation,
    ReservationState,
    SupersededCleanupRecord,
    SupersededCleanupState,
    recoverable_generation_records,
)
from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    EmbeddingDescriptor,
    GenerationManifest,
    GenerationVerification,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64


def _corpus() -> CorpusDescriptor:
    return CorpusDescriptor(
        corpus_version="sec-v1",
        chunker_version="tokens-800-100-v1",
        embedding=EmbeddingDescriptor(
            provider="openai",
            model="text-embedding-3-small",
            version="v1",
            dimensions=1536,
        ),
    )


def _manifest(
    *,
    accession_number: str = "0000320193-25-000079",
    generation_digest: str = _B,
    chunk_digest: str = _C,
) -> GenerationManifest:
    return GenerationManifest(
        corpus=_corpus(),
        symbol="AAPL",
        accession_number=accession_number,
        generation_id=f"gen-{generation_digest}",
        content_hash=_A,
        chunk_ids=(f"chunk-{chunk_digest}",),
    )


def _proof(manifest: GenerationManifest) -> GenerationVerification:
    return GenerationVerification.from_point_ids(manifest.generation_id, manifest.chunk_ids)


def _deadline() -> RequestDeadline:
    return RequestDeadline.after(30)


def _decode_request(request: httpx.Request) -> list[Any]:
    value = json.loads(request.content)
    assert isinstance(value, list)
    return value


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def test_control_models_are_frozen_slotted_and_require_exact_verification() -> None:
    manifest = _manifest()
    lease = AccessionLease(accession_digest=_accession_digest(manifest), owner_digest=_D)
    staged = GenerationStageRecord(manifest, GenerationState.STAGED)
    verified = replace(staged, state=GenerationState.VERIFIED, verification=_proof(manifest))
    checkpoint = IngestionCheckpoint(_D, None, 0, 0, False)
    reservation = Reservation(_E, _C, _D, 2, ReservationState.AUTHORIZED)

    with pytest.raises(FrozenInstanceError):
        lease.owner_digest = _E  # type: ignore[misc]
    for value in (lease, staged, checkpoint, reservation):
        assert not hasattr(value, "__dict__")
    assert GenerationStageRecord.from_json(verified.to_json()) == verified
    assert staged.state is GenerationState.STAGED

    with pytest.raises(ValueError, match="digest"):
        AccessionLease(accession_digest="AAPL", owner_digest=_D)
    with pytest.raises(ValueError, match="positive"):
        replace(reservation, units=True)
    with pytest.raises(ValueError, match="proof"):
        replace(staged, state=GenerationState.VERIFIED)


def test_control_constructor_rejects_unsafe_credentials_and_redacts_repr() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        RedisResearchControl("http://redis.example", SecretStr("token"))
    with pytest.raises(ValueError, match="approved HTTPS root"):
        RedisResearchControl("https://attacker.example", SecretStr("token"))
    with pytest.raises(ValueError, match="credentials"):
        RedisResearchControl("https://private-index.upstash.io", SecretStr(""))
    with pytest.raises(TypeError, match="SecretStr"):
        RedisResearchControl(  # type: ignore[arg-type]
            "https://private-index.upstash.io", "plaintext"
        )

    control = RedisResearchControl("https://private-index.upstash.io", SecretStr("top-secret"))
    rendered = repr(control)
    assert "top-secret" not in rendered
    assert "private-index" not in rendered
    with pytest.raises(ValueError, match="timeout"):
        RedisResearchControl(
            "https://private-index.upstash.io", SecretStr("token"), timeout_seconds=0
        )


@pytest.mark.asyncio
async def test_control_test_endpoint_requires_mock_transport() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="approved HTTPS root"):
            RedisResearchControl("https://redis.example", SecretStr("token"), client=client)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://redis.example",
        "https://upstash.io",
        "https://x.upstash.io.evil.example",
        "https://user@x.upstash.io",
        "https://x.upstash.io:443",
        "https://x.upstash.io/path",
        "https://x.upstash.io?query=1",
        "https://x.upstash.io#fragment",
    ],
)
@pytest.mark.asyncio
async def test_control_mock_transport_rejects_unsafe_endpoints_without_requests(
    endpoint: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"result": "OK"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="approved HTTPS root"):
            RedisResearchControl(endpoint, SecretStr("token"), client=client)

    assert requests == []


def test_control_record_round_trips_and_rejects_malformed_schemas() -> None:
    checkpoint = IngestionCheckpoint(_D, _E, 4, 1, True)
    reservation = Reservation(_E, _C, _D, 2, ReservationState.RELEASED)
    assert IngestionCheckpoint.from_json(checkpoint.to_json()) == checkpoint
    assert Reservation.from_json(reservation.to_json()) == reservation

    with pytest.raises(ValueError, match="schema"):
        IngestionCheckpoint.from_json('{"unexpected":true}')
    with pytest.raises(ValueError, match="reservation"):
        Reservation.from_json(
            _json(
                {
                    "budget_digest": _C,
                    "principal_digest": _D,
                    "reservation_digest": _E,
                    "state": "INVALID",
                    "units": 1,
                }
            )
        )


@pytest.mark.asyncio
async def test_acquire_stage_verify_publish_get_and_release_are_atomic() -> None:
    commands: list[list[Any]] = []
    manifest = _manifest()
    accession_digest = _accession_digest(manifest)
    staged = GenerationStageRecord(manifest, GenerationState.STAGED)
    verified = replace(staged, state=GenerationState.VERIFIED, verification=_proof(manifest))
    published = verified.with_state(GenerationState.PUBLISHED)
    replies: Any = iter(["OK", 1, 1, 1, manifest.generation_id, published.to_json(), 1])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer token"
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example/", SecretStr("token"), client=client)
        lease = await control.acquire_generation_lease(
            manifest=manifest,
            owner_digest=_D,
            ttl_seconds=120,
            deadline=_deadline(),
        )
        assert lease is not None
        actual_staged = await control.stage_generation(
            lease=lease, manifest=manifest, deadline=_deadline()
        )
        assert actual_staged == staged
        actual_verified = await control.mark_generation_verified(
            lease=lease,
            staged=staged,
            verification=_proof(manifest),
            deadline=_deadline(),
        )
        assert actual_verified == verified
        assert await control.publish_generation(
            lease=lease,
            verified_stage=verified,
            expected_previous_generation_id=None,
            manifest=manifest,
            superseded_manifest=None,
            deadline=_deadline(),
        )
        assert (
            await control.get_active_generation(
                corpus=manifest.corpus,
                symbol=manifest.symbol,
                accession_number=manifest.accession_number,
                deadline=_deadline(),
            )
            == manifest
        )
        assert await control.release_generation_lease(lease=lease, deadline=_deadline())

    lease_key = f"rag:v1:lease:{accession_digest}"
    generation_key = f"rag:v1:generation:{accession_digest}:{_B}"
    active_key = f"rag:v1:active:{accession_digest}"
    index_key = f"rag:v1:active-index:{_scope_digest(manifest.corpus, manifest.symbol)}"
    assert commands == [
        ["SET", lease_key, _D, "NX", "EX", 120],
        [
            "EVAL",
            _STAGE_SCRIPT,
            2,
            lease_key,
            generation_key,
            _D,
            *(record.to_json() for record in recoverable_generation_records(manifest)),
        ],
        [
            "EVAL",
            _VERIFY_SCRIPT,
            2,
            lease_key,
            generation_key,
            _D,
            staged.to_json(),
            verified.to_json(),
        ],
        [
            "EVAL",
            _PUBLISH_SCRIPT,
            6,
            lease_key,
            generation_key,
            active_key,
            index_key,
            active_key,
            index_key,
            _D,
            verified.to_json(),
            published.to_json(),
            "",
            manifest.generation_id,
            accession_digest,
            "",
            "",
        ],
        ["GET", active_key],
        ["GET", generation_key],
        ["EVAL", _LEASE_RELEASE_SCRIPT, 1, lease_key, _D],
    ]


@pytest.mark.asyncio
async def test_stage_atomically_reclaims_exact_abandoned_verified_generation() -> None:
    commands: list[list[Any]] = []
    manifest = _manifest()
    staged = GenerationStageRecord(manifest, GenerationState.STAGED)
    new_lease = AccessionLease(_accession_digest(manifest), _C)

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": 1})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert (
            await control.stage_generation(
                lease=new_lease,
                manifest=manifest,
                deadline=_deadline(),
            )
            == staged
        )

    lease_key = f"rag:v1:lease:{new_lease.accession_digest}"
    generation_key = f"rag:v1:generation:{new_lease.accession_digest}:{_B}"
    assert "current==ARGV[2] or current==ARGV[3]" in _STAGE_SCRIPT
    assert commands == [
        [
            "EVAL",
            _STAGE_SCRIPT,
            2,
            lease_key,
            generation_key,
            _C,
            *(record.to_json() for record in recoverable_generation_records(manifest)),
        ]
    ]


@pytest.mark.asyncio
async def test_publish_rejects_wrong_point_proof_and_conflicts_fail_closed() -> None:
    manifest = _manifest()
    lease = AccessionLease(_accession_digest(manifest), _D)
    staged = GenerationStageRecord(manifest, GenerationState.STAGED)
    wrong = GenerationVerification(manifest.generation_id, 2, _E)
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"result": -3})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        with pytest.raises(ValueError, match="point set"):
            await control.mark_generation_verified(
                lease=lease, staged=staged, verification=wrong, deadline=_deadline()
            )
        assert not called
        verified = replace(staged, state=GenerationState.VERIFIED, verification=_proof(manifest))
        superseded = _manifest(generation_digest=_E, chunk_digest=_D)
        assert not await control.publish_generation(
            lease=lease,
            verified_stage=verified,
            expected_previous_generation_id=f"gen-{_E}",
            manifest=manifest,
            superseded_manifest=superseded,
            deadline=_deadline(),
        )


@pytest.mark.asyncio
async def test_abort_and_clean_require_owned_compare_and_swap() -> None:
    commands: list[list[Any]] = []
    manifest = _manifest()
    lease = AccessionLease(_accession_digest(manifest), _D)
    staged = GenerationStageRecord(manifest, GenerationState.STAGED)
    aborted = staged.with_state(GenerationState.ABORTED)
    cleaned = aborted.with_state(GenerationState.CLEANED)
    replies: Any = iter([1, 1])

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert (
            await control.abort_generation(lease=lease, stage=staged, deadline=_deadline())
            == aborted
        )
        assert (
            await control.clean_generation(
                lease=lease,
                aborted_stage=aborted,
                marker_ttl_seconds=86_400,
                deadline=_deadline(),
            )
            == cleaned
        )

    lease_key = f"rag:v1:lease:{lease.accession_digest}"
    generation_key = f"rag:v1:generation:{lease.accession_digest}:{_B}"
    assert commands == [
        [
            "EVAL",
            _ABORT_SCRIPT,
            2,
            lease_key,
            generation_key,
            _D,
            staged.to_json(),
            aborted.to_json(),
        ],
        [
            "EVAL",
            _CLEAN_SCRIPT,
            2,
            lease_key,
            generation_key,
            _D,
            aborted.to_json(),
            cleaned.to_json(),
            86_400,
        ],
    ]


@pytest.mark.asyncio
async def test_replacement_publish_tracks_and_completes_superseded_cleanup_atomically() -> None:
    old = _manifest()
    active = _manifest(generation_digest=_D, chunk_digest=_E)
    lease = AccessionLease(_accession_digest(active), _C)
    staged = GenerationStageRecord(active, GenerationState.STAGED)
    verified = replace(staged, state=GenerationState.VERIFIED, verification=_proof(active))
    published = verified.with_state(GenerationState.PUBLISHED)
    pending = SupersededCleanupRecord(
        superseded_manifest=old,
        active_generation_id=active.generation_id,
        state=SupersededCleanupState.PENDING,
    )
    cleaned = replace(pending, state=SupersededCleanupState.CLEANED)
    cleanup_digest = _cleanup_digest(pending)
    commands: list[list[Any]] = []
    replies: Any = iter([1, [cleanup_digest], [pending.to_json()], 1])

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert await control.publish_generation(
            lease=lease,
            verified_stage=verified,
            expected_previous_generation_id=old.generation_id,
            manifest=active,
            superseded_manifest=old,
            deadline=_deadline(),
        )
        assert await control.list_pending_cleanups(
            corpus=old.corpus, symbol=old.symbol, deadline=_deadline()
        ) == (pending,)
        assert (
            await control.mark_superseded_cleaned(
                lease=lease,
                pending=pending,
                marker_ttl_seconds=86_400,
                deadline=_deadline(),
            )
            == cleaned
        )

    accession_digest = _accession_digest(active)
    lease_key = f"rag:v1:lease:{accession_digest}"
    active_key = f"rag:v1:active:{accession_digest}"
    generation_key = f"rag:v1:generation:{accession_digest}:{_D}"
    active_index = f"rag:v1:active-index:{_scope_digest(active.corpus, active.symbol)}"
    cleanup_key = f"rag:v1:cleanup:{cleanup_digest}"
    cleanup_index = f"rag:v1:cleanup-index:{_scope_digest(old.corpus, old.symbol)}"
    assert commands == [
        [
            "EVAL",
            _PUBLISH_SCRIPT,
            6,
            lease_key,
            generation_key,
            active_key,
            active_index,
            cleanup_key,
            cleanup_index,
            _C,
            verified.to_json(),
            published.to_json(),
            old.generation_id,
            active.generation_id,
            accession_digest,
            pending.to_json(),
            cleanup_digest,
        ],
        ["SMEMBERS", cleanup_index],
        ["MGET", cleanup_key],
        [
            "EVAL",
            _CLEAN_SUPERSEDED_SCRIPT,
            4,
            lease_key,
            active_key,
            cleanup_key,
            cleanup_index,
            _C,
            old.generation_id,
            pending.to_json(),
            cleaned.to_json(),
            cleanup_digest,
            86_400,
        ],
    ]


@pytest.mark.asyncio
async def test_failed_publish_cannot_create_cleanup_and_active_old_cannot_be_cleaned() -> None:
    old = _manifest()
    active = _manifest(generation_digest=_D, chunk_digest=_E)
    lease = AccessionLease(_accession_digest(active), _C)
    verified = GenerationStageRecord(
        active,
        GenerationState.VERIFIED,
        _proof(active),
    )
    pending = SupersededCleanupRecord(
        superseded_manifest=old,
        active_generation_id=active.generation_id,
        state=SupersededCleanupState.PENDING,
    )
    replies: Any = iter([-3, -2])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert not await control.publish_generation(
            lease=lease,
            verified_stage=verified,
            expected_previous_generation_id=old.generation_id,
            manifest=active,
            superseded_manifest=old,
            deadline=_deadline(),
        )
        assert (
            await control.mark_superseded_cleaned(
                lease=lease,
                pending=pending,
                marker_ttl_seconds=86_400,
                deadline=_deadline(),
            )
            is None
        )


@pytest.mark.asyncio
async def test_list_active_generations_batches_and_validates_records() -> None:
    first = _manifest()
    second = _manifest(
        accession_number="0000320193-25-000010",
        generation_digest=_D,
        chunk_digest=_E,
    )
    ordered = sorted((_accession_digest(first), _accession_digest(second)))
    by_digest = {_accession_digest(first): first, _accession_digest(second): second}
    manifests = tuple(by_digest[item] for item in ordered)
    records = tuple(
        GenerationStageRecord(
            manifest,
            GenerationState.PUBLISHED,
            _proof(manifest),
        )
        for manifest in manifests
    )
    replies: Any = iter(
        [
            list(reversed(ordered)),
            [manifest.generation_id for manifest in manifests],
            [record.to_json() for record in records],
        ]
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert (
            await control.list_active_generations(
                corpus=first.corpus, symbol="aapl", deadline=_deadline()
            )
            == manifests
        )


@pytest.mark.asyncio
async def test_checkpoint_compare_and_swap_and_load() -> None:
    initial = IngestionCheckpoint(_D, None, 0, 0, False)
    advanced = IngestionCheckpoint(_D, _E, 12, 1, False)
    commands: list[list[Any]] = []
    replies: Any = iter([1, 1, advanced.to_json()])

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert await control.save_checkpoint(
            checkpoint=initial, expected_previous=None, deadline=_deadline()
        )
        assert await control.save_checkpoint(
            checkpoint=advanced, expected_previous=initial, deadline=_deadline()
        )
        assert await control.load_checkpoint(job_digest=_D, deadline=_deadline()) == advanced

    key = f"rag:v1:checkpoint:{_D}"
    assert commands == [
        ["EVAL", _CHECKPOINT_SCRIPT, 1, key, "", initial.to_json()],
        ["EVAL", _CHECKPOINT_SCRIPT, 1, key, initial.to_json(), advanced.to_json()],
        ["GET", key],
    ]


@pytest.mark.asyncio
async def test_generation_stage_read_is_exact_and_fail_closed() -> None:
    manifest = _manifest()
    cleaned = GenerationStageRecord(manifest, GenerationState.CLEANED)
    commands: list[list[Any]] = []
    replies: Any = iter([cleaned.to_json(), None])

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert (
            await control.get_generation_stage(manifest=manifest, deadline=_deadline()) == cleaned
        )
        assert await control.get_generation_stage(manifest=manifest, deadline=_deadline()) is None

    key = (
        f"rag:v1:generation:{_accession_digest(manifest)}:"
        f"{manifest.generation_id.removeprefix('gen-')}"
    )
    assert commands == [["GET", key], ["GET", key]]


@pytest.mark.asyncio
async def test_failed_checkpoint_retry_claim_and_terminal_result_are_atomic_and_immutable() -> None:
    checkpoint = IngestionCheckpoint(_D, _E, 0, 1, True)
    checkpoint_digest = hashlib.sha256(checkpoint.to_json().encode("utf-8")).hexdigest()
    claim = IngestionRetryClaim(_D, _A, checkpoint_digest, _E)
    terminal = IngestionRetryResult(
        _D,
        _A,
        IngestionRetryState.FAILED,
        IngestionFailureStage.EMBEDDING,
        0,
        0,
    )
    commands: list[list[Any]] = []
    replies: Any = iter(
        [
            [checkpoint.to_json(), None, None],
            1,
            0,
            1,
            1,
            [checkpoint.to_json(), claim.to_json(), terminal.to_json()],
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        empty = await control.load_retry_snapshot(job_digest=_D, deadline=_deadline())
        assert empty.checkpoint == checkpoint
        assert empty.claim is None
        assert empty.result is None
        assert await control.claim_failed_retry(
            checkpoint=checkpoint, claim=claim, deadline=_deadline()
        )
        assert not await control.claim_failed_retry(
            checkpoint=checkpoint, claim=claim, deadline=_deadline()
        )
        assert await control.finish_failed_retry(claim=claim, result=terminal, deadline=_deadline())
        assert await control.finish_failed_retry(claim=claim, result=terminal, deadline=_deadline())
        final = await control.load_retry_snapshot(job_digest=_D, deadline=_deadline())
        assert final.claim == claim
        assert final.result == terminal

    checkpoint_key = f"rag:v1:checkpoint:{_D}"
    claim_key = f"rag:v1:checkpoint-retry:{_D}"
    result_key = f"rag:v1:checkpoint-retry-result:{_D}"
    assert commands == [
        ["MGET", checkpoint_key, claim_key, result_key],
        [
            "EVAL",
            _CLAIM_RETRY_SCRIPT,
            3,
            checkpoint_key,
            claim_key,
            result_key,
            checkpoint.to_json(),
            claim.to_json(),
        ],
        [
            "EVAL",
            _CLAIM_RETRY_SCRIPT,
            3,
            checkpoint_key,
            claim_key,
            result_key,
            checkpoint.to_json(),
            claim.to_json(),
        ],
        [
            "EVAL",
            _FINISH_RETRY_SCRIPT,
            2,
            claim_key,
            result_key,
            claim.to_json(),
            terminal.to_json(),
        ],
        [
            "EVAL",
            _FINISH_RETRY_SCRIPT,
            2,
            claim_key,
            result_key,
            claim.to_json(),
            terminal.to_json(),
        ],
        ["MGET", checkpoint_key, claim_key, result_key],
    ]


@pytest.mark.asyncio
async def test_reservation_authorizes_once_then_only_one_terminal_transition() -> None:
    authorized = Reservation(_E, _C, _D, 3, ReservationState.AUTHORIZED)
    committed = replace(authorized, state=ReservationState.COMMITTED)
    released = replace(authorized, state=ReservationState.RELEASED)
    commands: list[list[Any]] = []
    replies: Any = iter([3, 1, 0, committed.to_json()])

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert await control.authorize_reservation(
            reservation=authorized,
            limit=10,
            window_seconds=86_400,
            deadline=_deadline(),
        )
        assert await control.commit_reservation(reservation=authorized, deadline=_deadline())
        assert not await control.release_reservation(reservation=authorized, deadline=_deadline())
        assert (
            await control.get_reservation(reservation_digest=_E, deadline=_deadline()) == committed
        )

    key = f"rag:v1:reservation:{_E}"
    budget_key = f"rag:v1:budget:{_C}"
    assert commands == [
        [
            "EVAL",
            _AUTHORIZE_RESERVATION_SCRIPT,
            2,
            key,
            budget_key,
            authorized.to_json(),
            3,
            10,
            86_400,
        ],
        [
            "EVAL",
            _RESERVATION_TRANSITION_SCRIPT,
            1,
            key,
            authorized.to_json(),
            committed.to_json(),
        ],
        [
            "EVAL",
            _RELEASE_RESERVATION_SCRIPT,
            2,
            key,
            budget_key,
            authorized.to_json(),
            released.to_json(),
            3,
        ],
        ["GET", key],
    ]


@pytest.mark.asyncio
async def test_releasing_authorized_reservation_decrements_daily_budget_once() -> None:
    authorized = Reservation(_E, _C, _D, 3, ReservationState.AUTHORIZED)
    released = replace(authorized, state=ReservationState.RELEASED)
    commands: list[list[Any]] = []
    replies: Any = iter([3, 1, 0, released.to_json()])

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert await control.authorize_reservation(
            reservation=authorized,
            limit=10,
            window_seconds=86_400,
            deadline=_deadline(),
        )
        assert await control.release_reservation(reservation=authorized, deadline=_deadline())
        assert not await control.release_reservation(reservation=authorized, deadline=_deadline())
        assert (
            await control.get_reservation(reservation_digest=_E, deadline=_deadline()) == released
        )

    key = f"rag:v1:reservation:{_E}"
    budget_key = f"rag:v1:budget:{_C}"
    assert commands[1:] == [
        [
            "EVAL",
            _RELEASE_RESERVATION_SCRIPT,
            2,
            key,
            budget_key,
            authorized.to_json(),
            released.to_json(),
            3,
        ],
        [
            "EVAL",
            _RELEASE_RESERVATION_SCRIPT,
            2,
            key,
            budget_key,
            authorized.to_json(),
            released.to_json(),
            3,
        ],
        ["GET", key],
    ]


@pytest.mark.asyncio
async def test_conflicts_missing_records_and_budget_rejection_fail_closed() -> None:
    manifest = _manifest()
    lease = AccessionLease(_accession_digest(manifest), _D)
    staged = GenerationStageRecord(manifest, GenerationState.STAGED)
    aborted = staged.with_state(GenerationState.ABORTED)
    authorized = Reservation(_E, _C, _D, 3, ReservationState.AUTHORIZED)
    replies: Any = iter([None, -1, -2, -1, 0, 0, None, [], None, None, -1, -2])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert (
            await control.acquire_generation_lease(
                manifest=manifest, owner_digest=_D, ttl_seconds=30, deadline=_deadline()
            )
            is None
        )
        assert (
            await control.stage_generation(lease=lease, manifest=manifest, deadline=_deadline())
            is None
        )
        assert (
            await control.mark_generation_verified(
                lease=lease,
                staged=staged,
                verification=_proof(manifest),
                deadline=_deadline(),
            )
            is None
        )
        assert (
            await control.abort_generation(lease=lease, stage=staged, deadline=_deadline()) is None
        )
        assert (
            await control.clean_generation(
                lease=lease,
                aborted_stage=aborted,
                marker_ttl_seconds=60,
                deadline=_deadline(),
            )
            is None
        )
        assert not await control.release_generation_lease(lease=lease, deadline=_deadline())
        assert (
            await control.get_active_generation(
                corpus=manifest.corpus,
                symbol=manifest.symbol,
                accession_number=manifest.accession_number,
                deadline=_deadline(),
            )
            is None
        )
        assert (
            await control.list_active_generations(
                corpus=manifest.corpus, symbol=manifest.symbol, deadline=_deadline()
            )
            == ()
        )
        assert await control.load_checkpoint(job_digest=_D, deadline=_deadline()) is None
        assert await control.get_reservation(reservation_digest=_E, deadline=_deadline()) is None
        assert not await control.authorize_reservation(
            reservation=authorized,
            limit=10,
            window_seconds=86_400,
            deadline=_deadline(),
        )
        with pytest.raises(ResearchControlUnavailableError):
            await control.authorize_reservation(
                reservation=authorized,
                limit=10,
                window_seconds=86_400,
                deadline=_deadline(),
            )


@pytest.mark.asyncio
async def test_expired_deadline_and_network_failure_are_sanitized() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        raise httpx.ConnectError("private transport detail", request=request)

    expired = RequestDeadline(expires_at=0.0, _clock=lambda: 1.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        with pytest.raises(TimeoutError, match="deadline"):
            await control.acquire_generation_lease(
                manifest=_manifest(),
                owner_digest=_D,
                ttl_seconds=30,
                deadline=expired,
            )
        assert not called
        with pytest.raises(ResearchControlUnavailableError) as caught:
            await control.acquire_generation_lease(
                manifest=_manifest(),
                owner_digest=_D,
                ttl_seconds=30,
                deadline=_deadline(),
            )
    assert "private transport" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "status"),
    [
        (b"private provider body", 503),
        (b"not-json", 200),
        (_json({"error": "top-secret provider error"}).encode(), 200),
        (_json({"result": {"unexpected": True}}).encode(), 200),
    ],
)
async def test_control_fails_closed_without_leaking_provider_or_token_details(
    body: bytes, status: int
) -> None:
    provider_sentinel = "very-private-token"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl(
            "https://private-index.upstash.io", SecretStr(provider_sentinel), client=client
        )
        with pytest.raises(ResearchControlUnavailableError) as caught:
            await control.acquire_generation_lease(
                manifest=_manifest(),
                owner_digest=_D,
                ttl_seconds=30,
                deadline=_deadline(),
            )

    rendered = "".join(traceback.format_exception(caught.value))
    for private_value in (
        provider_sentinel,
        "private-index",
        "private provider body",
        "top-secret",
    ):
        assert private_value not in str(caught.value)
        assert private_value not in repr(caught.value)
        assert private_value not in rendered
        for frame, _ in traceback.walk_tb(caught.value.__traceback__):
            if frame.f_code.co_filename.endswith("redis_research.py"):
                assert private_value not in repr(frame.f_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_owned_http_client_disables_environment_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_kwargs: dict[str, object] = {}

    class OwnedClientProbe:
        def __init__(self, **kwargs: object) -> None:
            client_kwargs.update(kwargs)
            self.is_closed = False

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr(httpx, "AsyncClient", OwnedClientProbe)

    control = RedisResearchControl("https://private-index.upstash.io", SecretStr("token"))

    assert client_kwargs["trust_env"] is False
    assert client_kwargs["follow_redirects"] is False
    await control.aclose()
    assert control.is_closed


@pytest.mark.asyncio
async def test_control_never_follows_redirects_from_injected_client() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "redis.example":
            return httpx.Response(307, headers={"Location": "https://attacker.example"})
        return httpx.Response(200, json={"result": "OK"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        with pytest.raises(ResearchControlUnavailableError):
            await control._command(["PING"], deadline=_deadline())

    assert len(requests) == 1
    assert requests[0].url.host == "redis.example"


@pytest.mark.asyncio
async def test_control_rejects_oversized_streamed_response() -> None:
    oversized = _json({"result": "x" * 1_048_576}).encode()

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        with pytest.raises(ResearchControlUnavailableError):
            await control._command(["PING"], deadline=_deadline())


@pytest.mark.asyncio
async def test_streaming_response_enforces_absolute_deadline_between_chunks() -> None:
    now = [0.0]

    class SlowDripStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"result":"private provider '
            now[0] = 2.0
            await asyncio.sleep(0)
            yield b'body"}'

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=SlowDripStream())

    deadline = RequestDeadline(expires_at=1.0, _clock=lambda: now[0])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl(
            "https://redis.example",
            SecretStr("very-private-token"),
            client=client,
        )
        with pytest.raises(TimeoutError, match="deadline") as caught:
            await control._command(["PING"], deadline=deadline)

    for frame, _ in traceback.walk_tb(caught.value.__traceback__):
        if frame.f_code.co_filename.endswith("redis_research.py"):
            rendered_locals = repr(frame.f_locals)
            assert "private provider" not in rendered_locals
            assert "very-private-token" not in rendered_locals


@pytest.mark.asyncio
async def test_streaming_request_has_a_whole_exchange_deadline() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, json={"result": "OK"})

    started = time.monotonic()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        with pytest.raises(TimeoutError, match="deadline"):
            await control._command(["PING"], deadline=RequestDeadline.after(0.01))

    assert time.monotonic() - started < 0.1


@pytest.mark.asyncio
async def test_control_closes_only_an_owned_http_client() -> None:
    owned = RedisResearchControl("https://private-index.upstash.io", SecretStr("token"))
    await owned.aclose()
    assert owned.is_closed

    external = httpx.AsyncClient()
    injected = RedisResearchControl(
        "https://private-index.upstash.io", SecretStr("token"), client=external
    )
    await injected.aclose()
    assert not external.is_closed
    await external.aclose()
