from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.providers.redis_research import (
    _CLAIM_RETRY_ATTEMPT_TWO_SCRIPT,
    _FINISH_RETRY_ATTEMPT_TWO_SCRIPT,
    RedisResearchControl,
)
from app.research.control import (
    IngestionCheckpoint,
    IngestionFailureStage,
    IngestionRetryAttemptTwoClaim,
    IngestionRetryAttemptTwoSnapshot,
    IngestionRetryClaim,
    IngestionRetryResult,
    IngestionRetryState,
    ResearchControlUnavailableError,
)
from app.research.deadline import RequestDeadline

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64

# Exact canonical schemas introduced by ca77a251e56921168a0185b73eea2aa3f10747f8.
_CA77A25_CHECKPOINT_RAW = (
    '{"complete":true,"cursor_digest":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
    'eeeeeeeeeeeeeeee","failed_count":1,"job_digest":"dddddddddddddddddddddddddddddddddddd'
    'dddddddddddddddddddddddddddd","processed_count":0}'
)
_CA77A25_FIRST_CLAIM_RAW = (
    '{"attempt_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    '"checkpoint_digest":"7a67fd25bcc294ec9f1563bc220bf9aad5b742eef0188a84d58c75e9683f1bcc",'
    '"cursor_digest":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",'
    '"job_digest":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}'
)
_CA77A25_FIRST_RESULT_RAW = (
    '{"attempt_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    '"failure_stage":"vector_verification","inserted_count":0,"job_digest":"dddddddddddd'
    'dddddddddddddddddddddddddddddddddddddddddddddddddddd","removed_count":0,"state":"FAILED"}'
)


def _deadline() -> RequestDeadline:
    return RequestDeadline.after(30)


def _decode_request(request: httpx.Request) -> list[Any]:
    value = json.loads(request.content)
    assert isinstance(value, list)
    return value


def _legacy_records() -> tuple[IngestionCheckpoint, IngestionRetryClaim, IngestionRetryResult]:
    return (
        IngestionCheckpoint.from_json(_CA77A25_CHECKPOINT_RAW),
        IngestionRetryClaim.from_json(_CA77A25_FIRST_CLAIM_RAW),
        IngestionRetryResult.from_json(_CA77A25_FIRST_RESULT_RAW),
    )


def _attempt_two_claim() -> IngestionRetryAttemptTwoClaim:
    checkpoint, first_claim, first_result = _legacy_records()
    return IngestionRetryAttemptTwoClaim(
        job_digest=_D,
        attempt_digest=_B,
        checkpoint_digest=hashlib.sha256(checkpoint.to_json().encode("utf-8")).hexdigest(),
        cursor_digest=_E,
        first_claim_digest=hashlib.sha256(first_claim.to_json().encode("utf-8")).hexdigest(),
        first_result_digest=hashlib.sha256(first_result.to_json().encode("utf-8")).hexdigest(),
    )


def _attempt_two_result(
    *,
    state: IngestionRetryState = IngestionRetryState.SUCCEEDED,
    stage: IngestionFailureStage | None = None,
) -> IngestionRetryResult:
    return IngestionRetryResult(_D, _B, state, stage, 0, 0)


def test_attempt_two_claim_is_private_immutable_and_chains_exact_legacy_bytes() -> None:
    checkpoint, first_claim, first_result = _legacy_records()
    claim = _attempt_two_claim()

    assert type(claim).from_json(claim.to_json()) == claim
    assert set(json.loads(claim.to_json())) == {
        "attempt_digest",
        "checkpoint_digest",
        "cursor_digest",
        "first_claim_digest",
        "first_result_digest",
        "job_digest",
    }
    assert "AAPL" not in claim.to_json()
    assert "0000320193" not in claim.to_json()
    assert (
        claim.checkpoint_digest == hashlib.sha256(checkpoint.to_json().encode("utf-8")).hexdigest()
    )
    assert (
        claim.first_claim_digest
        == hashlib.sha256(first_claim.to_json().encode("utf-8")).hexdigest()
    )
    assert (
        claim.first_result_digest
        == hashlib.sha256(first_result.to_json().encode("utf-8")).hexdigest()
    )
    with pytest.raises(FrozenInstanceError):
        claim.attempt_digest = _C  # type: ignore[misc]


def test_ca77a25_legacy_records_round_trip_byte_for_byte() -> None:
    checkpoint, first_claim, first_result = _legacy_records()

    assert checkpoint.to_json() == _CA77A25_CHECKPOINT_RAW
    assert first_claim.to_json() == _CA77A25_FIRST_CLAIM_RAW
    assert first_result.to_json() == _CA77A25_FIRST_RESULT_RAW


def test_attempt_two_snapshot_requires_exact_terminal_vector_failure() -> None:
    checkpoint, first_claim, first_result = _legacy_records()
    claim = _attempt_two_claim()
    snapshot = IngestionRetryAttemptTwoSnapshot(checkpoint, first_claim, first_result, claim, None)
    assert snapshot.claim == claim
    assert snapshot.result is None

    wrong_stage = replace(first_result, failure_stage=IngestionFailureStage.EMBEDDING)
    with pytest.raises(ValueError, match="eligible"):
        IngestionRetryAttemptTwoSnapshot(checkpoint, first_claim, wrong_stage, claim, None)
    with pytest.raises(ValueError, match="legacy"):
        IngestionRetryAttemptTwoSnapshot(
            checkpoint,
            first_claim,
            None,  # type: ignore[arg-type]
            claim,
            None,
        )
    with pytest.raises(ValueError, match="attempt"):
        IngestionRetryAttemptTwoSnapshot(
            checkpoint,
            first_claim,
            first_result,
            claim,
            replace(_attempt_two_result(), attempt_digest=_C),
        )


@pytest.mark.asyncio
async def test_attempt_two_snapshot_loads_all_records_without_rewriting_legacy_bytes() -> None:
    checkpoint, first_claim, first_result = _legacy_records()
    claim = _attempt_two_claim()
    terminal = _attempt_two_result()
    legacy_raw = (
        checkpoint.to_json(),
        first_claim.to_json(),
        first_result.to_json(),
    )
    commands: list[list[Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(
            200,
            json={"result": [*legacy_raw, claim.to_json(), terminal.to_json()]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        snapshot = await control.load_retry_attempt_two_snapshot(
            job_digest=_D, deadline=_deadline()
        )

    assert snapshot.checkpoint.to_json() == legacy_raw[0]
    assert snapshot.first_claim.to_json() == legacy_raw[1]
    assert snapshot.first_result.to_json() == legacy_raw[2]
    assert snapshot.claim == claim
    assert snapshot.result == terminal
    assert commands == [
        [
            "MGET",
            f"rag:v1:checkpoint:{_D}",
            f"rag:v1:checkpoint-retry:{_D}",
            f"rag:v1:checkpoint-retry-result:{_D}",
            f"rag:v1:checkpoint-retry-attempt-2:{_D}",
            f"rag:v1:checkpoint-retry-attempt-2-result:{_D}",
        ]
    ]


@pytest.mark.asyncio
async def test_attempt_two_snapshot_rejects_records_for_a_different_requested_job() -> None:
    checkpoint, first_claim, first_result = _legacy_records()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": [
                    checkpoint.to_json(),
                    first_claim.to_json(),
                    first_result.to_json(),
                    None,
                    None,
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        with pytest.raises(ResearchControlUnavailableError):
            await control.load_retry_attempt_two_snapshot(
                job_digest=_C,
                deadline=_deadline(),
            )


@pytest.mark.asyncio
async def test_attempt_two_claim_is_atomic_and_concurrent_claim_loses() -> None:
    checkpoint, first_claim, first_result = _legacy_records()
    claim = _attempt_two_claim()
    commands: list[list[Any]] = []
    replies: Any = iter([1, 0])

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert await control.claim_failed_retry_attempt_two(
            checkpoint=checkpoint,
            first_claim=first_claim,
            first_result=first_result,
            claim=claim,
            deadline=_deadline(),
        )
        assert not await control.claim_failed_retry_attempt_two(
            checkpoint=checkpoint,
            first_claim=first_claim,
            first_result=first_result,
            claim=claim,
            deadline=_deadline(),
        )

    expected = [
        "EVAL",
        _CLAIM_RETRY_ATTEMPT_TWO_SCRIPT,
        5,
        f"rag:v1:checkpoint:{_D}",
        f"rag:v1:checkpoint-retry:{_D}",
        f"rag:v1:checkpoint-retry-result:{_D}",
        f"rag:v1:checkpoint-retry-attempt-2:{_D}",
        f"rag:v1:checkpoint-retry-attempt-2-result:{_D}",
        checkpoint.to_json(),
        first_claim.to_json(),
        first_result.to_json(),
        claim.to_json(),
    ]
    assert commands == [expected, expected]
    assert "redis.call('set',KEYS[4],ARGV[4])" in _CLAIM_RETRY_ATTEMPT_TWO_SCRIPT
    for legacy_key in (1, 2, 3):
        assert f"redis.call('set',KEYS[{legacy_key}]" not in _CLAIM_RETRY_ATTEMPT_TWO_SCRIPT
        assert f"redis.call('del',KEYS[{legacy_key}]" not in _CLAIM_RETRY_ATTEMPT_TWO_SCRIPT
        assert f"redis.call('expire',KEYS[{legacy_key}]" not in _CLAIM_RETRY_ATTEMPT_TWO_SCRIPT


@pytest.mark.asyncio
@pytest.mark.parametrize("cas_reply", [-1, -2, -3, 0])
async def test_attempt_two_stale_ambiguous_or_existing_claim_fails_closed(
    cas_reply: int,
) -> None:
    checkpoint, first_claim, first_result = _legacy_records()
    claim = _attempt_two_claim()
    commands: list[list[Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": cas_reply})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert not await control.claim_failed_retry_attempt_two(
            checkpoint=checkpoint,
            first_claim=first_claim,
            first_result=first_result,
            claim=claim,
            deadline=_deadline(),
        )

    assert len(commands) == 1
    assert all(command != "DEL" for command in commands[0])


@pytest.mark.asyncio
async def test_attempt_two_ineligible_lineage_is_rejected_before_redis() -> None:
    checkpoint, first_claim, first_result = _legacy_records()
    ineligible = replace(first_result, failure_stage=IngestionFailureStage.EMBEDDING)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected Redis request: {request!r}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        with pytest.raises(ValueError, match="eligible"):
            await control.claim_failed_retry_attempt_two(
                checkpoint=checkpoint,
                first_claim=first_claim,
                first_result=ineligible,
                claim=_attempt_two_claim(),
                deadline=_deadline(),
            )


@pytest.mark.asyncio
async def test_attempt_two_terminal_success_failure_and_exact_repeat_are_immutable() -> None:
    checkpoint, first_claim, first_result = _legacy_records()
    claim = _attempt_two_claim()
    succeeded = _attempt_two_result()
    failed = _attempt_two_result(
        state=IngestionRetryState.FAILED,
        stage=IngestionFailureStage.VECTOR_VERIFICATION,
    )
    commands: list[list[Any]] = []
    replies: Any = iter([1, 1, -2])

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(_decode_request(request))
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        assert await control.finish_failed_retry_attempt_two(
            checkpoint=checkpoint,
            first_claim=first_claim,
            first_result=first_result,
            claim=claim,
            result=succeeded,
            deadline=_deadline(),
        )
        assert await control.finish_failed_retry_attempt_two(
            checkpoint=checkpoint,
            first_claim=first_claim,
            first_result=first_result,
            claim=claim,
            result=succeeded,
            deadline=_deadline(),
        )
        assert not await control.finish_failed_retry_attempt_two(
            checkpoint=checkpoint,
            first_claim=first_claim,
            first_result=first_result,
            claim=claim,
            result=failed,
            deadline=_deadline(),
        )

    prefix = [
        "EVAL",
        _FINISH_RETRY_ATTEMPT_TWO_SCRIPT,
        5,
        f"rag:v1:checkpoint:{_D}",
        f"rag:v1:checkpoint-retry:{_D}",
        f"rag:v1:checkpoint-retry-result:{_D}",
        f"rag:v1:checkpoint-retry-attempt-2:{_D}",
        f"rag:v1:checkpoint-retry-attempt-2-result:{_D}",
        checkpoint.to_json(),
        first_claim.to_json(),
        first_result.to_json(),
        claim.to_json(),
    ]
    assert commands == [
        [*prefix, succeeded.to_json()],
        [*prefix, succeeded.to_json()],
        [*prefix, failed.to_json()],
    ]
    assert "redis.call('set',KEYS[5],ARGV[5])" in _FINISH_RETRY_ATTEMPT_TWO_SCRIPT
    for legacy_key in (1, 2, 3):
        assert f"redis.call('set',KEYS[{legacy_key}]" not in _FINISH_RETRY_ATTEMPT_TWO_SCRIPT
        assert f"redis.call('del',KEYS[{legacy_key}]" not in _FINISH_RETRY_ATTEMPT_TWO_SCRIPT
        assert f"redis.call('expire',KEYS[{legacy_key}]" not in _FINISH_RETRY_ATTEMPT_TWO_SCRIPT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        [],
        [None, None, None, None, None],
        [
            _legacy_records()[0].to_json(),
            _legacy_records()[1].to_json(),
            _legacy_records()[2].to_json(),
            None,
            _attempt_two_result().to_json(),
        ],
        [
            _legacy_records()[0].to_json(),
            _legacy_records()[1].to_json(),
            _legacy_records()[2].to_json(),
            "not-json",
            None,
        ],
    ],
)
async def test_attempt_two_malformed_or_partial_ledger_fails_closed(raw: list[Any]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": raw})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        control = RedisResearchControl("https://redis.example", SecretStr("token"), client=client)
        with pytest.raises(ResearchControlUnavailableError) as caught:
            await control.load_retry_attempt_two_snapshot(job_digest=_D, deadline=_deadline())

    assert str(caught.value) == "research control plane is unavailable"
    assert caught.value.__cause__ is None
