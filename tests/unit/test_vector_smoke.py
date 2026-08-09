from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass, field

import httpx
import pytest
from pydantic import SecretStr

from app import vector_smoke
from app.providers import upstash_vector_smoke
from app.providers.upstash_vector_smoke import (
    SMOKE_DIMENSIONS,
    SMOKE_NAMESPACE,
    VectorSmokeProbe,
    VectorSmokeReason,
    VectorSmokeResult,
)
from app.research.deadline import RequestDeadline
from app.vector_smoke import EXIT_CONFIG, EXIT_FAILURE, EXIT_SUCCESS, async_main


def probe(
    handler: httpx.MockTransport,
    *,
    poll_attempts: int = 3,
) -> tuple[VectorSmokeProbe, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=handler, trust_env=False)
    return (
        VectorSmokeProbe(
            "https://research-vector.upstash.io",
            SecretStr("private-vector-token"),
            client=client,
            poll_attempts=poll_attempts,
            poll_delay_seconds=0,
            point_id_factory=lambda: "smoke-" + "a" * 32,
        ),
        client,
    )


@pytest.mark.parametrize(
    ("url", "token", "poll_attempts", "poll_delay"),
    [
        ("http://research-vector.upstash.io", "secret", 3, 0),
        ("https://evil.example", "secret", 3, 0),
        ("https://research-vector.upstash.io?token=private", "secret", 3, 0),
        ("https://research-vector.upstash.io", "", 3, 0),
        ("https://research-vector.upstash.io", "secret", 0, 0),
        ("https://research-vector.upstash.io", "secret", 3, float("nan")),
    ],
)
def test_smoke_rejects_unsafe_configuration_without_echoing_values(
    url: str,
    token: str,
    poll_attempts: int,
    poll_delay: float,
) -> None:
    with pytest.raises(ValueError, match=r"invalid|required") as captured:
        VectorSmokeProbe(
            url,
            SecretStr(token),
            poll_attempts=poll_attempts,
            poll_delay_seconds=poll_delay,
        )

    rendered = str(captured.value)
    assert "private" not in rendered
    if token:
        assert token not in rendered


def test_smoke_configuration_discards_traceback_locals() -> None:
    sentinel = "private-invalid-smoke-endpoint"

    with pytest.raises(ValueError, match="endpoint is invalid") as captured:
        VectorSmokeProbe(
            f"https://research-vector.upstash.io?token={sentinel}",
            SecretStr("private-vector-token"),
        )

    traceback = captured.value.__traceback__
    rendered = ""
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("upstash_vector_smoke.py"):
            rendered += "".join(
                repr(value)
                for value in traceback.tb_frame.f_locals.values()
                if not inspect.iscoroutine(value)
            )
        traceback = traceback.tb_next
    assert sentinel not in rendered


@pytest.mark.asyncio
async def test_smoke_rejects_invalid_generated_identifier_before_network() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"result": "Success"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    adapter = VectorSmokeProbe(
        "https://research-vector.upstash.io",
        SecretStr("private-vector-token"),
        client=client,
        point_id_factory=lambda: "private-invalid-identifier",
    )
    try:
        with pytest.raises(ValueError, match="identifier is invalid") as captured:
            await adapter.run(deadline=RequestDeadline.after(1))
    finally:
        await client.aclose()

    assert calls == 0
    assert "private-invalid-identifier" not in str(captured.value)


@pytest.mark.asyncio
async def test_smoke_uses_one_synthetic_point_and_verifies_cleanup() -> None:
    calls: list[tuple[str, str, object]] = []
    state = "absent"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal state
        body = json.loads(request.content)
        calls.append((request.method, request.url.path, body))
        if request.url.path.startswith("/upsert/"):
            state = "present"
            return httpx.Response(200, json={"result": "Success"})
        if request.url.path.startswith("/delete/"):
            state = "absent"
            return httpx.Response(200, json={"result": {"deleted": 1}})
        point_id = body["ids"][0]
        result = (
            [{"id": point_id, "vector": [1.0] + [0.0] * (SMOKE_DIMENSIONS - 1)}]
            if state == "present"
            else [None]
        )
        return httpx.Response(200, json={"result": result})

    adapter, client = probe(httpx.MockTransport(handler))
    try:
        result = await adapter.run(deadline=RequestDeadline.after(1))
    finally:
        await client.aclose()

    assert result.passed is True
    assert result.reason is VectorSmokeReason.VERIFIED
    assert result.cleanup_verified is True
    assert result.expected_point_count == 1
    assert result.observed_point_count == 1
    assert result.null_point_count == 0
    assert [path for _, path, _ in calls] == [
        f"/upsert/{SMOKE_NAMESPACE}",
        f"/fetch/{SMOKE_NAMESPACE}",
        f"/delete/{SMOKE_NAMESPACE}",
        f"/fetch/{SMOKE_NAMESPACE}",
    ]
    upsert = calls[0][2]
    assert isinstance(upsert, list)
    assert len(upsert) == 1
    assert len(upsert[0]["vector"]) == 1536
    assert sum(value != 0 for value in upsert[0]["vector"]) == 1
    assert calls[2][2] == {"ids": [upsert[0]["id"]]}
    assert all(SMOKE_NAMESPACE in path for _, path, _ in calls)


@pytest.mark.asyncio
async def test_smoke_polls_reads_but_never_retries_writes() -> None:
    counts = {"upsert": 0, "fetch": 0, "delete": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        operation = request.url.path.split("/")[1]
        counts[operation] += 1
        if operation == "upsert":
            return httpx.Response(200, json={"result": "Success"})
        if operation == "delete":
            return httpx.Response(200, json={"result": {"deleted": 1}})
        point_id = json.loads(request.content)["ids"][0]
        if counts["delete"]:
            return httpx.Response(200, json={"result": [None]})
        if counts["fetch"] < 3:
            return httpx.Response(200, json={"result": [None]})
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": point_id,
                        "vector": [1.0] + [0.0] * (SMOKE_DIMENSIONS - 1),
                    }
                ]
            },
        )

    adapter, client = probe(httpx.MockTransport(handler), poll_attempts=3)
    try:
        result = await adapter.run(deadline=RequestDeadline.after(1))
    finally:
        await client.aclose()

    assert result.passed is True
    assert counts == {"upsert": 1, "fetch": 4, "delete": 1}


@pytest.mark.asyncio
async def test_smoke_reports_persistently_missing_point_after_bounded_polling() -> None:
    counts = {"upsert": 0, "fetch": 0, "delete": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        operation = request.url.path.split("/")[1]
        counts[operation] += 1
        if operation == "upsert":
            return httpx.Response(200, json={"result": "Success"})
        if operation == "delete":
            return httpx.Response(200, json={"result": {"deleted": 0}})
        return httpx.Response(200, json={"result": [None]})

    adapter, client = probe(httpx.MockTransport(handler), poll_attempts=3)
    try:
        result = await adapter.run(deadline=RequestDeadline.after(1))
    finally:
        await client.aclose()

    assert result.reason is VectorSmokeReason.MISSING_POINT
    assert result.observed_point_count == 0
    assert result.null_point_count == 1
    assert result.cleanup_verified is True
    assert counts == {"upsert": 1, "fetch": 4, "delete": 1}


@pytest.mark.asyncio
async def test_smoke_uses_one_absolute_deadline_and_reports_cleanup_failure_safely() -> None:
    now = [10.0]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        now[0] = 12.0
        return httpx.Response(200, json={"result": "Success"})

    adapter, client = probe(httpx.MockTransport(handler))
    try:
        result = await adapter.run(
            deadline=RequestDeadline.after(1, clock=lambda: now[0]),
        )
    finally:
        await client.aclose()

    assert result.passed is False
    assert result.reason is VectorSmokeReason.CLEANUP_UNVERIFIED
    assert result.cleanup_verified is False
    assert len(calls) == 1
    assert "private-vector-token" not in repr(result)


@pytest.mark.asyncio
async def test_provider_failure_is_sanitized_and_write_attempts_are_not_retried() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        operation = request.url.path.split("/")[1]
        calls.append(operation)
        if operation == "upsert":
            return httpx.Response(503, json={"error": "private-provider-response"})
        if operation == "delete":
            return httpx.Response(200, json={"result": {"deleted": 0}})
        return httpx.Response(200, json={"result": [None]})

    adapter, client = probe(httpx.MockTransport(handler))
    try:
        result = await adapter.run(deadline=RequestDeadline.after(1))
    finally:
        await client.aclose()

    assert result.passed is False
    assert result.reason is VectorSmokeReason.PROVIDER_FAILURE
    assert result.cleanup_verified is True
    assert calls == ["upsert", "delete", "fetch"]
    assert "private-provider-response" not in repr(result)
    assert "private-vector-token" not in repr(result)


@pytest.mark.asyncio
async def test_smoke_rejects_non_exact_write_success_shape() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        operation = request.url.path.split("/")[1]
        calls.append(operation)
        if operation == "upsert":
            return httpx.Response(
                200,
                json={"result": "Success", "private": "private-provider-response"},
            )
        if operation == "delete":
            return httpx.Response(200, json={"result": {"deleted": 0}})
        return httpx.Response(200, json={"result": [None]})

    adapter, client = probe(httpx.MockTransport(handler))
    try:
        result = await adapter.run(deadline=RequestDeadline.after(1))
    finally:
        await client.aclose()

    assert result.passed is False
    assert result.reason is VectorSmokeReason.PROVIDER_FAILURE
    assert result.cleanup_verified is True
    assert calls == ["upsert", "delete", "fetch"]
    assert "private-provider-response" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"unexpected": []}, VectorSmokeReason.RESPONSE_SHAPE),
        ({"result": []}, VectorSmokeReason.RESPONSE_SHAPE),
        (
            {"result": [{"id": "wrong", "vector": [1.0] + [0.0] * 1535}]},
            VectorSmokeReason.ID_MISMATCH,
        ),
        (
            {"result": [{"id": "smoke-" + "a" * 32, "vector": [0.0] * 1536}]},
            VectorSmokeReason.VECTOR_MISMATCH,
        ),
    ],
)
async def test_smoke_returns_fixed_sanitized_verification_reasons(
    payload: object,
    reason: VectorSmokeReason,
) -> None:
    deleted = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal deleted
        operation = request.url.path.split("/")[1]
        if operation == "upsert":
            return httpx.Response(200, json={"result": "Success"})
        if operation == "delete":
            deleted = True
            return httpx.Response(200, json={"result": {"deleted": 1}})
        return httpx.Response(200, json={"result": [None]} if deleted else payload)

    adapter, client = probe(httpx.MockTransport(handler))
    try:
        result = await adapter.run(deadline=RequestDeadline.after(1))
    finally:
        await client.aclose()

    assert result.reason is reason
    assert result.cleanup_verified is True
    assert "wrong" not in repr(result)


@dataclass(frozen=True)
class _Settings:
    upstash_vector_rest_url: str | None = "https://research-vector.upstash.io"
    upstash_vector_rest_token: SecretStr | None = field(
        default_factory=lambda: SecretStr("private-vector-token")
    )
    openai_api_key: SecretStr | None = field(
        default_factory=lambda: SecretStr("private-openai-key")
    )


@pytest.mark.asyncio
async def test_cli_defaults_to_dry_run_without_constructing_probe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    constructed = False

    def factory(*_: object, **__: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("probe must not be constructed")

    exit_code = await async_main([], settings_factory=_Settings, probe_factory=factory)

    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_SUCCESS
    assert output["dry_run"] is True
    assert output["applied"] is False
    assert output["network_calls"] == 0
    assert output["openai_project_confirmation"] == "not_independently_verified"
    assert "private" not in json.dumps(output)
    assert constructed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("flags", [["--apply"], ["--acknowledge-live-vector-smoke"]])
async def test_cli_requires_both_live_flags_without_network(
    flags: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = await async_main(
        flags,
        settings_factory=_Settings,
        probe_factory=lambda *_args, **_kwargs: pytest.fail("network factory constructed"),
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_CONFIG
    assert output == {
        "applied": False,
        "dry_run": True,
        "error_category": "authorization",
        "errors": ["live vector smoke requires both acknowledgements"],
        "network_calls": 0,
    }


@pytest.mark.asyncio
async def test_cli_live_path_reports_only_sanitized_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeProbe:
        async def run(self, *, deadline: RequestDeadline) -> VectorSmokeResult:
            assert isinstance(deadline, RequestDeadline)
            return VectorSmokeResult(
                passed=True,
                reason=VectorSmokeReason.VERIFIED,
                cleanup_verified=True,
                expected_point_count=1,
                observed_point_count=1,
                null_point_count=0,
                upsert_request_count=1,
                fetch_request_count=2,
                delete_request_count=1,
            )

        async def aclose(self) -> None:
            return None

    exit_code = await async_main(
        ["--apply", "--acknowledge-live-vector-smoke"],
        settings_factory=_Settings,
        probe_factory=lambda *_args, **_kwargs: FakeProbe(),
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_SUCCESS
    assert output["passed"] is True
    assert output["reason"] == "verified"
    assert output["cleanup_verified"] is True
    assert output["request_counts"] == {"delete": 1, "fetch": 2, "upsert": 1}
    assert "private" not in json.dumps(output)


@pytest.mark.asyncio
async def test_cli_sanitizes_provider_and_close_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingProbe:
        async def run(self, *, deadline: RequestDeadline) -> VectorSmokeResult:
            raise RuntimeError("private-provider-response")

        async def aclose(self) -> None:
            raise RuntimeError("private-close-response")

    exit_code = await async_main(
        ["--apply", "--acknowledge-live-vector-smoke"],
        settings_factory=_Settings,
        probe_factory=lambda *_args, **_kwargs: FailingProbe(),
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_FAILURE
    assert output["errors"] == ["vector smoke provider failure"]
    assert "private" not in json.dumps(output)


def test_smoke_modules_do_not_import_sec_openai_or_redis_providers() -> None:
    imported: set[str] = set()
    for module in (upstash_vector_smoke, vector_smoke):
        tree = ast.parse(inspect.getsource(module))
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )

    assert imported.isdisjoint(
        {
            "app.providers.openai_research",
            "app.providers.redis_research",
            "app.providers.sec_filings",
            "app.providers.sec_filing_parser",
        }
    )
