from __future__ import annotations

import json
import traceback
from typing import Any

import httpx
import pytest

from app.cache.upstash import UpstashCache
from app.errors import CacheUnavailableError


def _assert_exception_graph_is_secret_free(
    error: BaseException,
    *secrets: str,
) -> None:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered = "".join(traceback.format_exception(current))
        for secret in secrets:
            assert secret not in str(current)
            assert secret not in repr(current)
            assert secret not in rendered
            for frame, _ in traceback.walk_tb(current.__traceback__):
                if frame.f_code.co_filename.endswith("upstash.py"):
                    assert secret not in repr(frame.f_locals)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


@pytest.mark.asyncio
async def test_upstash_rejects_missing_credentials() -> None:
    with pytest.raises(ValueError, match="URL and token"):
        UpstashCache("", "token")
    with pytest.raises(ValueError, match="URL and token"):
        UpstashCache("https://cache.example", "")


@pytest.mark.asyncio
async def test_upstash_decodes_fresh_stale_and_expired_entries() -> None:
    now = [1_000.0]
    envelope = json.dumps({"value": {"symbol": "MSFT"}, "fresh_until": 1_010, "stale_until": 1_020})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer token"
        return httpx.Response(200, json={"result": envelope})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example/", "token", client=client, clock=lambda: now[0])
        fresh = await cache.get("quote")
        assert fresh is not None
        assert fresh.is_fresh
        now[0] = 1_011
        stale = await cache.get("quote")
        assert stale is not None
        assert stale.is_stale
        now[0] = 1_020
        assert await cache.get("quote") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"result": "not-json"},
        {"result": json.dumps({"value": 1, "fresh_until": "bad", "stale_until": 2_000})},
        {"result": json.dumps({"value": 1, "fresh_until": 2_000})},
    ],
)
async def test_upstash_rejects_corrupt_cache_entries(body: dict[str, Any]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", "token", client=client, clock=lambda: 1_000)
        with pytest.raises(CacheUnavailableError, match="invalid entry"):
            await cache.get("quote")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["network", "invalid-json", "redis-error", "invalid-body"])
async def test_upstash_sanitizes_transport_and_protocol_failures(failure: str) -> None:
    provider_sentinel = "cache-transport-secret-sentinel"

    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectError("private upstream details", request=request)
        if failure == "invalid-json":
            return httpx.Response(200, content=b"not-json")
        if failure == "redis-error":
            return httpx.Response(
                200,
                json={"error": f"private provider body {provider_sentinel}"},
            )
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", provider_sentinel, client=client)
        with pytest.raises(CacheUnavailableError) as caught:
            await cache.current_count("quota")

    rendered = "".join(traceback.format_exception(caught.value))
    for private_value in (
        provider_sentinel,
        "private upstream details",
        "private provider body",
    ):
        assert private_value not in str(caught.value)
        assert private_value not in repr(caught.value)
        assert private_value not in rendered
        for frame, _ in traceback.walk_tb(caught.value.__traceback__):
            if frame.f_code.co_filename.endswith("upstash.py"):
                assert private_value not in repr(frame.f_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_upstash_counter_and_lock_result_semantics() -> None:
    replies: Any = iter([-1, None, "7", "3", "4", None, 0])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": next(replies)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", "token", client=client)
        assert await cache.reserve_quota("quota", limit=1, window_seconds=60) is None
        assert await cache.current_count("missing") == 0
        assert await cache.current_count("quota") == 7
        assert await cache.increment_rate("rate", window_seconds=60) == 3
        assert await cache.increment("rate", ttl=60) == 4
        assert not await cache.acquire_lock("lock", "owner", ttl_seconds=5)
        assert not await cache.release_lock("lock", "owner")


@pytest.mark.asyncio
async def test_upstash_closes_only_the_client_it_owns() -> None:
    cache = UpstashCache("https://cache-name.upstash.io", "token")
    await cache.aclose()
    assert cache._client.is_closed

    external = httpx.AsyncClient()
    injected = UpstashCache("https://cache-name.upstash.io", "token", client=external)
    await injected.aclose()
    assert not external.is_closed
    await external.aclose()


@pytest.mark.asyncio
async def test_upstash_owned_client_disables_environment_and_redirects(
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

    cache = UpstashCache(" https://CACHE-NAME.upstash.io/ ", "token")

    assert cache._url == "https://cache-name.upstash.io"
    assert client_kwargs["trust_env"] is False
    assert client_kwargs["follow_redirects"] is False
    await cache.aclose()
    assert cache._client.is_closed


@pytest.mark.parametrize(
    "url",
    [
        "http://cache-name.upstash.io",
        "https://upstash.io",
        "https://cache-name.upstash.io.evil.example",
        "https://user@cache-name.upstash.io",
        "https://cache-name.upstash.io:443",
        "https://cache-name.upstash.io/commands",
        "https://cache-name.upstash.io?redirect=evil",
        "https://cache-name.upstash.io#fragment",
    ],
)
def test_upstash_rejects_unsafe_owned_endpoints_before_client_construction(
    url: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    constructed = False
    provider_sentinel = "cache-bearer-secret-sentinel"

    def fail_if_constructed(**_: object) -> None:
        nonlocal constructed
        constructed = True

    monkeypatch.setattr(httpx, "AsyncClient", fail_if_constructed)

    with pytest.raises(ValueError, match="Upstash REST URL") as caught:
        UpstashCache(url, provider_sentinel)

    assert not constructed
    surfaces = (str(caught.value), repr(caught.value), caplog.text)
    assert all(provider_sentinel not in surface for surface in surfaces)
    for frame, _ in traceback.walk_tb(caught.value.__traceback__):
        if frame.f_code.co_filename.endswith("upstash.py"):
            assert provider_sentinel not in repr(frame.f_locals)


@pytest.mark.asyncio
async def test_upstash_injected_client_allows_only_safe_example_root_and_never_redirects() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={"location": "https://attacker.example/collect"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        cache = UpstashCache(" https://CACHE.example/ ", "redirect-secret", client=client)
        assert cache._url == "https://cache.example"
        assert "redirect-secret" not in repr(cache)
        with pytest.raises(CacheUnavailableError, match="unavailable"):
            await cache.current_count("quota")

    assert len(requests) == 1
    assert requests[0].url.host == "cache.example"


@pytest.mark.asyncio
async def test_upstash_normal_injected_client_cannot_enable_safe_test_endpoint() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="Upstash REST URL"):
            UpstashCache("https://cache.example", "token", client=client)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8079",
        "https://cache.invalid",
        "https://cache.example.evil.test",
        "https://user@cache.example",
        "https://cache.example:443",
        "https://cache.example/path",
        "https://cache.example?query=value",
        "https://cache.example#fragment",
    ],
)
@pytest.mark.asyncio
async def test_upstash_injected_client_rejects_urls_outside_safe_test_policy(url: str) -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"result": None})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="Upstash REST URL"):
            UpstashCache(url, "token", client=client)

    assert requests == 0


@pytest.mark.parametrize(
    "operation",
    ["reserve_quota", "release_quota", "current_count", "increment_rate"],
)
@pytest.mark.asyncio
async def test_upstash_numeric_operations_reject_private_malformed_results_without_leaking(
    operation: str,
) -> None:
    provider_sentinel = "numeric-bearer-token-sentinel"
    provider_detail = "private-malformed-numeric-result"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"result": f"12 {provider_detail} {provider_sentinel}"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", provider_sentinel, client=client)

        async def invoke() -> object:
            if operation == "reserve_quota":
                return await cache.reserve_quota("quota")
            if operation == "release_quota":
                return await cache.release_quota("quota")
            if operation == "current_count":
                return await cache.current_count("quota")
            return await cache.increment_rate("rate", window_seconds=60)

        with pytest.raises(CacheUnavailableError) as caught:
            await invoke()

    _assert_exception_graph_is_secret_free(caught.value, provider_sentinel, provider_detail)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("result", [True, False, "01", "+1", " 1", "1 ", 1.0])
@pytest.mark.asyncio
async def test_upstash_numeric_operations_reject_noncanonical_wire_shapes(result: object) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": result})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", "token", client=client)
        with pytest.raises(CacheUnavailableError, match="rejected") as caught:
            await cache.increment_rate("rate", window_seconds=60)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "result",
    ["private-lock-result-sentinel", "1", True, 2, None],
)
@pytest.mark.asyncio
async def test_upstash_release_lock_rejects_non_protocol_results_without_leaking(
    result: object,
) -> None:
    provider_sentinel = "lock-bearer-token-sentinel"
    owner_token = "lock-owner-token-sentinel"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": result})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", provider_sentinel, client=client)
        with pytest.raises(CacheUnavailableError, match="rejected") as caught:
            await cache.release_lock("lock", owner_token)

    _assert_exception_graph_is_secret_free(
        caught.value,
        provider_sentinel,
        owner_token,
        "private-lock-result-sentinel",
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("operation", ["acquire", "release"])
@pytest.mark.parametrize("failure", ["network", "non-2xx", "invalid-json", "redis-error"])
@pytest.mark.asyncio
async def test_upstash_lock_failures_scrub_caller_tokens_from_all_adapter_frames(
    operation: str,
    failure: str,
) -> None:
    bearer_token = "lock-failure-bearer-sentinel"
    owner_token = "lock-failure-owner-sentinel"
    provider_detail = "private-lock-provider-detail"

    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectError(f"{provider_detail} {owner_token}", request=request)
        if failure == "non-2xx":
            return httpx.Response(503, content=f"{provider_detail} {owner_token}".encode())
        if failure == "invalid-json":
            return httpx.Response(200, content=f"not-json {provider_detail} {owner_token}".encode())
        return httpx.Response(200, json={"error": f"{provider_detail} {owner_token}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", bearer_token, client=client)

        async def invoke() -> bool:
            if operation == "acquire":
                return await cache.acquire_lock("lock", owner_token, ttl_seconds=5)
            return await cache.release_lock("lock", owner_token)

        with pytest.raises(CacheUnavailableError) as caught:
            await invoke()

    _assert_exception_graph_is_secret_free(
        caught.value,
        bearer_token,
        owner_token,
        provider_detail,
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(("result", "expected"), [("OK", True), (None, False)])
@pytest.mark.asyncio
async def test_upstash_acquire_lock_accepts_only_documented_results(
    result: object,
    expected: bool,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": result})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", "token", client=client)
        assert await cache.acquire_lock("lock", "owner", ttl_seconds=5) is expected


@pytest.mark.parametrize(
    "result",
    ["private-acquire-result-sentinel", "ok", "", False, 0, 1, {}, []],
)
@pytest.mark.asyncio
async def test_upstash_acquire_lock_rejects_malformed_results_without_leaking(
    result: object,
) -> None:
    bearer_token = "acquire-bearer-token-sentinel"
    owner_token = "acquire-owner-token-sentinel"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": result})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", bearer_token, client=client)
        with pytest.raises(CacheUnavailableError, match="rejected") as caught:
            await cache.acquire_lock("lock", owner_token, ttl_seconds=5)

    _assert_exception_graph_is_secret_free(
        caught.value,
        bearer_token,
        owner_token,
        "private-acquire-result-sentinel",
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_upstash_command_scrubbing_never_mutates_caller_owned_payload() -> None:
    command: list[object] = ["SET", "lock", "caller-owned-token", "NX", "EX", 5]
    expected = list(command)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unavailable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", "token", client=client)
        with pytest.raises(CacheUnavailableError):
            await cache._command(command)

    assert command == expected


@pytest.mark.parametrize(
    "result",
    [
        "not-json private-cache-entry-sentinel",
        {"value": "private-cache-entry-sentinel"},
        json.dumps(["private-cache-entry-sentinel"]),
        json.dumps(
            {
                "value": "private-cache-entry-sentinel",
                "fresh_until": "invalid private-cache-entry-sentinel",
                "stale_until": 2_000,
            }
        ),
        json.dumps(
            {
                "value": "private-cache-entry-sentinel",
                "fresh_until": 1_500,
                "stale_until": {"private": "private-cache-entry-sentinel"},
            }
        ),
    ],
)
@pytest.mark.asyncio
async def test_upstash_invalid_entries_do_not_leak_cached_payload_from_decoder_frames(
    result: object,
) -> None:
    provider_sentinel = "cache-entry-bearer-token-sentinel"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": result})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache(
            "https://cache.example",
            provider_sentinel,
            client=client,
            clock=lambda: 1_000,
        )
        with pytest.raises(CacheUnavailableError, match="invalid entry") as caught:
            await cache.get("private-key")

    _assert_exception_graph_is_secret_free(
        caught.value,
        provider_sentinel,
        "private-cache-entry-sentinel",
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
