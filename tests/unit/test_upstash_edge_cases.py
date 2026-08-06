from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.cache.upstash import UpstashCache
from app.errors import CacheUnavailableError


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
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectError("private upstream details", request=request)
        if failure == "invalid-json":
            return httpx.Response(200, content=b"not-json")
        if failure == "redis-error":
            return httpx.Response(200, json={"error": "token leaked"})
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = UpstashCache("https://cache.example", "token", client=client)
        with pytest.raises(CacheUnavailableError) as caught:
            await cache.current_count("quota")

    assert "private" not in str(caught.value)
    assert "token leaked" not in str(caught.value)


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
    cache = UpstashCache("https://cache.example", "token")
    await cache.aclose()
    assert cache._client.is_closed

    external = httpx.AsyncClient()
    injected = UpstashCache("https://cache.example", "token", client=external)
    await injected.aclose()
    assert not external.is_closed
    await external.aclose()
