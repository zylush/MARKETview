import asyncio
from typing import Any

import httpx

from app.cache.base import CacheKeyBuilder
from app.cache.memory import MemoryCache
from app.cache.upstash import UpstashCache


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_cache_keys_are_stable_versioned_and_do_not_expose_inputs() -> None:
    builder = CacheKeyBuilder(schema_version="v3")
    first = builder.build("history", {"symbol": "MSFT", "limit": 10})
    second = builder.build("history", {"limit": 10, "symbol": "MSFT"})

    assert first == second
    assert first.startswith("marketdata:v3:history:")
    assert "MSFT" not in first


def test_memory_cache_supports_stale_entries_quota_and_locks() -> None:
    async def scenario() -> None:
        now = [1000.0]
        cache = MemoryCache(clock=lambda: now[0])
        await cache.set("answer", {"value": 42}, ttl_seconds=10, stale_seconds=20)
        assert (await cache.get("answer")).is_fresh
        now[0] = 1011
        stale = await cache.get("answer")
        assert stale is not None
        assert stale.is_stale
        assert stale.value == {"value": 42}
        now[0] = 1031
        assert await cache.get("answer") is None

        assert await cache.reserve_quota("quota", limit=2, window_seconds=60) == 1
        assert await cache.reserve_quota("quota", limit=2, window_seconds=60) == 2
        assert await cache.reserve_quota("quota", limit=2, window_seconds=60) is None
        assert await cache.current_count("quota") == 2
        assert await cache.increment_rate("rate", window_seconds=60) == 1
        assert await cache.acquire_lock("lock", "token-a", ttl_seconds=10)
        assert not await cache.acquire_lock("lock", "token-b", ttl_seconds=10)
        assert not await cache.release_lock("lock", "token-b")
        assert await cache.release_lock("lock", "token-a")

    run(scenario())


def test_memory_quota_window_resets_without_counting_rejected_reservations() -> None:
    async def scenario() -> None:
        now = [1000.0]
        cache = MemoryCache(clock=lambda: now[0])
        assert await cache.reserve_quota("monthly", limit=1, window_seconds=50) == 1
        assert await cache.reserve_quota("monthly", limit=1, window_seconds=50) is None
        assert await cache.current_count("monthly") == 1
        now[0] += 51
        assert await cache.reserve_quota("monthly", limit=1, window_seconds=50) == 1

    run(scenario())


def test_upstash_uses_redis_rest_commands_and_token_safe_release() -> None:
    commands: list[list[Any]] = []
    replies = iter([None, "OK", 1, "OK", 1])

    def handler(request: httpx.Request) -> httpx.Response:
        commands.append(__import__("json").loads(request.content))
        return httpx.Response(200, json={"result": next(replies)})

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cache = UpstashCache("https://cache.example", "token", client=client)
            assert await cache.get("missing") is None
            await cache.set("k", {"v": 1}, ttl_seconds=10, stale_seconds=20)
            assert await cache.reserve_quota("q", limit=90, window_seconds=60) == 1
            assert await cache.acquire_lock("l", "owner", ttl_seconds=5)
            assert await cache.release_lock("l", "owner")
        finally:
            await client.aclose()

    run(scenario())

    assert commands[0] == ["GET", "missing"]
    assert commands[1][0:2] == ["SET", "k"]
    assert commands[1][-2:] == ["EX", 30]
    assert commands[2][0] == "EVAL"
    assert commands[3] == ["SET", "l", "owner", "NX", "EX", 5]
    assert commands[4][0] == "EVAL"
    assert "redis.call('get'" in commands[4][1]
