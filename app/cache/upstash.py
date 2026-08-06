from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx

from app.cache.base import CacheEntry
from app.errors import CacheUnavailableError

_COUNTER_SCRIPT = (
    "local n=redis.call('incr',KEYS[1]); "
    "if n==1 then redis.call('expire',KEYS[1],ARGV[1]); end; return n"
)
_QUOTA_SCRIPT = (
    "local n=tonumber(redis.call('get',KEYS[1]) or '0'); "
    "local limit=tonumber(ARGV[1]); if n>=limit then return -1 end; "
    "n=redis.call('incr',KEYS[1]); if n==1 then redis.call('expire',KEYS[1],ARGV[2]); end; return n"
)
_RELEASE_SCRIPT = (
    "if redis.call('get',KEYS[1]) == ARGV[1] "
    "then return redis.call('del',KEYS[1]) else return 0 end"
)


class UpstashCache:
    def __init__(
        self,
        url: str,
        token: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not url or not token:
            raise ValueError("Upstash REST URL and token are required")
        self._url = url.rstrip("/")
        self._clock = clock
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _command(self, command: list[Any]) -> Any:
        try:
            response = await self._client.post(self._url, headers=self._headers, json=command)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise CacheUnavailableError("cache service is unavailable") from exc
        if not isinstance(body, dict) or body.get("error") is not None:
            raise CacheUnavailableError("cache service rejected the command")
        return body.get("result")

    async def get(self, key: str) -> CacheEntry[Any] | None:
        raw = await self._command(["GET", key])
        if raw is None:
            return None
        try:
            envelope = json.loads(raw)
            now = self._clock()
            if float(envelope["stale_until"]) <= now:
                return None
            return CacheEntry(
                value=envelope["value"], is_fresh=float(envelope["fresh_until"]) > now
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise CacheUnavailableError("cache contained an invalid entry") from exc

    async def set(self, key: str, value: Any, *, ttl_seconds: int, stale_seconds: int = 0) -> None:
        now = self._clock()
        envelope = json.dumps(
            {
                "value": value,
                "fresh_until": now + ttl_seconds,
                "stale_until": now + ttl_seconds + stale_seconds,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        await self._command(["SET", key, envelope, "EX", ttl_seconds + stale_seconds])

    async def reserve_quota(
        self, key: str, *, limit: int = 90, window_seconds: int = 86400
    ) -> int | None:
        count = int(await self._command(["EVAL", _QUOTA_SCRIPT, 1, key, limit, window_seconds]))
        return count if count >= 0 else None

    async def current_count(self, key: str) -> int:
        value = await self._command(["GET", key])
        return int(value) if value is not None else 0

    async def increment_rate(self, key: str, *, window_seconds: int) -> int:
        return int(await self._command(["EVAL", _COUNTER_SCRIPT, 1, key, window_seconds]))

    async def increment(self, key: str, ttl: int) -> int:
        return await self.increment_rate(key, window_seconds=ttl)

    async def acquire_lock(self, key: str, token: str, *, ttl_seconds: int) -> bool:
        result = await self._command(["SET", key, token, "NX", "EX", ttl_seconds])
        return bool(result == "OK")

    async def release_lock(self, key: str, token: str) -> bool:
        return bool(await self._command(["EVAL", _RELEASE_SCRIPT, 1, key, token]))
