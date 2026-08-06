from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.cache.base import CacheEntry


@dataclass(frozen=True, slots=True)
class _StoredValue:
    value: Any
    fresh_until: float
    stale_until: float


@dataclass(frozen=True, slots=True)
class _Counter:
    value: int
    expires_at: float


@dataclass(frozen=True, slots=True)
class _LockValue:
    token: str
    expires_at: float


class MemoryCache:
    """Concurrency-safe async cache substitute for tests and single-process development."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._values: dict[str, _StoredValue] = {}
        self._counters: dict[str, _Counter] = {}
        self._locks: dict[str, _LockValue] = {}
        self._mutex = asyncio.Lock()

    async def get(self, key: str) -> CacheEntry[Any] | None:
        async with self._mutex:
            now = self._clock()
            stored = self._values.get(key)
            if stored is None:
                return None
            if stored.stale_until <= now:
                self._values = {name: value for name, value in self._values.items() if name != key}
                return None
            return CacheEntry(value=stored.value, is_fresh=stored.fresh_until > now)

    async def set(self, key: str, value: Any, *, ttl_seconds: int, stale_seconds: int = 0) -> None:
        if ttl_seconds <= 0 or stale_seconds < 0:
            raise ValueError("cache TTL values must be positive")
        async with self._mutex:
            now = self._clock()
            stored = _StoredValue(
                value=value,
                fresh_until=now + ttl_seconds,
                stale_until=now + ttl_seconds + stale_seconds,
            )
            self._values = {**self._values, key: stored}

    async def reserve_quota(
        self, key: str, *, limit: int = 90, window_seconds: int = 86400
    ) -> int | None:
        if window_seconds <= 0 or limit <= 0:
            raise ValueError("quota limit and window must be positive")
        async with self._mutex:
            now = self._clock()
            existing = self._counters.get(key)
            if existing and existing.expires_at > now and existing.value >= limit:
                return None
            updated = _Counter(
                value=(existing.value + 1 if existing and existing.expires_at > now else 1),
                expires_at=(
                    existing.expires_at
                    if existing and existing.expires_at > now
                    else now + window_seconds
                ),
            )
            self._counters = {**self._counters, key: updated}
            return updated.value

    async def increment_rate(self, key: str, *, window_seconds: int) -> int:
        return await self._increment(key, window_seconds)

    async def increment(self, key: str, ttl: int) -> int:
        return await self.increment_rate(key, window_seconds=ttl)

    async def _increment(self, key: str, window_seconds: int) -> int:
        if window_seconds <= 0:
            raise ValueError("counter window must be positive")
        async with self._mutex:
            now = self._clock()
            existing = self._counters.get(key)
            updated = _Counter(
                value=(existing.value + 1 if existing and existing.expires_at > now else 1),
                expires_at=(
                    existing.expires_at
                    if existing and existing.expires_at > now
                    else now + window_seconds
                ),
            )
            self._counters = {**self._counters, key: updated}
            return updated.value

    async def current_count(self, key: str) -> int:
        async with self._mutex:
            counter = self._counters.get(key)
            return counter.value if counter and counter.expires_at > self._clock() else 0

    async def acquire_lock(self, key: str, token: str, *, ttl_seconds: int) -> bool:
        async with self._mutex:
            now = self._clock()
            existing = self._locks.get(key)
            if existing and existing.expires_at > now:
                return False
            self._locks = {
                **self._locks,
                key: _LockValue(token=token, expires_at=now + ttl_seconds),
            }
            return True

    async def release_lock(self, key: str, token: str) -> bool:
        async with self._mutex:
            existing = self._locks.get(key)
            if existing is None or existing.token != token:
                return False
            self._locks = {name: value for name, value in self._locks.items() if name != key}
            return True
