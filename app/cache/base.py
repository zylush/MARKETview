from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class CacheEntry[ValueT]:
    value: ValueT
    is_fresh: bool

    @property
    def is_stale(self) -> bool:
        return not self.is_fresh


class CacheKeyBuilder:
    def __init__(self, *, schema_version: str = "v1", prefix: str = "marketdata") -> None:
        self._schema_version = schema_version
        self._prefix = prefix

    def build(self, namespace: str, parameters: dict[str, Any]) -> str:
        canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        safe_namespace = "".join(
            char for char in namespace.lower() if char.isalnum() or char in "_-"
        )
        return f"{self._prefix}:{self._schema_version}:{safe_namespace}:{digest}"


class Cache(Protocol):
    async def get(self, key: str) -> CacheEntry[Any] | None: ...
    async def set(
        self, key: str, value: Any, *, ttl_seconds: int, stale_seconds: int = 0
    ) -> None: ...
    async def reserve_quota(
        self, key: str, *, limit: int = 90, window_seconds: int = 86400
    ) -> int | None: ...
    async def current_count(self, key: str) -> int: ...
    async def increment_rate(self, key: str, *, window_seconds: int) -> int: ...
    async def acquire_lock(self, key: str, token: str, *, ttl_seconds: int) -> bool: ...
    async def release_lock(self, key: str, token: str) -> bool: ...
