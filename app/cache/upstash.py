from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

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
_QUOTA_ROLLBACK_SCRIPT = (
    "local n=tonumber(redis.call('get',KEYS[1]) or '0'); "
    "if n<=0 then redis.call('del',KEYS[1]); return 0 end; "
    "n=redis.call('decr',KEYS[1]); "
    "if n<=0 then redis.call('del',KEYS[1]); return 0 end; return n"
)
_RELEASE_SCRIPT = (
    "if redis.call('get',KEYS[1]) == ARGV[1] "
    "then return redis.call('del',KEYS[1]) else return 0 end"
)
_UPSTASH_REST_HOST = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+upstash\.io$")
_SAFE_TEST_HOST = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+example$")
_CANONICAL_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_MIN_REDIS_INTEGER = -(2**63)
_MAX_REDIS_INTEGER = 2**63 - 1


def _strict_integer_result(
    value: object,
    *,
    minimum: int,
    allow_none: bool = False,
) -> int:
    parsed: int | None = None
    invalid = False
    if value is None:
        if allow_none:
            parsed = 0
        else:
            invalid = True
    elif type(value) is int:
        parsed = value
    elif (
        type(value) is str and len(value) <= 20 and _CANONICAL_INTEGER.fullmatch(value) is not None
    ):
        parsed = int(value)
    else:
        invalid = True
    value = None
    if (
        invalid
        or parsed is None
        or parsed < minimum
        or not _MIN_REDIS_INTEGER <= parsed <= _MAX_REDIS_INTEGER
    ):
        raise CacheUnavailableError("cache service rejected the command") from None
    return parsed


def _strict_binary_result(value: object) -> bool:
    result: bool | None = None
    if type(value) is int and value in {0, 1}:
        result = value == 1
    value = None
    if result is None:
        raise CacheUnavailableError("cache service rejected the command") from None
    return result


def _strict_lock_acquire_result(value: object) -> bool:
    result: bool | None = None
    invalid = False
    if type(value) is str and value == "OK":
        result = True
    elif value is None:
        result = False
    else:
        invalid = True
    value = None
    if invalid or result is None:
        raise CacheUnavailableError("cache service rejected the command") from None
    return result


def _decode_cache_entry(
    value: Any,
    clock: Callable[[], float],
) -> tuple[CacheEntry[Any] | None, bool]:
    entry: CacheEntry[Any] | None = None
    envelope: Any = None
    invalid = False
    try:
        if value is not None:
            envelope = json.loads(value)
            now = clock()
            if float(envelope["stale_until"]) <= now:
                entry = None
            else:
                entry = CacheEntry(
                    value=envelope["value"],
                    is_fresh=float(envelope["fresh_until"]) > now,
                )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        invalid = True
    value = None
    envelope = None
    return entry, invalid


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
        if not isinstance(token, str) or not token:
            raise ValueError("Upstash REST URL and token are required")
        token_secret = SecretStr(token)
        token = ""
        if not isinstance(url, str) or not url:
            raise ValueError("Upstash REST URL and token are required")
        allow_test_endpoint = client is not None and isinstance(
            getattr(client, "_transport", None), httpx.MockTransport
        )
        self._url = self._validated_url(url, allow_test_endpoint=allow_test_endpoint)
        self._clock = clock
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            trust_env=False,
            follow_redirects=False,
        )
        self._owns_client = client is None
        self._token = token_secret

    def __repr__(self) -> str:
        return "UpstashCache(url=<redacted>, token=SecretStr('**********'))"

    @staticmethod
    def _validated_url(value: str, *, allow_test_endpoint: bool) -> str:
        normalized = value.strip()
        if not normalized or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError("Upstash REST URL must be an approved HTTPS root")
        try:
            parsed = urlsplit(normalized)
            port = parsed.port
        except ValueError:
            raise ValueError("Upstash REST URL must be an approved HTTPS root") from None
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        approved_host = _UPSTASH_REST_HOST.fullmatch(hostname) is not None
        safe_test_host = allow_test_endpoint and _SAFE_TEST_HOST.fullmatch(hostname) is not None
        if (
            parsed.scheme != "https"
            or not (approved_host or safe_test_host)
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Upstash REST URL must be an approved HTTPS root")
        return f"https://{hostname}"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _command(self, command: list[Any]) -> Any:
        owned_command = list(command)
        command = []
        body: Any = None
        request: httpx.Request | None = None
        response: httpx.Response | None = None
        failed = False
        try:
            request = self._client.build_request(
                "POST",
                self._url,
                headers={
                    "Authorization": f"Bearer {self._token.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json=owned_command,
            )
            owned_command = []
            response = await self._client.send(request, follow_redirects=False)
            request = None
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            command = []
            owned_command = []
            request = None
            response = None
            body = None
            failed = True
        if failed:
            raise CacheUnavailableError("cache service is unavailable") from None
        if not isinstance(body, dict) or body.get("error") is not None:
            command = []
            owned_command = []
            request = None
            response = None
            body = None
            raise CacheUnavailableError("cache service rejected the command") from None
        result = body.get("result")
        command = []
        owned_command = []
        request = None
        response = None
        body = None
        return result

    async def get(self, key: str) -> CacheEntry[Any] | None:
        entry, invalid = _decode_cache_entry(await self._command(["GET", key]), self._clock)
        if invalid:
            entry = None
            raise CacheUnavailableError("cache contained an invalid entry") from None
        return entry

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
        count = _strict_integer_result(
            await self._command(["EVAL", _QUOTA_SCRIPT, 1, key, limit, window_seconds]),
            minimum=-1,
        )
        if count == -1:
            return None
        if count == 0:
            raise CacheUnavailableError("cache service rejected the command") from None
        return count

    async def release_quota(self, key: str) -> int:
        return _strict_integer_result(
            await self._command(["EVAL", _QUOTA_ROLLBACK_SCRIPT, 1, key]),
            minimum=0,
        )

    async def current_count(self, key: str) -> int:
        return _strict_integer_result(
            await self._command(["GET", key]),
            minimum=0,
            allow_none=True,
        )

    async def increment_rate(self, key: str, *, window_seconds: int) -> int:
        return _strict_integer_result(
            await self._command(["EVAL", _COUNTER_SCRIPT, 1, key, window_seconds]),
            minimum=1,
        )

    async def increment(self, key: str, ttl: int) -> int:
        return await self.increment_rate(key, window_seconds=ttl)

    async def acquire_lock(self, key: str, token: str, *, ttl_seconds: int) -> bool:
        token_secret = SecretStr(token)
        token = ""
        return _strict_lock_acquire_result(
            await self._command(
                ["SET", key, token_secret.get_secret_value(), "NX", "EX", ttl_seconds]
            )
        )

    async def release_lock(self, key: str, token: str) -> bool:
        token_secret = SecretStr(token)
        token = ""
        return _strict_binary_result(
            await self._command(["EVAL", _RELEASE_SCRIPT, 1, key, token_secret.get_secret_value()])
        )
