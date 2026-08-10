from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from app.research.deadline import RequestDeadline

SMOKE_DIMENSIONS = 1536
SMOKE_NAMESPACE = "marketview-nonprod-smoke-v1"
_POINT_ID = re.compile(r"^smoke-[0-9a-f]{32}$")
_MAX_RESPONSE_BYTES = 262_144
_SYNTHETIC_VECTOR = (1.0,) + (0.0,) * (SMOKE_DIMENSIONS - 1)


class VectorSmokeReason(StrEnum):
    VERIFIED = "verified"
    MISSING_POINT = "missing_point"
    RESPONSE_SHAPE = "response_shape"
    ID_MISMATCH = "id_mismatch"
    VECTOR_MISMATCH = "vector_mismatch"
    PROVIDER_FAILURE = "provider_failure"
    TIMEOUT = "timeout"
    CLEANUP_UNVERIFIED = "cleanup_unverified"


@dataclass(frozen=True, slots=True)
class VectorSmokeResult:
    passed: bool
    reason: VectorSmokeReason
    cleanup_verified: bool
    expected_point_count: int
    observed_point_count: int
    null_point_count: int
    upsert_request_count: int
    fetch_request_count: int
    delete_request_count: int


@dataclass(frozen=True, slots=True)
class _WireOutcome:
    kind: Literal["ok", "timeout", "unavailable", "invalid"]
    payload: object | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _Observation:
    reason: VectorSmokeReason
    observed_count: int
    null_count: int


class VectorSmokeProbe:
    """A gated one-point data-plane probe isolated from the filing namespace."""

    def __init__(
        self,
        url: str,
        token: SecretStr,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        poll_attempts: int = 4,
        poll_delay_seconds: float = 0.05,
        point_id_factory: Callable[[], str] | None = None,
    ) -> None:
        try:
            normalized_url = self._validated_url(url)
        except ValueError as error:
            url = ""
            token = SecretStr("")
            error = ValueError(str(error))
            raise error from None
        if not isinstance(token, SecretStr) or not token.get_secret_value():
            raise ValueError("a vector smoke credential is required")
        if client is not None and transport is not None:
            raise ValueError("the vector smoke client configuration is invalid")
        if type(poll_attempts) is not int or not 1 <= poll_attempts <= 10:
            raise ValueError("the vector smoke poll limit is invalid")
        if (
            isinstance(poll_delay_seconds, bool)
            or not isinstance(poll_delay_seconds, (int, float))
            or not math.isfinite(float(poll_delay_seconds))
            or not 0 <= float(poll_delay_seconds) <= 1
        ):
            raise ValueError("the vector smoke poll delay is invalid")

        if point_id_factory is None:
            import secrets

            def random_point_id() -> str:
                return "smoke-" + secrets.token_hex(16)

            point_id_factory = random_point_id
        if not callable(point_id_factory):
            raise ValueError("the vector smoke identifier factory is invalid")

        self._url = normalized_url
        self._token = token
        self._poll_attempts = poll_attempts
        self._poll_delay_seconds = float(poll_delay_seconds)
        self._point_id_factory = point_id_factory
        self._client = client or httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(5.0),
            trust_env=False,
        )
        self._owns_client = client is None

    @staticmethod
    def _validated_url(value: object) -> str:
        if not isinstance(value, str):
            value = None
            raise ValueError("the vector smoke endpoint is invalid")
        try:
            parsed = urlsplit(value.strip().rstrip("/"))
            port = parsed.port
        except ValueError:
            value = None
            raise ValueError("the vector smoke endpoint is invalid") from None
        hostname = parsed.hostname
        if (
            parsed.scheme.lower() != "https"
            or not hostname
            or not hostname.lower().endswith(".upstash.io")
            or hostname.lower() == "upstash.io"
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            value = None
            parsed = urlsplit("https://redacted.upstash.io")
            hostname = None
            raise ValueError("the vector smoke endpoint is invalid")
        return f"https://{hostname.lower()}"

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def run(self, *, deadline: RequestDeadline) -> VectorSmokeResult:
        if not isinstance(deadline, RequestDeadline):
            raise ValueError("the vector smoke deadline is invalid")
        point_id = self._point_id_factory()
        if not isinstance(point_id, str) or _POINT_ID.fullmatch(point_id) is None:
            raise ValueError("the vector smoke identifier is invalid")

        primary = _Observation(VectorSmokeReason.PROVIDER_FAILURE, 0, 0)
        upsert_count = 0
        fetch_count = 0
        delete_count = 0
        cleanup_verified = False
        try:
            upsert_count = 1
            upsert = await self._send(
                "POST",
                "upsert",
                [{"id": point_id, "vector": list(_SYNTHETIC_VECTOR)}],
                deadline=deadline,
            )
            if upsert.kind == "ok" and self._write_succeeded(upsert.payload):
                primary, read_count = await self._poll_present(point_id, deadline=deadline)
                fetch_count += read_count
            elif upsert.kind == "timeout":
                primary = _Observation(VectorSmokeReason.TIMEOUT, 0, 0)
        finally:
            delete_count, deleted = await self._delete_once(point_id, deadline=deadline)
            if deleted:
                cleanup_verified, cleanup_reads = await self._poll_absent(
                    point_id,
                    deadline=deadline,
                )
                fetch_count += cleanup_reads

        reason = VectorSmokeReason.CLEANUP_UNVERIFIED if not cleanup_verified else primary.reason
        return VectorSmokeResult(
            passed=reason is VectorSmokeReason.VERIFIED and cleanup_verified,
            reason=reason,
            cleanup_verified=cleanup_verified,
            expected_point_count=1,
            observed_point_count=primary.observed_count,
            null_point_count=primary.null_count,
            upsert_request_count=upsert_count,
            fetch_request_count=fetch_count,
            delete_request_count=delete_count,
        )

    async def _delete_once(
        self,
        point_id: str,
        *,
        deadline: RequestDeadline,
    ) -> tuple[int, bool]:
        if deadline.remaining_seconds() <= 0:
            return 0, False
        outcome = await self._send(
            "DELETE",
            "delete",
            {"ids": [point_id]},
            deadline=deadline,
        )
        return 1, outcome.kind == "ok" and self._delete_succeeded(outcome.payload)

    async def _poll_present(
        self,
        point_id: str,
        *,
        deadline: RequestDeadline,
    ) -> tuple[_Observation, int]:
        last = _Observation(VectorSmokeReason.MISSING_POINT, 0, 1)
        requests = 0
        for attempt in range(self._poll_attempts):
            if attempt and not await self._wait_for_poll(deadline):
                return _Observation(VectorSmokeReason.TIMEOUT, 0, 0), requests
            outcome = await self._fetch(point_id, deadline=deadline)
            requests += 1
            if outcome.kind == "timeout":
                return _Observation(VectorSmokeReason.TIMEOUT, 0, 0), requests
            if outcome.kind != "ok":
                return _Observation(VectorSmokeReason.PROVIDER_FAILURE, 0, 0), requests
            last = self._decode_present(outcome.payload, point_id)
            if last.reason is not VectorSmokeReason.MISSING_POINT:
                return last, requests
        return last, requests

    async def _poll_absent(
        self,
        point_id: str,
        *,
        deadline: RequestDeadline,
    ) -> tuple[bool, int]:
        requests = 0
        for attempt in range(self._poll_attempts):
            if attempt and not await self._wait_for_poll(deadline):
                return False, requests
            outcome = await self._fetch(point_id, deadline=deadline)
            requests += 1
            if outcome.kind != "ok":
                return False, requests
            observation = self._decode_present(outcome.payload, point_id)
            if observation.reason is VectorSmokeReason.MISSING_POINT:
                return True, requests
            if observation.reason is not VectorSmokeReason.VERIFIED:
                return False, requests
        return False, requests

    async def _wait_for_poll(self, deadline: RequestDeadline) -> bool:
        remaining = deadline.remaining_seconds()
        if remaining <= 0:
            return False
        if self._poll_delay_seconds == 0:
            return True
        try:
            async with asyncio.timeout(remaining):
                await asyncio.sleep(min(self._poll_delay_seconds, remaining))
        except TimeoutError:
            return False
        return deadline.remaining_seconds() > 0

    async def _fetch(self, point_id: str, *, deadline: RequestDeadline) -> _WireOutcome:
        return await self._send(
            "POST",
            "fetch",
            {
                "ids": [point_id],
                "includeVectors": True,
                "includeMetadata": False,
                "includeData": False,
            },
            deadline=deadline,
        )

    @staticmethod
    def _decode_present(payload: object, point_id: str) -> _Observation:
        if not isinstance(payload, dict) or set(payload) != {"result"}:
            return _Observation(VectorSmokeReason.RESPONSE_SHAPE, 0, 0)
        result = payload.get("result")
        if not isinstance(result, list) or len(result) != 1:
            return _Observation(VectorSmokeReason.RESPONSE_SHAPE, 0, 0)
        record = result[0]
        if record is None:
            return _Observation(VectorSmokeReason.MISSING_POINT, 0, 1)
        if not isinstance(record, dict) or set(record) != {"id", "vector"}:
            return _Observation(VectorSmokeReason.RESPONSE_SHAPE, 0, 0)
        if record.get("id") != point_id:
            return _Observation(VectorSmokeReason.ID_MISMATCH, 1, 0)
        vector = record.get("vector")
        if (
            not isinstance(vector, list)
            or len(vector) != SMOKE_DIMENSIONS
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in vector
            )
            or tuple(float(value) for value in vector) != _SYNTHETIC_VECTOR
        ):
            return _Observation(VectorSmokeReason.VECTOR_MISMATCH, 1, 0)
        return _Observation(VectorSmokeReason.VERIFIED, 1, 0)

    @staticmethod
    def _write_succeeded(payload: object) -> bool:
        return bool(
            isinstance(payload, dict)
            and set(payload) == {"result"}
            and payload.get("result") == "Success"
        )

    @staticmethod
    def _delete_succeeded(payload: object) -> bool:
        if not isinstance(payload, dict) or set(payload) != {"result"}:
            return False
        result = payload.get("result")
        return bool(
            result == "Success"
            or (
                isinstance(result, dict)
                and set(result) == {"deleted"}
                and type(result.get("deleted")) is int
                and int(result["deleted"]) in {0, 1}
            )
        )

    async def _send(
        self,
        method: str,
        operation: str,
        payload: object,
        *,
        deadline: RequestDeadline,
    ) -> _WireOutcome:
        remaining = deadline.remaining_seconds()
        if remaining <= 0 or self._client.is_closed:
            return _WireOutcome("timeout" if remaining <= 0 else "unavailable")
        response: httpx.Response | None = None
        raw: bytes | None = None
        failure: Literal["timeout", "unavailable", "invalid"] | None = None
        parsed: object | None = None
        try:
            headers = {
                "Authorization": f"Bearer {self._token.get_secret_value()}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            request = self._client.build_request(
                method,
                f"{self._url}/{operation}/{SMOKE_NAMESPACE}",
                headers=headers,
                json=payload,
            )
            headers = {}
            async with asyncio.timeout(remaining):
                response = await self._client.send(
                    request,
                    stream=True,
                    follow_redirects=False,
                )
                if response.status_code != 200:
                    failure = "unavailable"
                elif response.headers.get("content-type", "").split(";", 1)[0].lower() != (
                    "application/json"
                ):
                    failure = "invalid"
                else:
                    parts: list[bytes] = []
                    received = 0
                    async for part in response.aiter_bytes():
                        received += len(part)
                        if received > _MAX_RESPONSE_BYTES:
                            failure = "invalid"
                            break
                        parts.append(bytes(part))
                    if failure is None:
                        raw = b"".join(parts)
        except (TimeoutError, httpx.TimeoutException):
            failure = "timeout"
        except Exception:
            failure = "unavailable"
        finally:
            if response is not None:
                await response.aclose()
        if failure is None and raw is not None:
            try:
                parsed = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                failure = "invalid"
        raw = None
        response = None
        if failure is not None:
            return _WireOutcome(failure)
        return _WireOutcome("ok", payload=parsed)


__all__ = [
    "SMOKE_DIMENSIONS",
    "SMOKE_NAMESPACE",
    "VectorSmokeProbe",
    "VectorSmokeReason",
    "VectorSmokeResult",
]
