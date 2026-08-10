from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import replace
from typing import NoReturn
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from app.research.control import (
    AccessionLease,
    GenerationStageRecord,
    GenerationState,
    IngestionCheckpoint,
    IngestionRetryAttemptTwoClaim,
    IngestionRetryAttemptTwoSnapshot,
    IngestionRetryClaim,
    IngestionRetryResult,
    IngestionRetrySnapshot,
    ResearchControlUnavailableError,
    Reservation,
    ReservationState,
    SupersededCleanupRecord,
    SupersededCleanupState,
    recoverable_generation_records,
    require_public_digest,
)
from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    GenerationManifest,
    GenerationVerification,
)

_LEASE_RELEASE_SCRIPT = (
    "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('del',KEYS[1]) end; return 0"
)
_STAGE_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "local current=redis.call('get',KEYS[2]); "
    "if not current or current==ARGV[2] or current==ARGV[3] or current==ARGV[4] "
    "or current==ARGV[5] or current==ARGV[6] or current==ARGV[7] then "
    "redis.call('set',KEYS[2],ARGV[2]); return 1 end; return -2"
)
_VERIFY_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "if redis.call('get',KEYS[2])~=ARGV[2] then return -2 end; "
    "redis.call('set',KEYS[2],ARGV[3]); return 1"
)
_PUBLISH_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "if redis.call('get',KEYS[2])~=ARGV[2] then return -2 end; "
    "local current=redis.call('get',KEYS[3]); "
    "if ARGV[4]=='' then if current then return -3 end "
    "elseif current~=ARGV[4] then return -3 end; "
    "if ARGV[7]~='' then local cleanup=redis.call('get',KEYS[5]); "
    "if cleanup and cleanup~=ARGV[7] then return -4 end end; "
    "redis.call('set',KEYS[2],ARGV[3]); redis.call('set',KEYS[3],ARGV[5]); "
    "redis.call('sadd',KEYS[4],ARGV[6]); "
    "if ARGV[7]~='' then redis.call('set',KEYS[5],ARGV[7]); "
    "redis.call('sadd',KEYS[6],ARGV[8]) end; return 1"
)
_ABORT_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "if redis.call('get',KEYS[2])~=ARGV[2] then return -2 end; "
    "redis.call('set',KEYS[2],ARGV[3]); return 1"
)
_CLEAN_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "if redis.call('get',KEYS[2])~=ARGV[2] then return -2 end; "
    "redis.call('set',KEYS[2],ARGV[3],'EX',ARGV[4]); return 1"
)
_CLEAN_SUPERSEDED_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "if redis.call('get',KEYS[2])==ARGV[2] then return -2 end; "
    "if redis.call('get',KEYS[3])~=ARGV[3] then return 0 end; "
    "redis.call('set',KEYS[3],ARGV[4],'EX',ARGV[6]); "
    "redis.call('srem',KEYS[4],ARGV[5]); return 1"
)
_CHECKPOINT_SCRIPT = (
    "local current=redis.call('get',KEYS[1]); "
    "if ARGV[1]=='' then if current then return 0 end "
    "elseif current~=ARGV[1] then return 0 end; "
    "redis.call('set',KEYS[1],ARGV[2]); return 1"
)
_CLAIM_RETRY_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "if redis.call('exists',KEYS[2])==1 or redis.call('exists',KEYS[3])==1 "
    "then return 0 end; redis.call('set',KEYS[2],ARGV[2]); return 1"
)
_FINISH_RETRY_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "local current=redis.call('get',KEYS[2]); "
    "if current==ARGV[2] then return 1 end; "
    "if current then return -2 end; redis.call('set',KEYS[2],ARGV[2]); return 1"
)
_CLAIM_RETRY_ATTEMPT_TWO_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "if redis.call('get',KEYS[2])~=ARGV[2] then return -2 end; "
    "if redis.call('get',KEYS[3])~=ARGV[3] then return -3 end; "
    "if redis.call('exists',KEYS[4])==1 or redis.call('exists',KEYS[5])==1 "
    "then return 0 end; redis.call('set',KEYS[4],ARGV[4]); return 1"
)
_FINISH_RETRY_ATTEMPT_TWO_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return -1 end; "
    "if redis.call('get',KEYS[2])~=ARGV[2] then return -2 end; "
    "if redis.call('get',KEYS[3])~=ARGV[3] then return -3 end; "
    "if redis.call('get',KEYS[4])~=ARGV[4] then return -4 end; "
    "local current=redis.call('get',KEYS[5]); "
    "if current==ARGV[5] then return 1 end; "
    "if current then return 0 end; redis.call('set',KEYS[5],ARGV[5]); return 1"
)
_AUTHORIZE_RESERVATION_SCRIPT = (
    "if redis.call('exists',KEYS[1])==1 then return 0 end; "
    "local current=tonumber(redis.call('get',KEYS[2]) or '0'); "
    "if current>0 and redis.call('ttl',KEYS[2])<1 then return -2 end; "
    "local units=tonumber(ARGV[2]); local limit=tonumber(ARGV[3]); "
    "if current+units>limit then return -1 end; "
    "redis.call('set',KEYS[1],ARGV[1],'EX',ARGV[4]); "
    "local updated=redis.call('incrby',KEYS[2],units); "
    "if current==0 then redis.call('expire',KEYS[2],ARGV[4]) end; return updated"
)
_RESERVATION_TRANSITION_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return 0 end; "
    "redis.call('set',KEYS[1],ARGV[2],'KEEPTTL'); return 1"
)
_RELEASE_RESERVATION_SCRIPT = (
    "if redis.call('get',KEYS[1])~=ARGV[1] then return 0 end; "
    "local current=tonumber(redis.call('get',KEYS[2]) or '0'); "
    "local units=tonumber(ARGV[3]); if current<units then return -1 end; "
    "redis.call('set',KEYS[1],ARGV[2],'KEEPTTL'); local remaining=current-units; "
    "if remaining==0 then redis.call('del',KEYS[2]) "
    "else redis.call('set',KEYS[2],remaining,'KEEPTTL') end; return 1"
)

_SCHEMA_VERSION = "v1"
_KEY_PREFIX = f"rag:{_SCHEMA_VERSION}"
_MAX_RESPONSE_BYTES = 1_048_576
_GENERATION_ID = re.compile(r"^gen-([0-9a-f]{64})$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_UPSTASH_REST_HOST = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+upstash\.io$")
_SAFE_TEST_HOST = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.example$")


def _raise_unavailable() -> NoReturn:
    raise ResearchControlUnavailableError("research control plane is unavailable")


def _safe_endpoint(endpoint: object, *, allow_test_endpoint: bool) -> str:
    if not isinstance(endpoint, str):
        raise ValueError("research control endpoint must be an approved HTTPS root")
    normalized = endpoint.strip()
    if not normalized or any(
        ord(character) < 32 or ord(character) == 127 for character in endpoint
    ):
        raise ValueError("research control endpoint must be an approved HTTPS root")
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
    except ValueError:
        raise ValueError("research control endpoint must be an approved HTTPS root") from None
    hostname = parsed.hostname.lower() if parsed.hostname else ""
    approved_host = _UPSTASH_REST_HOST.fullmatch(hostname) is not None
    safe_test_host = allow_test_endpoint and _SAFE_TEST_HOST.fullmatch(hostname) is not None
    if not (
        parsed.scheme == "https"
        and (approved_host or safe_test_host)
        and parsed.username is None
        and parsed.password is None
        and port is None
        and parsed.query == ""
        and parsed.fragment == ""
        and parsed.path in {"", "/"}
    ):
        raise ValueError("research control endpoint must be an approved HTTPS root")
    return f"https://{hostname}"


def _positive_seconds(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _digest_parts(*parts: str) -> str:
    material = "\x1f".join(parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _accession_digest(manifest: GenerationManifest) -> str:
    return _digest_parts(
        manifest.corpus.canonical_key,
        manifest.symbol,
        manifest.accession_number,
    )


def _cleanup_digest(record: SupersededCleanupRecord) -> str:
    return _digest_parts(
        _accession_digest(record.superseded_manifest),
        record.superseded_manifest.generation_id,
        record.active_generation_id,
    )


def _scope_digest(corpus: CorpusDescriptor, symbol: str) -> str:
    normalized = symbol.strip().upper()
    if _SYMBOL.fullmatch(normalized) is None:
        raise ValueError("research symbol is invalid")
    return _digest_parts(corpus.canonical_key, normalized)


def _filing_identity_digest(corpus: CorpusDescriptor, symbol: str, accession_number: str) -> str:
    normalized_symbol = symbol.strip().upper()
    if _SYMBOL.fullmatch(normalized_symbol) is None:
        raise ValueError("research symbol is invalid")
    if not isinstance(accession_number, str) or _ACCESSION.fullmatch(accession_number) is None:
        raise ValueError("research accession number is invalid")
    return _digest_parts(corpus.canonical_key, normalized_symbol, accession_number)


def _generation_digest(generation_id: object) -> str:
    if not isinstance(generation_id, str):
        raise ValueError("generation ID is invalid")
    matched = _GENERATION_ID.fullmatch(generation_id)
    if matched is None:
        raise ValueError("generation ID is invalid")
    return matched.group(1)


class RedisResearchControl:
    """Atomic RAG ingestion and budget control state over the Upstash Redis REST API."""

    def __init__(
        self,
        endpoint: str,
        token: SecretStr,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 3.0,
    ) -> None:
        allow_test_endpoint = client is not None and isinstance(
            getattr(client, "_transport", None), httpx.MockTransport
        )
        self._endpoint = _safe_endpoint(endpoint, allow_test_endpoint=allow_test_endpoint)
        if not isinstance(token, SecretStr):
            raise TypeError("research control token must be SecretStr")
        if not token.get_secret_value():
            raise ValueError("research control credentials are required")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("research control timeout must be positive")
        self._token = token
        self._client = client or httpx.AsyncClient(
            timeout=float(timeout_seconds),
            trust_env=False,
            follow_redirects=False,
        )
        self._owns_client = client is None

    def __repr__(self) -> str:
        return "RedisResearchControl(endpoint=<redacted>, token=SecretStr('**********'))"

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    async def __aenter__(self) -> RedisResearchControl:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _command(self, command: list[object], *, deadline: RequestDeadline) -> object:
        deadline.raise_if_expired()
        timeout = deadline.remaining_seconds()
        if timeout <= 0:
            raise TimeoutError("research request deadline expired")
        failed = False
        timed_out = False
        body: object = None
        response: httpx.Response | None = None
        chunks: tuple[bytes, ...] = ()
        chunk = b""
        try:
            async with asyncio.timeout(timeout):
                async with self._client.stream(
                    "POST",
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._token.get_secret_value()}",
                        "Content-Type": "application/json",
                    },
                    json=command,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    if 200 <= response.status_code < 300:
                        received = 0
                        async for chunk in response.aiter_bytes():
                            deadline.raise_if_expired()
                            received += len(chunk)
                            if received > _MAX_RESPONSE_BYTES:
                                failed = True
                                break
                            chunks = (*chunks, chunk)
                        if not failed:
                            body = json.loads(b"".join(chunks))
                    else:
                        failed = True
        except TimeoutError:
            body = None
            response = None
            chunks = ()
            chunk = b""
            timed_out = True
        except (httpx.HTTPError, TypeError, ValueError):
            failed = True
        if timed_out:
            raise TimeoutError("research request deadline expired")
        if failed:
            body = None
            response = None
            chunks = ()
            chunk = b""
            _raise_unavailable()
        if (
            not isinstance(body, dict)
            or "result" not in body
            or not frozenset(body).issubset({"result", "error"})
            or body.get("error") is not None
        ):
            body = None
            response = None
            chunks = ()
            chunk = b""
            _raise_unavailable()
        result = body["result"]
        body = None
        response = None
        chunks = ()
        chunk = b""
        return result

    @staticmethod
    def _cas_result(value: object) -> bool:
        if type(value) is not int or value not in {-4, -3, -2, -1, 0, 1}:
            _raise_unavailable()
        return value == 1

    @staticmethod
    def _lease_key(accession_digest: str) -> str:
        return f"{_KEY_PREFIX}:lease:{require_public_digest(accession_digest)}"

    @staticmethod
    def _generation_key(accession_digest: str, generation_id: str) -> str:
        accession = require_public_digest(accession_digest)
        generation = _generation_digest(generation_id)
        return f"{_KEY_PREFIX}:generation:{accession}:{generation}"

    @staticmethod
    def _active_key(accession_digest: str) -> str:
        return f"{_KEY_PREFIX}:active:{require_public_digest(accession_digest)}"

    @staticmethod
    def _index_key(corpus: CorpusDescriptor, symbol: str) -> str:
        return f"{_KEY_PREFIX}:active-index:{_scope_digest(corpus, symbol)}"

    @staticmethod
    def _checkpoint_key(job_digest: str) -> str:
        return f"{_KEY_PREFIX}:checkpoint:{require_public_digest(job_digest)}"

    @staticmethod
    def _retry_claim_key(job_digest: str) -> str:
        return f"{_KEY_PREFIX}:checkpoint-retry:{require_public_digest(job_digest)}"

    @staticmethod
    def _retry_result_key(job_digest: str) -> str:
        return f"{_KEY_PREFIX}:checkpoint-retry-result:{require_public_digest(job_digest)}"

    @staticmethod
    def _retry_attempt_two_claim_key(job_digest: str) -> str:
        return f"{_KEY_PREFIX}:checkpoint-retry-attempt-2:{require_public_digest(job_digest)}"

    @staticmethod
    def _retry_attempt_two_result_key(job_digest: str) -> str:
        return (
            f"{_KEY_PREFIX}:checkpoint-retry-attempt-2-result:{require_public_digest(job_digest)}"
        )

    @staticmethod
    def _reservation_key(reservation_digest: str) -> str:
        return f"{_KEY_PREFIX}:reservation:{require_public_digest(reservation_digest)}"

    @staticmethod
    def _budget_key(budget_digest: str) -> str:
        return f"{_KEY_PREFIX}:budget:{require_public_digest(budget_digest)}"

    @staticmethod
    def _cleanup_key(cleanup_digest: str) -> str:
        return f"{_KEY_PREFIX}:cleanup:{require_public_digest(cleanup_digest)}"

    @staticmethod
    def _cleanup_index_key(corpus: CorpusDescriptor, symbol: str) -> str:
        return f"{_KEY_PREFIX}:cleanup-index:{_scope_digest(corpus, symbol)}"

    @staticmethod
    def _assert_lease_matches(lease: AccessionLease, manifest: GenerationManifest) -> None:
        if lease.accession_digest != _accession_digest(manifest):
            raise ValueError("lease and generation accession digests must match")

    async def acquire_generation_lease(
        self,
        *,
        manifest: GenerationManifest,
        owner_digest: str,
        ttl_seconds: int,
        deadline: RequestDeadline,
    ) -> AccessionLease | None:
        lease = AccessionLease(
            accession_digest=_accession_digest(manifest),
            owner_digest=owner_digest,
        )
        ttl = _positive_seconds(ttl_seconds, "lease TTL")
        result = await self._command(
            ["SET", self._lease_key(lease.accession_digest), lease.owner_digest, "NX", "EX", ttl],
            deadline=deadline,
        )
        if result is None:
            return None
        if result != "OK":
            _raise_unavailable()
        return lease

    async def release_generation_lease(
        self, *, lease: AccessionLease, deadline: RequestDeadline
    ) -> bool:
        result = await self._command(
            [
                "EVAL",
                _LEASE_RELEASE_SCRIPT,
                1,
                self._lease_key(lease.accession_digest),
                lease.owner_digest,
            ],
            deadline=deadline,
        )
        if type(result) is not int or result not in {0, 1}:
            _raise_unavailable()
        return result == 1

    async def stage_generation(
        self,
        *,
        lease: AccessionLease,
        manifest: GenerationManifest,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None:
        self._assert_lease_matches(lease, manifest)
        recoverable = recoverable_generation_records(manifest)
        staged = recoverable[0]
        result = await self._command(
            [
                "EVAL",
                _STAGE_SCRIPT,
                2,
                self._lease_key(lease.accession_digest),
                self._generation_key(lease.accession_digest, manifest.generation_id),
                lease.owner_digest,
                *(record.to_json() for record in recoverable),
            ],
            deadline=deadline,
        )
        return staged if self._cas_result(result) else None

    async def mark_generation_verified(
        self,
        *,
        lease: AccessionLease,
        staged: GenerationStageRecord,
        verification: GenerationVerification,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None:
        if staged.state is not GenerationState.STAGED:
            raise ValueError("generation must be staged before verification")
        self._assert_lease_matches(lease, staged.manifest)
        if not verification.proves(staged.manifest):
            raise ValueError("generation verification does not prove the staged point set")
        verified = replace(
            staged,
            state=GenerationState.VERIFIED,
            verification=verification,
        )
        result = await self._command(
            [
                "EVAL",
                _VERIFY_SCRIPT,
                2,
                self._lease_key(lease.accession_digest),
                self._generation_key(lease.accession_digest, staged.manifest.generation_id),
                lease.owner_digest,
                staged.to_json(),
                verified.to_json(),
            ],
            deadline=deadline,
        )
        return verified if self._cas_result(result) else None

    async def publish_generation(
        self,
        *,
        lease: AccessionLease,
        verified_stage: GenerationStageRecord,
        expected_previous_generation_id: str | None,
        manifest: GenerationManifest,
        superseded_manifest: GenerationManifest | None,
        deadline: RequestDeadline,
    ) -> bool:
        if (
            verified_stage.state is not GenerationState.VERIFIED
            or verified_stage.manifest != manifest
            or verified_stage.verification is None
            or not verified_stage.verification.proves(manifest)
        ):
            raise ValueError("publication requires the exact verified generation manifest")
        self._assert_lease_matches(lease, manifest)
        expected = ""
        if expected_previous_generation_id is not None:
            _generation_digest(expected_previous_generation_id)
            expected = expected_previous_generation_id
        if (expected_previous_generation_id is None) != (superseded_manifest is None):
            raise ValueError("replacement publication requires the superseded manifest")
        pending: SupersededCleanupRecord | None = None
        cleanup_digest = ""
        cleanup_key = self._active_key(lease.accession_digest)
        cleanup_index = self._index_key(manifest.corpus, manifest.symbol)
        if superseded_manifest is not None:
            if (
                superseded_manifest.corpus != manifest.corpus
                or superseded_manifest.symbol != manifest.symbol
                or superseded_manifest.accession_number != manifest.accession_number
                or superseded_manifest.generation_id != expected_previous_generation_id
                or superseded_manifest.generation_id == manifest.generation_id
            ):
                raise ValueError("superseded manifest does not match the replacement")
            pending = SupersededCleanupRecord(
                superseded_manifest=superseded_manifest,
                active_generation_id=manifest.generation_id,
                state=SupersededCleanupState.PENDING,
            )
            cleanup_digest = _cleanup_digest(pending)
            cleanup_key = self._cleanup_key(cleanup_digest)
            cleanup_index = self._cleanup_index_key(
                superseded_manifest.corpus, superseded_manifest.symbol
            )
        published = verified_stage.with_state(GenerationState.PUBLISHED)
        result = await self._command(
            [
                "EVAL",
                _PUBLISH_SCRIPT,
                6,
                self._lease_key(lease.accession_digest),
                self._generation_key(lease.accession_digest, manifest.generation_id),
                self._active_key(lease.accession_digest),
                self._index_key(manifest.corpus, manifest.symbol),
                cleanup_key,
                cleanup_index,
                lease.owner_digest,
                verified_stage.to_json(),
                published.to_json(),
                expected,
                manifest.generation_id,
                lease.accession_digest,
                "" if pending is None else pending.to_json(),
                cleanup_digest,
            ],
            deadline=deadline,
        )
        return self._cas_result(result)

    async def abort_generation(
        self,
        *,
        lease: AccessionLease,
        stage: GenerationStageRecord,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None:
        if stage.state not in {GenerationState.STAGED, GenerationState.VERIFIED}:
            raise ValueError("only a staged or verified generation can be aborted")
        self._assert_lease_matches(lease, stage.manifest)
        aborted = stage.with_state(GenerationState.ABORTED)
        result = await self._command(
            [
                "EVAL",
                _ABORT_SCRIPT,
                2,
                self._lease_key(lease.accession_digest),
                self._generation_key(lease.accession_digest, stage.manifest.generation_id),
                lease.owner_digest,
                stage.to_json(),
                aborted.to_json(),
            ],
            deadline=deadline,
        )
        return aborted if self._cas_result(result) else None

    async def clean_generation(
        self,
        *,
        lease: AccessionLease,
        aborted_stage: GenerationStageRecord,
        marker_ttl_seconds: int,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None:
        if aborted_stage.state is not GenerationState.ABORTED:
            raise ValueError("only an aborted generation can be cleaned")
        self._assert_lease_matches(lease, aborted_stage.manifest)
        ttl = _positive_seconds(marker_ttl_seconds, "cleanup marker TTL")
        cleaned = aborted_stage.with_state(GenerationState.CLEANED)
        result = await self._command(
            [
                "EVAL",
                _CLEAN_SCRIPT,
                2,
                self._lease_key(lease.accession_digest),
                self._generation_key(lease.accession_digest, aborted_stage.manifest.generation_id),
                lease.owner_digest,
                aborted_stage.to_json(),
                cleaned.to_json(),
                ttl,
            ],
            deadline=deadline,
        )
        return cleaned if self._cas_result(result) else None

    async def get_active_generation(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        accession_number: str,
        deadline: RequestDeadline,
    ) -> GenerationManifest | None:
        accession_digest = _filing_identity_digest(corpus, symbol, accession_number)
        active = await self._command(["GET", self._active_key(accession_digest)], deadline=deadline)
        if active is None:
            return None
        generation_id = self._parse_generation_id(active)
        raw = await self._command(
            ["GET", self._generation_key(accession_digest, generation_id)],
            deadline=deadline,
        )
        record = self._parse_stage(raw)
        if (
            record.state is not GenerationState.PUBLISHED
            or record.manifest.corpus != corpus
            or record.manifest.symbol != symbol.strip().upper()
            or record.manifest.accession_number != accession_number
            or record.manifest.generation_id != generation_id
        ):
            _raise_unavailable()
        return record.manifest

    async def get_generation_stage(
        self,
        *,
        manifest: GenerationManifest,
        deadline: RequestDeadline,
    ) -> GenerationStageRecord | None:
        accession_digest = _accession_digest(manifest)
        raw = await self._command(
            ["GET", self._generation_key(accession_digest, manifest.generation_id)],
            deadline=deadline,
        )
        if raw is None:
            return None
        record = self._parse_stage(raw)
        if record.manifest != manifest:
            _raise_unavailable()
        return record

    async def list_active_generations(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        deadline: RequestDeadline,
    ) -> tuple[GenerationManifest, ...]:
        normalized_symbol = symbol.strip().upper()
        raw_digests = await self._command(
            ["SMEMBERS", self._index_key(corpus, normalized_symbol)],
            deadline=deadline,
        )
        if not isinstance(raw_digests, list):
            _raise_unavailable()
        failed = False
        accession_digests: list[str] = []
        for raw in raw_digests:
            try:
                accession_digests.append(require_public_digest(raw, "accession digest"))
            except ValueError:
                failed = True
        if failed or len(accession_digests) != len(set(accession_digests)):
            _raise_unavailable()
        accession_digests.sort()
        if not accession_digests:
            return ()
        raw_active = await self._command(
            ["MGET", *(self._active_key(item) for item in accession_digests)],
            deadline=deadline,
        )
        if not isinstance(raw_active, list) or len(raw_active) != len(accession_digests):
            _raise_unavailable()
        if any(raw is None for raw in raw_active):
            _raise_unavailable()
        generation_pairs: list[tuple[str, str]] = []
        for accession_digest, raw in zip(accession_digests, raw_active, strict=True):
            if raw is None:
                continue
            generation_pairs.append((accession_digest, self._parse_generation_id(raw)))
        if not generation_pairs:
            return ()
        raw_records = await self._command(
            [
                "MGET",
                *(
                    self._generation_key(accession_digest, generation_id)
                    for accession_digest, generation_id in generation_pairs
                ),
            ],
            deadline=deadline,
        )
        if not isinstance(raw_records, list) or len(raw_records) != len(generation_pairs):
            _raise_unavailable()
        records = tuple(self._parse_stage(raw) for raw in raw_records)
        for record, (accession_digest, generation_id) in zip(
            records, generation_pairs, strict=True
        ):
            if (
                record.state is not GenerationState.PUBLISHED
                or record.manifest.corpus != corpus
                or record.manifest.symbol != normalized_symbol
                or record.manifest.generation_id != generation_id
                or _accession_digest(record.manifest) != accession_digest
            ):
                _raise_unavailable()
        return tuple(record.manifest for record in records)

    async def list_pending_cleanups(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        deadline: RequestDeadline,
    ) -> tuple[SupersededCleanupRecord, ...]:
        normalized_symbol = symbol.strip().upper()
        index_key = self._cleanup_index_key(corpus, normalized_symbol)
        raw_digests = await self._command(["SMEMBERS", index_key], deadline=deadline)
        if not isinstance(raw_digests, list):
            _raise_unavailable()
        failed = False
        cleanup_digests: list[str] = []
        for raw in raw_digests:
            try:
                cleanup_digests.append(require_public_digest(raw, "cleanup digest"))
            except ValueError:
                failed = True
        if failed or len(cleanup_digests) != len(set(cleanup_digests)):
            _raise_unavailable()
        cleanup_digests.sort()
        if not cleanup_digests:
            return ()
        raw_records = await self._command(
            ["MGET", *(self._cleanup_key(item) for item in cleanup_digests)],
            deadline=deadline,
        )
        if (
            not isinstance(raw_records, list)
            or len(raw_records) != len(cleanup_digests)
            or any(raw is None for raw in raw_records)
        ):
            _raise_unavailable()
        records = tuple(self._parse_cleanup(raw) for raw in raw_records)
        for record, cleanup_digest in zip(records, cleanup_digests, strict=True):
            manifest = record.superseded_manifest
            if (
                record.state is not SupersededCleanupState.PENDING
                or manifest.corpus != corpus
                or manifest.symbol != normalized_symbol
                or _cleanup_digest(record) != cleanup_digest
            ):
                _raise_unavailable()
        return records

    async def mark_superseded_cleaned(
        self,
        *,
        lease: AccessionLease,
        pending: SupersededCleanupRecord,
        marker_ttl_seconds: int,
        deadline: RequestDeadline,
    ) -> SupersededCleanupRecord | None:
        if pending.state is not SupersededCleanupState.PENDING:
            raise ValueError("only pending superseded cleanup can be completed")
        self._assert_lease_matches(lease, pending.superseded_manifest)
        ttl = _positive_seconds(marker_ttl_seconds, "cleanup marker TTL")
        cleaned = replace(pending, state=SupersededCleanupState.CLEANED)
        cleanup_digest = _cleanup_digest(pending)
        result = await self._command(
            [
                "EVAL",
                _CLEAN_SUPERSEDED_SCRIPT,
                4,
                self._lease_key(lease.accession_digest),
                self._active_key(lease.accession_digest),
                self._cleanup_key(cleanup_digest),
                self._cleanup_index_key(
                    pending.superseded_manifest.corpus,
                    pending.superseded_manifest.symbol,
                ),
                lease.owner_digest,
                pending.superseded_manifest.generation_id,
                pending.to_json(),
                cleaned.to_json(),
                cleanup_digest,
                ttl,
            ],
            deadline=deadline,
        )
        return cleaned if self._cas_result(result) else None

    @staticmethod
    def _parse_generation_id(raw: object) -> str:
        failed = False
        try:
            _generation_digest(raw)
            generation_id = raw
        except ValueError:
            failed = True
            generation_id = None
        if failed or not isinstance(generation_id, str):
            _raise_unavailable()
        return generation_id

    @staticmethod
    def _parse_stage(raw: object) -> GenerationStageRecord:
        failed = False
        try:
            record = GenerationStageRecord.from_json(raw)
        except ValueError:
            failed = True
            record = None
        if failed or record is None:
            _raise_unavailable()
        return record

    @staticmethod
    def _parse_cleanup(raw: object) -> SupersededCleanupRecord:
        failed = False
        try:
            record = SupersededCleanupRecord.from_json(raw)
        except ValueError:
            failed = True
            record = None
        if failed or record is None:
            _raise_unavailable()
        return record

    async def save_checkpoint(
        self,
        *,
        checkpoint: IngestionCheckpoint,
        expected_previous: IngestionCheckpoint | None,
        deadline: RequestDeadline,
    ) -> bool:
        if expected_previous is not None:
            if expected_previous.job_digest != checkpoint.job_digest:
                raise ValueError("checkpoint job digests must match")
            if (
                checkpoint.processed_count < expected_previous.processed_count
                or checkpoint.failed_count < expected_previous.failed_count
                or (expected_previous.complete and not checkpoint.complete)
            ):
                raise ValueError("checkpoint progress must be monotonic")
        result = await self._command(
            [
                "EVAL",
                _CHECKPOINT_SCRIPT,
                1,
                self._checkpoint_key(checkpoint.job_digest),
                "" if expected_previous is None else expected_previous.to_json(),
                checkpoint.to_json(),
            ],
            deadline=deadline,
        )
        return self._cas_result(result)

    async def load_checkpoint(
        self, *, job_digest: str, deadline: RequestDeadline
    ) -> IngestionCheckpoint | None:
        raw = await self._command(["GET", self._checkpoint_key(job_digest)], deadline=deadline)
        if raw is None:
            return None
        failed = False
        try:
            checkpoint = IngestionCheckpoint.from_json(raw)
        except ValueError:
            failed = True
            checkpoint = None
        if failed or checkpoint is None or checkpoint.job_digest != job_digest:
            _raise_unavailable()
        return checkpoint

    async def load_retry_snapshot(
        self, *, job_digest: str, deadline: RequestDeadline
    ) -> IngestionRetrySnapshot:
        checked = require_public_digest(job_digest, "job digest")
        raw = await self._command(
            [
                "MGET",
                self._checkpoint_key(checked),
                self._retry_claim_key(checked),
                self._retry_result_key(checked),
            ],
            deadline=deadline,
        )
        if not isinstance(raw, list) or len(raw) != 3:
            _raise_unavailable()
        try:
            checkpoint = None if raw[0] is None else IngestionCheckpoint.from_json(raw[0])
            claim = None if raw[1] is None else IngestionRetryClaim.from_json(raw[1])
            result = None if raw[2] is None else IngestionRetryResult.from_json(raw[2])
            snapshot = IngestionRetrySnapshot(checkpoint, claim, result)
        except ValueError:
            _raise_unavailable()
        if any(item.job_digest != checked for item in (checkpoint, claim, result) if item):
            _raise_unavailable()
        return snapshot

    async def claim_failed_retry(
        self,
        *,
        checkpoint: IngestionCheckpoint,
        claim: IngestionRetryClaim,
        deadline: RequestDeadline,
    ) -> bool:
        if checkpoint.job_digest != claim.job_digest:
            raise ValueError("retry claim and checkpoint job digests must match")
        result = await self._command(
            [
                "EVAL",
                _CLAIM_RETRY_SCRIPT,
                3,
                self._checkpoint_key(claim.job_digest),
                self._retry_claim_key(claim.job_digest),
                self._retry_result_key(claim.job_digest),
                checkpoint.to_json(),
                claim.to_json(),
            ],
            deadline=deadline,
        )
        return self._cas_result(result)

    async def finish_failed_retry(
        self,
        *,
        claim: IngestionRetryClaim,
        result: IngestionRetryResult,
        deadline: RequestDeadline,
    ) -> bool:
        if claim.job_digest != result.job_digest or claim.attempt_digest != result.attempt_digest:
            raise ValueError("retry claim and result do not match")
        saved = await self._command(
            [
                "EVAL",
                _FINISH_RETRY_SCRIPT,
                2,
                self._retry_claim_key(claim.job_digest),
                self._retry_result_key(claim.job_digest),
                claim.to_json(),
                result.to_json(),
            ],
            deadline=deadline,
        )
        return self._cas_result(saved)

    async def load_retry_attempt_two_snapshot(
        self, *, job_digest: str, deadline: RequestDeadline
    ) -> IngestionRetryAttemptTwoSnapshot:
        checked = require_public_digest(job_digest, "job digest")
        raw = await self._command(
            [
                "MGET",
                self._checkpoint_key(checked),
                self._retry_claim_key(checked),
                self._retry_result_key(checked),
                self._retry_attempt_two_claim_key(checked),
                self._retry_attempt_two_result_key(checked),
            ],
            deadline=deadline,
        )
        if not isinstance(raw, list) or len(raw) != 5:
            _raise_unavailable()
        try:
            checkpoint = None if raw[0] is None else IngestionCheckpoint.from_json(raw[0])
            first_claim = None if raw[1] is None else IngestionRetryClaim.from_json(raw[1])
            first_result = None if raw[2] is None else IngestionRetryResult.from_json(raw[2])
            claim = None if raw[3] is None else IngestionRetryAttemptTwoClaim.from_json(raw[3])
            result = None if raw[4] is None else IngestionRetryResult.from_json(raw[4])
            if checkpoint is None or first_claim is None or first_result is None:
                raise ValueError("attempt-two snapshot is missing its legacy records")
            snapshot = IngestionRetryAttemptTwoSnapshot(
                checkpoint,
                first_claim,
                first_result,
                claim,
                result,
            )
        except ValueError:
            _raise_unavailable()
        if snapshot.checkpoint.job_digest != checked:
            _raise_unavailable()
        return snapshot

    async def claim_failed_retry_attempt_two(
        self,
        *,
        checkpoint: IngestionCheckpoint,
        first_claim: IngestionRetryClaim,
        first_result: IngestionRetryResult,
        claim: IngestionRetryAttemptTwoClaim,
        deadline: RequestDeadline,
    ) -> bool:
        IngestionRetryAttemptTwoSnapshot(checkpoint, first_claim, first_result, claim, None)
        saved = await self._command(
            [
                "EVAL",
                _CLAIM_RETRY_ATTEMPT_TWO_SCRIPT,
                5,
                self._checkpoint_key(claim.job_digest),
                self._retry_claim_key(claim.job_digest),
                self._retry_result_key(claim.job_digest),
                self._retry_attempt_two_claim_key(claim.job_digest),
                self._retry_attempt_two_result_key(claim.job_digest),
                checkpoint.to_json(),
                first_claim.to_json(),
                first_result.to_json(),
                claim.to_json(),
            ],
            deadline=deadline,
        )
        return self._cas_result(saved)

    async def finish_failed_retry_attempt_two(
        self,
        *,
        checkpoint: IngestionCheckpoint,
        first_claim: IngestionRetryClaim,
        first_result: IngestionRetryResult,
        claim: IngestionRetryAttemptTwoClaim,
        result: IngestionRetryResult,
        deadline: RequestDeadline,
    ) -> bool:
        IngestionRetryAttemptTwoSnapshot(checkpoint, first_claim, first_result, claim, result)
        saved = await self._command(
            [
                "EVAL",
                _FINISH_RETRY_ATTEMPT_TWO_SCRIPT,
                5,
                self._checkpoint_key(claim.job_digest),
                self._retry_claim_key(claim.job_digest),
                self._retry_result_key(claim.job_digest),
                self._retry_attempt_two_claim_key(claim.job_digest),
                self._retry_attempt_two_result_key(claim.job_digest),
                checkpoint.to_json(),
                first_claim.to_json(),
                first_result.to_json(),
                claim.to_json(),
                result.to_json(),
            ],
            deadline=deadline,
        )
        return self._cas_result(saved)

    async def authorize_reservation(
        self,
        *,
        reservation: Reservation,
        limit: int,
        window_seconds: int,
        deadline: RequestDeadline,
    ) -> bool:
        if reservation.state is not ReservationState.AUTHORIZED:
            raise ValueError("new reservation must be authorized")
        checked_limit = _positive_seconds(limit, "daily budget limit")
        window = _positive_seconds(window_seconds, "daily budget window")
        if reservation.units > checked_limit:
            return False
        result = await self._command(
            [
                "EVAL",
                _AUTHORIZE_RESERVATION_SCRIPT,
                2,
                self._reservation_key(reservation.reservation_digest),
                self._budget_key(reservation.budget_digest),
                reservation.to_json(),
                reservation.units,
                checked_limit,
                window,
            ],
            deadline=deadline,
        )
        if type(result) is not int:
            _raise_unavailable()
        if result in {-1, 0}:
            return False
        if result < 1 or result > checked_limit:
            _raise_unavailable()
        return True

    async def commit_reservation(
        self, *, reservation: Reservation, deadline: RequestDeadline
    ) -> bool:
        return await self._transition_reservation(
            reservation, ReservationState.COMMITTED, deadline=deadline
        )

    async def release_reservation(
        self, *, reservation: Reservation, deadline: RequestDeadline
    ) -> bool:
        if reservation.state is not ReservationState.AUTHORIZED:
            raise ValueError("only an authorized reservation can transition")
        released = replace(reservation, state=ReservationState.RELEASED)
        result = await self._command(
            [
                "EVAL",
                _RELEASE_RESERVATION_SCRIPT,
                2,
                self._reservation_key(reservation.reservation_digest),
                self._budget_key(reservation.budget_digest),
                reservation.to_json(),
                released.to_json(),
                reservation.units,
            ],
            deadline=deadline,
        )
        if type(result) is not int or result not in {-1, 0, 1}:
            _raise_unavailable()
        if result == -1:
            _raise_unavailable()
        return result == 1

    async def _transition_reservation(
        self,
        reservation: Reservation,
        target: ReservationState,
        *,
        deadline: RequestDeadline,
    ) -> bool:
        if reservation.state is not ReservationState.AUTHORIZED:
            raise ValueError("only an authorized reservation can transition")
        terminal = replace(reservation, state=target)
        result = await self._command(
            [
                "EVAL",
                _RESERVATION_TRANSITION_SCRIPT,
                1,
                self._reservation_key(reservation.reservation_digest),
                reservation.to_json(),
                terminal.to_json(),
            ],
            deadline=deadline,
        )
        return self._cas_result(result)

    async def get_reservation(
        self, *, reservation_digest: str, deadline: RequestDeadline
    ) -> Reservation | None:
        raw = await self._command(
            ["GET", self._reservation_key(reservation_digest)], deadline=deadline
        )
        if raw is None:
            return None
        failed = False
        try:
            reservation = Reservation.from_json(raw)
        except ValueError:
            failed = True
            reservation = None
        if failed or reservation is None or reservation.reservation_digest != reservation_digest:
            _raise_unavailable()
        return reservation
