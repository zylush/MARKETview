from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, cast
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from app.research.deadline import RequestDeadline
from app.research.domain import (
    CorpusDescriptor,
    EmbeddedChunk,
    EmbeddingDescriptor,
    EmbeddingVector,
    EvidenceChunk,
    FilingReference,
    GenerationManifest,
    GenerationVerification,
    SearchHit,
)

_UPSTASH_HOST_SUFFIX = ".upstash.io"
_NAMESPACE = re.compile(r"^sec-filings-v[1-9][0-9]{0,5}$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")
_MAX_RESPONSE_BYTES = 1_048_576
_MAX_BATCH_SIZE = 100
_MAX_CANDIDATES = 50
_MAX_ACTIVE_GENERATIONS = 100


class UpstashVectorConfigurationError(ValueError):
    """Raised when the vector adapter is constructed with an unsafe configuration."""


class UpstashVectorUnavailableError(RuntimeError):
    """Sanitized vector service failure."""


class UpstashVectorTimeoutError(UpstashVectorUnavailableError, TimeoutError):
    """Sanitized vector service deadline failure."""


@dataclass(frozen=True, slots=True)
class _WireOutcome:
    kind: Literal["ok", "timeout", "unavailable", "invalid"]
    status: int | None = None
    payload: object | None = field(default=None, repr=False)


class UpstashVectorStore:
    """Generation-scoped Upstash Vector REST data-plane adapter.

    Publication and active-generation ownership deliberately remain in the
    Redis research control plane. This adapter only stages, verifies, deletes,
    and searches immutable vector records.
    """

    def __init__(
        self,
        url: str,
        token: SecretStr,
        *,
        namespace: str,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        response_max_bytes: int = _MAX_RESPONSE_BYTES,
    ) -> None:
        normalized_url = self._validated_url(url)
        if not isinstance(token, SecretStr) or not token.get_secret_value():
            raise UpstashVectorConfigurationError("a vector service secret is required")
        if not isinstance(namespace, str) or _NAMESPACE.fullmatch(namespace) is None:
            raise UpstashVectorConfigurationError("the vector namespace is not approved")
        if client is not None and transport is not None:
            raise UpstashVectorConfigurationError(
                "provide either an injected client or transport, not both"
            )
        if (
            type(response_max_bytes) is not int
            or not 1024 <= response_max_bytes <= _MAX_RESPONSE_BYTES
        ):
            raise UpstashVectorConfigurationError("the vector response limit is invalid")

        self._url = normalized_url
        self._token = token
        self._namespace = namespace
        self._response_max_bytes = response_max_bytes
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
            raise UpstashVectorConfigurationError("the vector service URL is invalid")
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            raise UpstashVectorConfigurationError("the vector service URL is invalid") from None
        hostname = parsed.hostname
        if (
            parsed.scheme != "https"
            or not hostname
            or not hostname.endswith(_UPSTASH_HOST_SUFFIX)
            or hostname == _UPSTASH_HOST_SUFFIX[1:]
            or parsed.netloc != hostname
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise UpstashVectorConfigurationError("the vector service URL is invalid")
        return f"https://{hostname}"

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    def manages(self, resource: object) -> bool:
        return self._owns_client and resource is self._client

    @staticmethod
    def point_set_hash(chunk_ids: tuple[str, ...]) -> str:
        return hashlib.sha256("\n".join(chunk_ids).encode("ascii")).hexdigest()

    @staticmethod
    def point_digest(
        point_id: str,
        data: str,
        metadata: dict[str, object],
    ) -> str:
        """Return the canonical integrity digest for one immutable vector point."""

        if (
            not isinstance(point_id, str)
            or not isinstance(data, str)
            or not isinstance(metadata, dict)
            or "point_digest" in metadata
        ):
            raise ValueError("vector point digest input is invalid")
        try:
            material = json.dumps(
                {"id": point_id, "data": data, "metadata": metadata},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            raise ValueError("vector point digest input is invalid") from None
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    async def stage_generation(
        self,
        manifest: GenerationManifest,
        chunks: tuple[EmbeddedChunk, ...],
        *,
        deadline: RequestDeadline,
    ) -> None:
        checked = self._validate_stage(manifest, chunks)
        for start in range(0, len(checked), _MAX_BATCH_SIZE):
            batch = checked[start : start + _MAX_BATCH_SIZE]
            request_payload = tuple(self._point_payload(chunk) for chunk in batch)
            outcome = await self._send(
                "POST",
                "/upsert",
                request_payload,
                deadline=deadline,
            )
            request_payload = ()
            if not self._success(outcome):
                self._raise_outcome(outcome)

    async def verify_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> GenerationVerification | None:
        if not isinstance(manifest, GenerationManifest):
            raise ValueError("generation manifest is invalid")
        records: list[tuple[str, dict[str, object], str]] = []
        for start in range(0, len(manifest.chunk_ids), _MAX_BATCH_SIZE):
            batch_ids = manifest.chunk_ids[start : start + _MAX_BATCH_SIZE]
            request_payload: dict[str, object] = {
                "ids": list(batch_ids),
                "includeVectors": False,
                "includeMetadata": True,
                "includeData": True,
            }
            outcome = await self._send(
                "POST",
                "/fetch",
                request_payload,
                deadline=deadline,
            )
            request_payload = {}
            if outcome.kind != "ok":
                self._raise_outcome(outcome)
            decoded = self._decode_fetch(outcome.payload)
            outcome = _WireOutcome("ok")
            if decoded is None:
                return None
            records.extend(decoded)

        point_ids = tuple(point_id for point_id, _, _ in records)
        if point_ids != manifest.chunk_ids or self.point_set_hash(point_ids) != self.point_set_hash(
            manifest.chunk_ids
        ):
            return None
        if not all(
            self._record_matches_manifest(point_id, record, data, manifest)
            for point_id, record, data in records
        ):
            return None
        return GenerationVerification.from_point_ids(manifest.generation_id, point_ids)

    async def abort_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> None:
        await self._delete_manifest_generation(manifest, deadline=deadline)

    async def delete_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> None:
        await self._delete_manifest_generation(manifest, deadline=deadline)

    async def _delete_manifest_generation(
        self,
        manifest: GenerationManifest,
        *,
        deadline: RequestDeadline,
    ) -> None:
        if not isinstance(manifest, GenerationManifest):
            raise ValueError("generation manifest is invalid")
        request_payload: dict[str, object] = {
            "filter": self._generation_filter(manifest),
        }
        outcome = await self._send(
            "DELETE",
            "/delete",
            request_payload,
            deadline=deadline,
        )
        request_payload = {}
        if not self._delete_success(outcome):
            self._raise_outcome(outcome)

    async def search(
        self,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        vector: EmbeddingVector,
        active_generations: tuple[GenerationManifest, ...],
        limit: int,
        deadline: RequestDeadline,
    ) -> tuple[SearchHit, ...]:
        checked_symbol, active = self._validate_search(
            corpus,
            symbol,
            vector,
            active_generations,
            limit,
        )
        if not active:
            return ()
        candidate_limit = min(_MAX_CANDIDATES, limit * 5)
        request_payload: dict[str, object] = {
            "vector": list(vector.values),
            "topK": candidate_limit,
            "includeVectors": False,
            "includeMetadata": True,
            "includeData": True,
            "filter": self._search_filter(
                corpus,
                checked_symbol,
                tuple(item.generation_id for item in active.values()),
            ),
        }
        outcome = await self._send(
            "POST",
            "/query",
            request_payload,
            deadline=deadline,
        )
        request_payload = {}
        if outcome.kind != "ok":
            self._raise_outcome(outcome)
        decoded, malformed = self._decode_hits(
            outcome.payload,
            corpus=corpus,
            symbol=checked_symbol,
            active=active,
            candidate_limit=candidate_limit,
        )
        outcome = _WireOutcome("ok")
        if malformed:
            raise UpstashVectorUnavailableError(
                "the vector service returned invalid data"
            ) from None
        return decoded[:limit]

    @staticmethod
    def _validate_stage(
        manifest: GenerationManifest,
        chunks: tuple[EmbeddedChunk, ...],
    ) -> tuple[EmbeddedChunk, ...]:
        if not isinstance(manifest, GenerationManifest):
            raise ValueError("generation manifest is invalid")
        checked = tuple(chunks)
        if not checked or any(not isinstance(chunk, EmbeddedChunk) for chunk in checked):
            raise ValueError("generation chunks are invalid")
        if tuple(chunk.evidence.chunk_id for chunk in checked) != manifest.chunk_ids:
            raise ValueError("generation manifest does not match its chunks")
        if any(
            chunk.evidence.corpus != manifest.corpus
            or chunk.evidence.symbol != manifest.symbol
            or chunk.evidence.accession_number != manifest.accession_number
            or chunk.evidence.generation_id != manifest.generation_id
            or chunk.evidence.content_hash != manifest.content_hash
            or chunk.embedding.descriptor != manifest.corpus.embedding
            or chunk.evidence.chunk_id
            != EvidenceChunk.canonical_chunk_id(
                chunk.evidence.reference,
                chunk.evidence.corpus,
                chunk.evidence.content_hash,
                chunk.evidence.ordinal,
                text=chunk.evidence.text,
            )
            for chunk in checked
        ):
            raise ValueError("generation manifest metadata does not match its chunks")
        return checked

    @staticmethod
    def _validate_search(
        corpus: CorpusDescriptor,
        symbol: str,
        vector: EmbeddingVector,
        active_generations: tuple[GenerationManifest, ...],
        limit: int,
    ) -> tuple[str, dict[str, GenerationManifest]]:
        if not isinstance(corpus, CorpusDescriptor):
            raise ValueError("research corpus is invalid")
        if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
            raise ValueError("research symbol is invalid")
        if not isinstance(vector, EmbeddingVector) or vector.descriptor != corpus.embedding:
            raise ValueError("query embedding does not match the research corpus")
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("research result limit must be between 1 and 20")
        manifests = tuple(active_generations)
        if len(manifests) > _MAX_ACTIVE_GENERATIONS or any(
            not isinstance(item, GenerationManifest)
            or item.corpus != corpus
            or item.symbol != symbol
            for item in manifests
        ):
            raise ValueError("active generation scope does not match the query")
        active = {item.accession_number: item for item in manifests}
        if len(active) != len(manifests) or len({item.generation_id for item in manifests}) != len(
            manifests
        ):
            raise ValueError("active generation scope contains duplicate accessions")
        return symbol, active

    @staticmethod
    def _point_payload(chunk: EmbeddedChunk) -> dict[str, object]:
        evidence = chunk.evidence
        metadata = UpstashVectorStore._metadata(evidence)
        signed_metadata = {
            **metadata,
            "point_digest": UpstashVectorStore.point_digest(
                evidence.chunk_id,
                evidence.text,
                metadata,
            ),
        }
        return {
            "id": evidence.chunk_id,
            "vector": list(chunk.embedding.values),
            "metadata": signed_metadata,
            "data": evidence.text,
        }

    @staticmethod
    def _metadata(evidence: EvidenceChunk) -> dict[str, object]:
        descriptor = evidence.corpus.embedding
        return {
            "symbol": evidence.symbol,
            "cik": evidence.cik,
            "accession_number": evidence.accession_number,
            "filing_type": evidence.filing_type,
            "title": evidence.title,
            "filed_date": evidence.filed_date.isoformat(),
            "source_url": evidence.source_url,
            "chunk_id": evidence.chunk_id,
            "content_hash": evidence.content_hash,
            "ordinal": evidence.ordinal,
            "corpus_id": evidence.corpus.canonical_key,
            "corpus_version": evidence.corpus.corpus_version,
            "chunker_version": evidence.corpus.chunker_version,
            "embedding_key": descriptor.canonical_key,
            "embedding_provider": descriptor.provider,
            "embedding_model": descriptor.model,
            "embedding_version": descriptor.version,
            "embedding_dimensions": descriptor.dimensions,
            "generation_id": evidence.generation_id,
            "evidence_digest": evidence.evidence_digest,
        }

    @staticmethod
    def _metadata_matches_manifest(
        point_id: str,
        metadata: dict[str, object],
        manifest: GenerationManifest,
    ) -> bool:
        return bool(
            metadata.get("chunk_id") == point_id
            and metadata.get("symbol") == manifest.symbol
            and metadata.get("accession_number") == manifest.accession_number
            and metadata.get("generation_id") == manifest.generation_id
            and metadata.get("content_hash") == manifest.content_hash
            and metadata.get("corpus_id") == manifest.corpus.canonical_key
            and metadata.get("embedding_key") == manifest.corpus.embedding.canonical_key
        )

    @classmethod
    def _record_integrity_matches(
        cls,
        point_id: str,
        metadata: dict[str, object],
        data: str,
    ) -> bool:
        supplied = metadata.get("point_digest")
        if not isinstance(supplied, str) or re.fullmatch(r"[0-9a-f]{64}", supplied) is None:
            return False
        unsigned = {key: value for key, value in metadata.items() if key != "point_digest"}
        try:
            expected = cls.point_digest(point_id, data, unsigned)
        except ValueError:
            return False
        return hmac.compare_digest(supplied, expected)

    @classmethod
    def _record_matches_manifest(
        cls,
        point_id: str,
        metadata: dict[str, object],
        data: str,
        manifest: GenerationManifest,
    ) -> bool:
        if not cls._record_integrity_matches(
            point_id,
            metadata,
            data,
        ) or not cls._metadata_matches_manifest(point_id, metadata, manifest):
            return False
        try:
            descriptor = EmbeddingDescriptor(
                provider=cast(str, metadata["embedding_provider"]),
                model=cast(str, metadata["embedding_model"]),
                version=cast(str, metadata["embedding_version"]),
                dimensions=cast(int, metadata["embedding_dimensions"]),
            )
            evidence_corpus = CorpusDescriptor(
                corpus_version=cast(str, metadata["corpus_version"]),
                chunker_version=cast(str, metadata["chunker_version"]),
                embedding=descriptor,
            )
            reference = FilingReference(
                symbol=cast(str, metadata["symbol"]),
                cik=cast(str, metadata["cik"]),
                accession_number=cast(str, metadata["accession_number"]),
                filing_type=cast(str, metadata["filing_type"]),
                title=cast(str, metadata["title"]),
                filed_date=date.fromisoformat(cast(str, metadata["filed_date"])),
                source_url=cast(str, metadata["source_url"]),
            )
            evidence = EvidenceChunk(
                reference=reference,
                corpus=evidence_corpus,
                generation_id=cast(str, metadata["generation_id"]),
                chunk_id=cast(str, metadata["chunk_id"]),
                content_hash=cast(str, metadata["content_hash"]),
                ordinal=cast(int, metadata["ordinal"]),
                text=data,
            )
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            evidence.chunk_id == point_id
            and evidence.chunk_id
            == EvidenceChunk.canonical_chunk_id(
                evidence.reference,
                evidence.corpus,
                evidence.content_hash,
                evidence.ordinal,
                text=evidence.text,
            )
            and evidence.evidence_digest == metadata.get("evidence_digest")
            and evidence.corpus == manifest.corpus
        )

    @staticmethod
    def _decode_fetch(
        payload: object,
    ) -> tuple[tuple[str, dict[str, object], str], ...] | None:
        if not isinstance(payload, dict) or set(payload) != {"result"}:
            return None
        result = payload.get("result")
        if not isinstance(result, list):
            return None
        decoded: list[tuple[str, dict[str, object], str]] = []
        for item in result:
            if not isinstance(item, dict):
                return None
            point_id = item.get("id")
            record_metadata = item.get("metadata")
            data = item.get("data")
            if (
                not isinstance(point_id, str)
                or not isinstance(record_metadata, dict)
                or not isinstance(data, str)
            ):
                return None
            decoded.append((point_id, dict(record_metadata), data))
        return tuple(decoded)

    @staticmethod
    def _decode_hits(
        payload: object,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        active: dict[str, GenerationManifest],
        candidate_limit: int,
    ) -> tuple[tuple[SearchHit, ...], bool]:
        if not isinstance(payload, dict) or set(payload) != {"result"}:
            return (), True
        result = payload.get("result")
        if not isinstance(result, list) or len(result) > candidate_limit:
            return (), True
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for item in result:
            parsed, malformed = UpstashVectorStore._decode_hit(
                item,
                corpus=corpus,
                symbol=symbol,
                active=active,
            )
            if malformed:
                return (), True
            if parsed is None:
                continue
            if parsed.evidence.chunk_id in seen:
                return (), True
            seen.add(parsed.evidence.chunk_id)
            hits.append(parsed)
        return tuple(hits), False

    @staticmethod
    def _decode_hit(
        value: object,
        *,
        corpus: CorpusDescriptor,
        symbol: str,
        active: dict[str, GenerationManifest],
    ) -> tuple[SearchHit | None, bool]:
        if not isinstance(value, dict):
            return None, True
        point_id = value.get("id")
        score = value.get("score")
        metadata = value.get("metadata")
        text = value.get("data")
        if (
            not isinstance(point_id, str)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not isinstance(metadata, dict)
            or not isinstance(text, str)
        ):
            return None, True
        copied = dict(metadata)
        accession = copied.get("accession_number")
        active_manifest = active.get(accession) if isinstance(accession, str) else None
        if (
            copied.get("symbol") != symbol
            or copied.get("corpus_id") != corpus.canonical_key
            or copied.get("embedding_key") != corpus.embedding.canonical_key
            or active_manifest is None
            or copied.get("generation_id") != active_manifest.generation_id
            or copied.get("content_hash") != active_manifest.content_hash
            or point_id not in active_manifest.chunk_ids
        ):
            return None, False
        if not UpstashVectorStore._record_integrity_matches(point_id, copied, text):
            return None, True
        try:
            descriptor = EmbeddingDescriptor(
                provider=copied["embedding_provider"],
                model=copied["embedding_model"],
                version=copied["embedding_version"],
                dimensions=copied["embedding_dimensions"],
            )
            evidence_corpus = CorpusDescriptor(
                corpus_version=copied["corpus_version"],
                chunker_version=copied["chunker_version"],
                embedding=descriptor,
            )
            reference = FilingReference(
                symbol=copied["symbol"],
                cik=copied["cik"],
                accession_number=copied["accession_number"],
                filing_type=copied["filing_type"],
                title=copied["title"],
                filed_date=date.fromisoformat(copied["filed_date"]),
                source_url=copied["source_url"],
            )
            evidence = EvidenceChunk(
                reference=reference,
                corpus=evidence_corpus,
                generation_id=copied["generation_id"],
                chunk_id=copied["chunk_id"],
                content_hash=copied["content_hash"],
                ordinal=copied["ordinal"],
                text=text,
            )
            hit = SearchHit(
                evidence=evidence,
                score=float(score),
                embedding_descriptor=descriptor,
                active_generation_id=active_manifest.generation_id,
            )
        except (KeyError, TypeError, ValueError):
            return None, True
        if (
            evidence.chunk_id != point_id
            or evidence.corpus != corpus
            or descriptor != corpus.embedding
            or evidence.evidence_digest != copied.get("evidence_digest")
            or evidence.chunk_id
            != EvidenceChunk.canonical_chunk_id(
                evidence.reference,
                evidence.corpus,
                evidence.content_hash,
                evidence.ordinal,
                text=evidence.text,
            )
        ):
            return None, True
        return hit, False

    @staticmethod
    def _quoted(value: str) -> str:
        if not isinstance(value, str) or "'" in value or "\\" in value:
            raise ValueError("vector filter value is invalid")
        return f"'{value}'"

    @classmethod
    def _generation_filter(cls, manifest: GenerationManifest) -> str:
        return " AND ".join(
            (
                f"symbol = {cls._quoted(manifest.symbol)}",
                f"corpus_id = {cls._quoted(manifest.corpus.canonical_key)}",
                f"accession_number = {cls._quoted(manifest.accession_number)}",
                f"generation_id = {cls._quoted(manifest.generation_id)}",
            )
        )

    @classmethod
    def _search_filter(
        cls,
        corpus: CorpusDescriptor,
        symbol: str,
        generation_ids: tuple[str, ...],
    ) -> str:
        if not generation_ids or len(generation_ids) > _MAX_ACTIVE_GENERATIONS:
            raise ValueError("active generation filter is invalid")
        generation_filter = ", ".join(cls._quoted(item) for item in generation_ids)
        return " AND ".join(
            (
                f"symbol = {cls._quoted(symbol)}",
                f"corpus_id = {cls._quoted(corpus.canonical_key)}",
                f"embedding_key = {cls._quoted(corpus.embedding.canonical_key)}",
                f"generation_id IN ({generation_filter})",
            )
        )

    @staticmethod
    def _success(outcome: _WireOutcome) -> bool:
        return bool(
            outcome.kind == "ok"
            and isinstance(outcome.payload, dict)
            and set(outcome.payload) == {"result"}
            and outcome.payload.get("result") == "Success"
        )

    @staticmethod
    def _delete_success(outcome: _WireOutcome) -> bool:
        if (
            outcome.kind != "ok"
            or not isinstance(outcome.payload, dict)
            or set(outcome.payload) != {"result"}
        ):
            return False
        result = outcome.payload.get("result")
        return bool(
            isinstance(result, dict)
            and set(result) == {"deleted"}
            and type(result.get("deleted")) is int
            and int(result["deleted"]) >= 0
        )

    async def _send(
        self,
        method: str,
        path: str,
        payload: object,
        *,
        deadline: RequestDeadline,
    ) -> _WireOutcome:
        if not isinstance(deadline, RequestDeadline):
            raise ValueError("request deadline is invalid")
        if self._client.is_closed:
            raise RuntimeError("the vector service client is closed")
        remaining = deadline.remaining_seconds()
        if remaining <= 0:
            return _WireOutcome("timeout")

        raw: bytes | None = None
        response: httpx.Response | None = None
        failed: Literal["timeout", "unavailable", "invalid"] | None = None
        parsed: object | None = None
        try:
            headers = {
                "Authorization": f"Bearer {self._token.get_secret_value()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            request = self._client.build_request(
                method,
                f"{self._url}{path}/{self._namespace}",
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
                    failed = "unavailable"
                elif response.headers.get("content-type", "").split(";", 1)[0].lower() != (
                    "application/json"
                ):
                    failed = "invalid"
                else:
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            too_large = int(declared) > self._response_max_bytes
                        except ValueError:
                            too_large = True
                        if too_large:
                            failed = "invalid"
                    parts: list[bytes] = []
                    received = 0
                    if failed is None:
                        async for part in response.aiter_bytes():
                            received += len(part)
                            if received > self._response_max_bytes:
                                failed = "invalid"
                                break
                            parts.append(bytes(part))
                        raw = b"".join(parts)
        except (TimeoutError, httpx.TimeoutException):
            failed = "timeout"
        except httpx.HTTPError:
            failed = "unavailable"
        finally:
            if response is not None:
                await response.aclose()

        if failed is None and raw is not None:
            try:
                parsed = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                failed = "invalid"
        raw = None
        response = None
        if failed is not None:
            return _WireOutcome(failed)
        return _WireOutcome("ok", status=200, payload=parsed)

    @staticmethod
    def _raise_outcome(outcome: _WireOutcome) -> None:
        kind = outcome.kind
        outcome = _WireOutcome("invalid")
        if kind == "timeout":
            raise UpstashVectorTimeoutError("the vector service request timed out") from None
        raise UpstashVectorUnavailableError("the vector service is unavailable") from None


__all__ = [
    "UpstashVectorConfigurationError",
    "UpstashVectorStore",
    "UpstashVectorTimeoutError",
    "UpstashVectorUnavailableError",
]
