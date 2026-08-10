from __future__ import annotations

import asyncio
import math
import secrets
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from app.providers.openai_research import OpenAIEmbedder
from app.providers.redis_research import RedisResearchControl
from app.providers.upstash_vector import UpstashVectorStore
from app.research.control import Reservation, ReservationState
from app.research.deadline import RequestDeadline
from app.research.domain import CorpusDescriptor, EmbeddingVector, GenerationManifest
from app.research.ports import GenerationInspectionState
from app.research.retrieval import SafeHitClassification, classify_safe_hits

_SYMBOL = "AAPL"
_FILING_TYPE = "10-K"
_QUESTION = "What material risks did Apple disclose in its latest Form 10-K?"
_DIAGNOSTIC_CASE = "aapl_latest_10k_risks_v1"
_PRINCIPAL_DIGEST = sha256(f"operator:{_DIAGNOSTIC_CASE}".encode()).hexdigest()
_UPSTASH_HOST_SUFFIX = ".upstash.io"


class _SettingsLike(Protocol):
    openai_api_key: SecretStr | None
    upstash_vector_rest_url: str | None
    upstash_vector_rest_token: SecretStr | None
    upstash_redis_rest_url: str | None
    upstash_redis_rest_token: SecretStr | None
    research_embedding_provider: str
    research_embedding_model: str
    research_embedding_dimensions: int
    research_vector_provider: str
    research_vector_namespace: str
    research_index_schema_version: str
    research_chunk_tokens: int
    research_chunk_overlap_tokens: int
    research_max_results: int
    research_vector_overfetch: int
    research_minimum_score: float
    research_daily_global_limit: int


@dataclass(frozen=True, slots=True)
class DiagnosticPreflight:
    active_generation_count: int
    pending_cleanup_count: int
    inspection_state: str | None
    expected_point_count: int | None
    observed_point_count: int | None
    private_manifest: GenerationManifest | None = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.active_generation_count) is not int or self.active_generation_count < 0:
            raise ValueError("diagnostic active-generation count is invalid")
        if type(self.pending_cleanup_count) is not int or self.pending_cleanup_count < 0:
            raise ValueError("diagnostic pending-cleanup count is invalid")
        if self.inspection_state is not None and (
            not isinstance(self.inspection_state, str)
            or not self.inspection_state
            or len(self.inspection_state) > 32
        ):
            raise ValueError("diagnostic inspection state is invalid")
        counts = (self.expected_point_count, self.observed_point_count)
        if (counts[0] is None) != (counts[1] is None):
            raise ValueError("diagnostic inspection counts are incomplete")
        if counts[0] is not None and any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("diagnostic inspection count is invalid")
        if self.private_manifest is not None and not isinstance(
            self.private_manifest, GenerationManifest
        ):
            raise ValueError("diagnostic private manifest is invalid")

    @property
    def is_exact(self) -> bool:
        return bool(
            self.active_generation_count == 1
            and self.pending_cleanup_count == 0
            and self.inspection_state == GenerationInspectionState.EXACT.value
            and self.expected_point_count is not None
            and self.expected_point_count > 0
            and self.observed_point_count == self.expected_point_count
            and self.private_manifest is not None
        )


class DiagnosticRuntime(Protocol):
    async def preflight(
        self,
        *,
        symbol: str,
        filing_type: str,
        deadline: RequestDeadline,
    ) -> DiagnosticPreflight: ...

    async def authorize(self, *, deadline: RequestDeadline) -> Reservation | None: ...

    async def revalidate(
        self,
        *,
        preflight: DiagnosticPreflight,
        deadline: RequestDeadline,
    ) -> bool: ...

    async def commit(
        self,
        *,
        reservation: Reservation,
        deadline: RequestDeadline,
    ) -> bool: ...

    async def embed(self, *, text: str, deadline: RequestDeadline) -> EmbeddingVector: ...

    async def search(
        self,
        *,
        symbol: str,
        filing_type: str,
        vector: EmbeddingVector,
        preflight: DiagnosticPreflight,
        deadline: RequestDeadline,
    ) -> object: ...

    async def release(
        self,
        *,
        reservation: Reservation,
        deadline: RequestDeadline,
    ) -> bool: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _Configuration:
    requested_candidate_limit: int
    minimum_score: float


def _secret_is_present(value: object) -> bool:
    return isinstance(value, SecretStr) and bool(value.get_secret_value())


def _root_https_endpoint(value: object, *, require_upstash_host: bool) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        return False
    hostname = parsed.hostname
    approved_host = bool(
        hostname
        and hostname.endswith(_UPSTASH_HOST_SUFFIX)
        and hostname != _UPSTASH_HOST_SUFFIX[1:]
    )
    return bool(
        parsed.scheme == "https"
        and hostname
        and (approved_host or not require_upstash_host)
        and parsed.netloc == hostname
        and parsed.username is None
        and parsed.password is None
        and port is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def _configuration(settings: _SettingsLike) -> _Configuration:
    if not _root_https_endpoint(
        settings.upstash_vector_rest_url, require_upstash_host=True
    ) or not _root_https_endpoint(settings.upstash_redis_rest_url, require_upstash_host=False):
        raise ValueError("retrieval diagnostic endpoint configuration is invalid")
    if not all(
        _secret_is_present(value)
        for value in (
            settings.openai_api_key,
            settings.upstash_vector_rest_token,
            settings.upstash_redis_rest_token,
        )
    ):
        raise ValueError("retrieval diagnostic credential configuration is incomplete")
    expected = (
        (settings.research_embedding_provider, "openai"),
        (settings.research_embedding_model, "text-embedding-3-small"),
        (settings.research_embedding_dimensions, 1536),
        (settings.research_vector_provider, "upstash"),
    )
    if any(actual != required for actual, required in expected):
        raise ValueError("retrieval diagnostic provider configuration is unsupported")
    if (
        not isinstance(settings.research_vector_namespace, str)
        or not settings.research_vector_namespace
        or not isinstance(settings.research_index_schema_version, str)
        or not settings.research_index_schema_version
        or type(settings.research_chunk_tokens) is not int
        or type(settings.research_chunk_overlap_tokens) is not int
        or not 0 <= settings.research_chunk_overlap_tokens < settings.research_chunk_tokens
        or type(settings.research_max_results) is not int
        or not 1 <= settings.research_max_results <= 20
        or type(settings.research_vector_overfetch) is not int
        or not 1 <= settings.research_vector_overfetch <= 10
        or settings.research_max_results * settings.research_vector_overfetch > 20
        or type(settings.research_daily_global_limit) is not int
        or settings.research_daily_global_limit < 1
    ):
        raise ValueError("retrieval diagnostic configuration is invalid")
    minimum_score = float(settings.research_minimum_score)
    if not math.isfinite(minimum_score) or not 0.0 <= minimum_score <= 1.0:
        raise ValueError("retrieval diagnostic score configuration is invalid")
    return _Configuration(
        requested_candidate_limit=(
            settings.research_max_results * settings.research_vector_overfetch
        ),
        minimum_score=minimum_score,
    )


def _daily_window_seconds(now: datetime | None = None) -> int:
    current = now or datetime.now(UTC)
    next_day = datetime.combine(
        current.date() + timedelta(days=1),
        datetime.min.time(),
        UTC,
    )
    return max(1, int((next_day - current).total_seconds()))


class RetrievalDiagnosticRuntime:
    """Private provider graph for the fixed retrieval-only operator diagnostic."""

    def __init__(
        self,
        *,
        corpus: CorpusDescriptor,
        embedder: OpenAIEmbedder,
        store: UpstashVectorStore,
        control: RedisResearchControl,
        client: httpx.AsyncClient,
        maximum_results: int,
        overfetch_factor: int,
        minimum_score: float,
        daily_limit: int,
    ) -> None:
        self._corpus = corpus
        self._embedder = embedder
        self._store = store
        self._control = control
        self._client = client
        self._maximum_results = maximum_results
        self._candidate_limit = maximum_results * overfetch_factor
        self._minimum_score = minimum_score
        self._daily_limit = daily_limit

    @staticmethod
    def _validate_fixed_case(symbol: str, filing_type: str) -> None:
        if symbol != _SYMBOL or filing_type != _FILING_TYPE:
            raise ValueError("retrieval diagnostic case is fixed")

    async def preflight(
        self,
        *,
        symbol: str,
        filing_type: str,
        deadline: RequestDeadline,
    ) -> DiagnosticPreflight:
        self._validate_fixed_case(symbol, filing_type)
        active = await self._control.list_active_generations(
            corpus=self._corpus,
            symbol=symbol,
            deadline=deadline,
        )
        pending = await self._control.list_pending_cleanups(
            corpus=self._corpus,
            symbol=symbol,
            deadline=deadline,
        )
        selected = active[0] if len(active) == 1 else None
        if selected is None or pending:
            return DiagnosticPreflight(
                active_generation_count=len(active),
                pending_cleanup_count=len(pending),
                inspection_state=None,
                expected_point_count=None,
                observed_point_count=None,
                private_manifest=selected,
            )
        inspection = await self._store.inspect_generation(selected, deadline=deadline)
        return DiagnosticPreflight(
            active_generation_count=1,
            pending_cleanup_count=0,
            inspection_state=inspection.state.value,
            expected_point_count=inspection.expected_point_count,
            observed_point_count=inspection.observed_point_count,
            private_manifest=selected,
        )

    async def authorize(self, *, deadline: RequestDeadline) -> Reservation | None:
        today = datetime.now(UTC).date().isoformat()
        reservation = Reservation(
            reservation_digest=sha256(secrets.token_bytes(32)).hexdigest(),
            budget_digest=sha256(f"research:v1:daily:{today}".encode()).hexdigest(),
            principal_digest=_PRINCIPAL_DIGEST,
            units=1,
            state=ReservationState.AUTHORIZED,
        )
        try:
            authorized = await self._control.authorize_reservation(
                reservation=reservation,
                limit=self._daily_limit,
                window_seconds=_daily_window_seconds(),
                deadline=deadline,
            )
        except Exception:
            with suppress(Exception):
                await self._control.release_reservation(
                    reservation=reservation,
                    deadline=RequestDeadline.after(0.25),
                )
            raise RuntimeError("retrieval diagnostic authorization failed") from None
        return reservation if authorized else None

    async def revalidate(
        self,
        *,
        preflight: DiagnosticPreflight,
        deadline: RequestDeadline,
    ) -> bool:
        current = await self.preflight(
            symbol=_SYMBOL,
            filing_type=_FILING_TYPE,
            deadline=deadline,
        )
        return current.is_exact and current == preflight

    async def commit(
        self,
        *,
        reservation: Reservation,
        deadline: RequestDeadline,
    ) -> bool:
        return await self._control.commit_reservation(
            reservation=reservation,
            deadline=deadline,
        )

    async def embed(self, *, text: str, deadline: RequestDeadline) -> EmbeddingVector:
        if text != _QUESTION:
            raise ValueError("retrieval diagnostic question is fixed")
        return await self._embedder.embed_query(text, deadline=deadline)

    async def search(
        self,
        *,
        symbol: str,
        filing_type: str,
        vector: EmbeddingVector,
        preflight: DiagnosticPreflight,
        deadline: RequestDeadline,
    ) -> SafeHitClassification:
        self._validate_fixed_case(symbol, filing_type)
        manifest = preflight.private_manifest
        if not preflight.is_exact or manifest is None:
            raise ValueError("retrieval diagnostic preflight is not exact")
        hits = await self._store.search(
            corpus=self._corpus,
            symbol=symbol,
            vector=vector,
            active_generations=(manifest,),
            limit=self._candidate_limit,
            deadline=deadline,
        )
        return classify_safe_hits(
            symbol,
            self._corpus,
            (manifest,),
            hits,
            minimum_score=self._minimum_score,
            max_results=self._maximum_results,
        )

    async def release(
        self,
        *,
        reservation: Reservation,
        deadline: RequestDeadline,
    ) -> bool:
        return await self._control.release_reservation(
            reservation=reservation,
            deadline=deadline,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def build_retrieval_diagnostic_runtime(settings: _SettingsLike) -> RetrievalDiagnosticRuntime:
    _configuration(settings)
    if (
        settings.openai_api_key is None
        or settings.upstash_vector_rest_url is None
        or settings.upstash_vector_rest_token is None
        or settings.upstash_redis_rest_url is None
        or settings.upstash_redis_rest_token is None
    ):
        raise ValueError("retrieval diagnostic configuration is incomplete")
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0),
        follow_redirects=False,
        trust_env=False,
    )
    try:
        embedder = OpenAIEmbedder(api_key=settings.openai_api_key, client=client)
        store = UpstashVectorStore(
            settings.upstash_vector_rest_url,
            settings.upstash_vector_rest_token,
            namespace=settings.research_vector_namespace,
            client=client,
        )
        control = RedisResearchControl(
            settings.upstash_redis_rest_url,
            settings.upstash_redis_rest_token,
            client=client,
            timeout_seconds=3.0,
        )
        corpus = CorpusDescriptor(
            corpus_version=settings.research_index_schema_version,
            chunker_version=(
                f"tokens-{settings.research_chunk_tokens}-"
                f"{settings.research_chunk_overlap_tokens}-v1"
            ),
            embedding=embedder.descriptor,
        )
        return RetrievalDiagnosticRuntime(
            corpus=corpus,
            embedder=embedder,
            store=store,
            control=control,
            client=client,
            maximum_results=settings.research_max_results,
            overfetch_factor=settings.research_vector_overfetch,
            minimum_score=float(settings.research_minimum_score),
            daily_limit=settings.research_daily_global_limit,
        )
    except Exception:
        with suppress(Exception):
            asyncio.get_running_loop().create_task(client.aclose())
        raise


__all__ = [
    "DiagnosticPreflight",
    "DiagnosticRuntime",
    "RetrievalDiagnosticRuntime",
    "build_retrieval_diagnostic_runtime",
]
