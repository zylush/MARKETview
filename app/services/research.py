from __future__ import annotations

import asyncio
import math
import secrets
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol, cast

import httpx

from app.config import Settings
from app.errors import MarketDataError
from app.providers.openai_research import OpenAIAnswerGenerator, OpenAIEmbedder
from app.providers.redis_research import RedisResearchControl
from app.providers.upstash_vector import UpstashVectorStore
from app.research.control import Reservation, ReservationState
from app.research.deadline import RequestDeadline
from app.research.domain import CorpusDescriptor, ResearchAnswer
from app.research.service import (
    ResearchCoreService,
    ResearchCorpusUnavailableError,
    ResearchPolicy,
)

_INDETERMINATE_RELEASE_TIMEOUT_SECONDS = 0.25


class ResearchCostControl(Protocol):
    async def authorize_reservation(
        self,
        *,
        reservation: Reservation,
        limit: int,
        window_seconds: int,
        deadline: RequestDeadline,
    ) -> bool: ...

    async def commit_reservation(
        self,
        *,
        reservation: Reservation,
        deadline: RequestDeadline,
    ) -> bool: ...

    async def release_reservation(
        self,
        *,
        reservation: Reservation,
        deadline: RequestDeadline,
    ) -> bool: ...


class ResearchQueryCore(Protocol):
    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        deadline: RequestDeadline | None = None,
        before_paid_call: Callable[[], Awaitable[None]] | None = None,
    ) -> ResearchAnswer: ...


class ResearchServiceProtocol(Protocol):
    async def authorize_research_reservation(
        self,
        principal_digest: str,
        *,
        daily_limit: int,
        window_seconds: int,
        deadline: RequestDeadline | None = None,
    ) -> Reservation | None: ...

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        reservation: Reservation | None = None,
        deadline: RequestDeadline | None = None,
    ) -> ResearchAnswer: ...

    async def release_research_reservation(
        self,
        reservation: Reservation,
        *,
        deadline: RequestDeadline | None = None,
    ) -> bool: ...


class ResearchUnavailableError(MarketDataError):
    code = "research_unavailable"
    status_code = 503


class DisabledResearchService:
    """Fail closed without constructing any research provider or HTTP client."""

    async def authorize_research_reservation(
        self,
        principal_digest: str,
        *,
        daily_limit: int,
        window_seconds: int,
        deadline: RequestDeadline | None = None,
    ) -> Reservation | None:
        del principal_digest, daily_limit, window_seconds, deadline
        raise ResearchUnavailableError("research is not configured")

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        reservation: Reservation | None = None,
        deadline: RequestDeadline | None = None,
    ) -> ResearchAnswer:
        del symbol, question, reservation, deadline
        raise ResearchUnavailableError("research is not configured")

    async def release_research_reservation(
        self,
        reservation: Reservation,
        *,
        deadline: RequestDeadline | None = None,
    ) -> bool:
        del reservation, deadline
        return False


class EnabledResearchService:
    """Query-only production façade with commit-before-paid-call accounting."""

    def __init__(
        self,
        *,
        core: ResearchQueryCore,
        control_plane: ResearchCostControl,
        timeout_seconds: float,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or not 0 < float(timeout_seconds) < 8
        ):
            raise ValueError("research runtime timeout must be below eight seconds")
        self._core = core
        self._control_plane = control_plane
        self._timeout_seconds = float(timeout_seconds)

    def _deadline(self, value: RequestDeadline | None) -> RequestDeadline:
        if value is not None:
            if not isinstance(value, RequestDeadline):
                raise ValueError("research request deadline is invalid")
            value.raise_if_expired()
            return value.child(self._timeout_seconds)
        return RequestDeadline.after(self._timeout_seconds)

    async def _release_indeterminate_reservation(self, reservation: Reservation) -> None:
        """Best-effort cleanup after authorization loses its authoritative reply."""

        cleanup = asyncio.create_task(
            self._control_plane.release_reservation(
                reservation=reservation,
                deadline=RequestDeadline.after(
                    min(_INDETERMINATE_RELEASE_TIMEOUT_SECONDS, self._timeout_seconds)
                ),
            )
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            with suppress(Exception):
                await cleanup
            raise
        except (RuntimeError, TimeoutError):
            return

    async def authorize_research_reservation(
        self,
        principal_digest: str,
        *,
        daily_limit: int,
        window_seconds: int,
        deadline: RequestDeadline | None = None,
    ) -> Reservation | None:
        operation_deadline = self._deadline(deadline)
        today = datetime.now(UTC).date().isoformat()
        reservation = Reservation(
            reservation_digest=sha256(secrets.token_bytes(32)).hexdigest(),
            budget_digest=sha256(f"research:v1:daily:{today}".encode()).hexdigest(),
            principal_digest=principal_digest,
            units=1,
            state=ReservationState.AUTHORIZED,
        )
        failed = False
        authorized = False
        try:
            authorized = await self._control_plane.authorize_reservation(
                reservation=reservation,
                limit=daily_limit,
                window_seconds=window_seconds,
                deadline=operation_deadline,
            )
        except asyncio.CancelledError:
            await self._release_indeterminate_reservation(reservation)
            raise
        except (RuntimeError, TimeoutError):
            failed = True
        if failed:
            # The atomic release compares the exact authorized record, so a concurrent
            # commit wins and remains charged while an applied authorization is refunded.
            await self._release_indeterminate_reservation(reservation)
            raise ResearchUnavailableError("research service is unavailable") from None
        return reservation if authorized else None

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        reservation: Reservation | None = None,
        deadline: RequestDeadline | None = None,
    ) -> ResearchAnswer:
        if (
            not isinstance(reservation, Reservation)
            or reservation.state is not ReservationState.AUTHORIZED
        ):
            question = ""
            symbol = ""
            raise ResearchUnavailableError("research service is unavailable")
        operation_deadline = self._deadline(deadline)

        async def commit_before_paid_call() -> None:
            committed = await self._control_plane.commit_reservation(
                reservation=reservation,
                deadline=operation_deadline,
            )
            if not committed:
                raise ResearchUnavailableError("research service is unavailable")

        failed = False
        answer: ResearchAnswer | None = None
        try:
            answer = await self._core.query_research(
                symbol,
                question,
                deadline=operation_deadline,
                before_paid_call=commit_before_paid_call,
            )
        except (ResearchCorpusUnavailableError, ResearchUnavailableError, RuntimeError):
            failed = True
        if failed or not isinstance(answer, ResearchAnswer):
            answer = None
            question = ""
            symbol = ""
            raise ResearchUnavailableError("research service is unavailable") from None
        return answer

    async def release_research_reservation(
        self,
        reservation: Reservation,
        *,
        deadline: RequestDeadline | None = None,
    ) -> bool:
        operation_deadline = self._deadline(deadline)
        failed = False
        released = False
        try:
            released = await self._control_plane.release_reservation(
                reservation=reservation,
                deadline=operation_deadline,
            )
        except RuntimeError:
            failed = True
        return False if failed else released


class ResearchRuntime(EnabledResearchService):
    """Enabled service plus explicit ownership of its shared network resources."""

    def __init__(
        self,
        *,
        core: ResearchQueryCore,
        control_plane: ResearchCostControl,
        timeout_seconds: float,
        components: tuple[object, ...] = (),
        owned_resources: tuple[object, ...] = (),
    ) -> None:
        super().__init__(
            core=core,
            control_plane=control_plane,
            timeout_seconds=timeout_seconds,
        )
        self._components = self._deduplicated(components)
        self._owned_resources = self._deduplicated(owned_resources)
        self._closed = False

    @staticmethod
    def _deduplicated(resources: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(
            resource
            for index, resource in enumerate(resources)
            if all(resource is not previous for previous in resources[:index])
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in (*self._components, *self._owned_resources):
            closer = getattr(resource, "aclose", None)
            if closer is not None:
                await closer()

    def manages(self, resource: object) -> bool:
        if any(resource is owned for owned in self._owned_resources):
            return True
        return any(
            bool(manages and manages(resource))
            for component in self._components
            if (manages := getattr(component, "manages", None)) is not None
        )


def _internal_timeout(outer_seconds: float) -> float:
    outer = float(outer_seconds)
    reserve = min(0.25, outer / 10)
    return min(7.5, outer - reserve)


def build_research_runtime(
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
) -> ResearchRuntime:
    """Construct the query-only RAG graph without performing network I/O."""

    if not isinstance(settings, Settings) or not settings.research_enabled:
        raise ValueError("enabled research settings are required")
    if (
        settings.openai_api_key is None
        or settings.upstash_vector_rest_url is None
        or settings.upstash_vector_rest_token is None
        or settings.upstash_redis_rest_url is None
        or settings.upstash_redis_rest_token is None
    ):
        raise ValueError("research runtime configuration is incomplete")
    timeout_seconds = _internal_timeout(settings.research_timeout_seconds)
    policy = ResearchPolicy(
        minimum_score=float(settings.research_minimum_score),
        max_results=settings.research_max_results,
        overfetch_factor=settings.research_vector_overfetch,
        request_timeout_seconds=timeout_seconds,
    )
    owns_client = client is None
    shared_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(settings.research_timeout_seconds),
        follow_redirects=False,
        trust_env=False,
    )
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        client=shared_client,
    )
    generator = OpenAIAnswerGenerator(
        api_key=settings.openai_api_key,
        client=shared_client,
        max_output_tokens=settings.research_generation_max_output_tokens,
    )
    vector_store = UpstashVectorStore(
        settings.upstash_vector_rest_url,
        settings.upstash_vector_rest_token,
        namespace=settings.research_vector_namespace,
        client=shared_client,
    )
    control_plane = RedisResearchControl(
        settings.upstash_redis_rest_url,
        settings.upstash_redis_rest_token,
        client=shared_client,
        timeout_seconds=min(3.0, timeout_seconds),
    )
    corpus = CorpusDescriptor(
        corpus_version=settings.research_index_schema_version,
        chunker_version=(
            f"tokens-{settings.research_chunk_tokens}-{settings.research_chunk_overlap_tokens}-v1"
        ),
        embedding=embedder.descriptor,
    )
    core = ResearchCoreService(
        corpus=corpus,
        chunker=None,
        embedder=embedder,
        store=vector_store,
        control_plane=control_plane,
        generator=generator,
        policy=policy,
    )
    components: tuple[object, ...] = (
        embedder,
        generator,
        vector_store,
        control_plane,
    )
    return ResearchRuntime(
        core=core,
        control_plane=cast(ResearchCostControl, control_plane),
        timeout_seconds=timeout_seconds,
        components=components,
        owned_resources=(shared_client,) if owns_client else (),
    )


__all__ = [
    "DisabledResearchService",
    "EnabledResearchService",
    "ResearchCoreService",
    "ResearchPolicy",
    "ResearchRuntime",
    "ResearchServiceProtocol",
    "ResearchUnavailableError",
    "build_research_runtime",
]
