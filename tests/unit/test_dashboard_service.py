from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Callable
from dataclasses import replace

import pytest

from app.research.control import Reservation, ReservationState
from app.research.deadline import RequestDeadline
from app.research.domain import ResearchAnswer
from app.research.service import ResearchCorpusUnavailableError
from app.services.dashboard import DashboardService
from app.services.research import (
    EnabledResearchService,
    ResearchUnavailableError,
)


def run(coro):
    return asyncio.run(coro)


class ManagedComponent:
    def __init__(self, resource: object) -> None:
        self.resource = resource
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1

    def manages(self, resource: object) -> bool:
        return resource is self.resource


def test_dashboard_closes_and_reports_all_owned_component_resources() -> None:
    market_resource = object()
    symbol_resource = object()
    research_resource = object()
    market = ManagedComponent(market_resource)
    symbols = ManagedComponent(symbol_resource)
    research = ManagedComponent(research_resource)
    dashboard = DashboardService(market, symbols, research)  # type: ignore[arg-type]

    run(dashboard.aclose())

    assert market.close_calls == 1
    assert symbols.close_calls == 1
    assert research.close_calls == 1
    assert dashboard.manages(market_resource)
    assert dashboard.manages(symbol_resource)
    assert dashboard.manages(research_resource)
    assert not dashboard.manages(object())


def test_dashboard_closes_shared_component_only_once() -> None:
    shared = ManagedComponent(object())
    dashboard = DashboardService(shared, shared, shared)  # type: ignore[arg-type]

    run(dashboard.aclose())

    assert shared.close_calls == 1


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ReservationControl:
    def __init__(self, *, commit_result: bool = True) -> None:
        self.commit_result = commit_result
        self.calls: list[tuple[str, object]] = []
        self.reservation = Reservation(
            reservation_digest=digest("reservation"),
            budget_digest=digest("daily-budget"),
            principal_digest=digest("single-owner"),
            units=1,
            state=ReservationState.AUTHORIZED,
        )

    async def authorize_reservation(self, **kwargs: object) -> bool:
        self.calls.append(("reserve", dict(kwargs)))
        self.reservation = kwargs["reservation"]  # type: ignore[assignment]
        return True

    async def commit_reservation(self, **kwargs: object) -> bool:
        self.calls.append(("commit", dict(kwargs)))
        return self.commit_result

    async def release_reservation(self, **kwargs: object) -> bool:
        self.calls.append(("release", dict(kwargs)))
        return True


class IndeterminateAuthorizationControl(ReservationControl):
    def __init__(
        self,
        error: Exception,
        *,
        expire_request: Callable[[], None] | None = None,
        commit_before_error: bool = False,
        release_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.error = error
        self.expire_request = expire_request
        self.commit_before_error = commit_before_error
        self.release_error = release_error
        self.current: Reservation | None = None

    async def authorize_reservation(self, **kwargs: object) -> bool:
        self.calls.append(("reserve", dict(kwargs)))
        reservation = kwargs["reservation"]
        assert isinstance(reservation, Reservation)
        self.reservation = reservation
        self.current = reservation
        if self.commit_before_error:
            self.current = replace(reservation, state=ReservationState.COMMITTED)
        if self.expire_request is not None:
            self.expire_request()
        raise self.error

    async def release_reservation(self, **kwargs: object) -> bool:
        self.calls.append(("release", dict(kwargs)))
        if self.release_error is not None:
            raise self.release_error
        reservation = kwargs["reservation"]
        assert isinstance(reservation, Reservation)
        if self.current != reservation or self.current.state is not ReservationState.AUTHORIZED:
            return False
        self.current = replace(reservation, state=ReservationState.RELEASED)
        return True


class BlockingIndeterminateAuthorizationControl(IndeterminateAuthorizationControl):
    def __init__(self) -> None:
        super().__init__(RuntimeError("private unknown authorization result"))
        self.allow_cleanup = asyncio.Event()
        self.cleanup_completed = asyncio.Event()

    async def release_reservation(self, **kwargs: object) -> bool:
        self.calls.append(("release", dict(kwargs)))
        await self.allow_cleanup.wait()
        reservation = kwargs["reservation"]
        assert isinstance(reservation, Reservation)
        if self.current != reservation or self.current.state is not ReservationState.AUTHORIZED:
            return False
        self.current = replace(reservation, state=ReservationState.RELEASED)
        self.cleanup_completed.set()
        return True


class AppliedThenCancelledAuthorizationControl(ReservationControl):
    def __init__(self) -> None:
        super().__init__()
        self.current: Reservation | None = None
        self.authorization_started = asyncio.Event()
        self.cleanup_completed = asyncio.Event()

    async def authorize_reservation(self, **kwargs: object) -> bool:
        self.calls.append(("reserve", dict(kwargs)))
        reservation = kwargs["reservation"]
        assert isinstance(reservation, Reservation)
        self.reservation = reservation
        self.current = reservation
        self.authorization_started.set()
        await asyncio.Event().wait()
        return True

    async def release_reservation(self, **kwargs: object) -> bool:
        self.calls.append(("release", dict(kwargs)))
        reservation = kwargs["reservation"]
        assert isinstance(reservation, Reservation)
        if self.current != reservation or self.current.state is not ReservationState.AUTHORIZED:
            return False
        self.current = replace(reservation, state=ReservationState.RELEASED)
        self.cleanup_completed.set()
        return True


class QueryCore:
    def __init__(self, *, missing: bool = False) -> None:
        self.missing = missing
        self.events: list[str] = []

    async def query_research(self, symbol: str, question: str, **kwargs: object) -> ResearchAnswer:
        del question
        self.events.append("preflight")
        if self.missing:
            raise ResearchCorpusUnavailableError("research corpus is unavailable")
        callback = kwargs["before_paid_call"]
        assert callable(callback)
        await callback()
        self.events.append("embedding")
        return ResearchAnswer.insufficient(symbol)


@pytest.mark.asyncio
async def test_enabled_research_reserves_and_commits_before_paid_query() -> None:
    control = ReservationControl()
    core = QueryCore()
    service = EnabledResearchService(core=core, control_plane=control, timeout_seconds=7.5)
    deadline = RequestDeadline.after(8)

    reservation = await service.authorize_research_reservation(
        digest("single-owner"),
        daily_limit=100,
        window_seconds=3600,
        deadline=deadline,
    )
    assert reservation is control.reservation
    answer = await service.query_research(
        "AAPL",
        "What risk was disclosed?",
        reservation=reservation,
        deadline=deadline,
    )

    assert answer.insufficient_evidence is True
    assert core.events == ["preflight", "embedding"]
    assert [name for name, _ in control.calls] == ["reserve", "commit"]


@pytest.mark.parametrize(
    "authorization_error",
    [
        TimeoutError("private authorization timeout"),
        RuntimeError("private unknown authorization result"),
    ],
)
@pytest.mark.asyncio
async def test_indeterminate_authorization_is_released_and_sanitized(
    authorization_error: Exception,
) -> None:
    now = [0.0]
    control = IndeterminateAuthorizationControl(
        authorization_error,
        expire_request=lambda: now.__setitem__(0, 8.0),
    )
    service = EnabledResearchService(
        core=QueryCore(),
        control_plane=control,
        timeout_seconds=7.5,
    )

    with pytest.raises(ResearchUnavailableError) as caught:
        await service.authorize_research_reservation(
            digest("single-owner"),
            daily_limit=100,
            window_seconds=3600,
            deadline=RequestDeadline.after(8, clock=lambda: now[0]),
        )

    assert caught.value.args == ("research service is unavailable",)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert str(authorization_error) not in str(caught.value)
    assert [name for name, _ in control.calls] == ["reserve", "release"]
    reservation_call = control.calls[0][1]
    release_call = control.calls[1][1]
    assert isinstance(reservation_call, dict)
    assert isinstance(release_call, dict)
    assert release_call["reservation"] is reservation_call["reservation"]
    cleanup_deadline = release_call["deadline"]
    assert isinstance(cleanup_deadline, RequestDeadline)
    assert 0 < cleanup_deadline.remaining_seconds() <= 0.25
    assert control.current is not None
    assert control.current.state is ReservationState.RELEASED


@pytest.mark.asyncio
async def test_indeterminate_authorization_release_cannot_refund_committed_reservation() -> None:
    control = IndeterminateAuthorizationControl(
        RuntimeError("unknown authorization result"),
        commit_before_error=True,
    )
    service = EnabledResearchService(
        core=QueryCore(),
        control_plane=control,
        timeout_seconds=7.5,
    )

    with pytest.raises(ResearchUnavailableError, match="research service is unavailable"):
        await service.authorize_research_reservation(
            digest("single-owner"),
            daily_limit=100,
            window_seconds=3600,
        )

    assert [name for name, _ in control.calls] == ["reserve", "release"]
    assert control.current is not None
    assert control.current.state is ReservationState.COMMITTED


@pytest.mark.asyncio
async def test_indeterminate_authorization_cleanup_failure_remains_sanitized() -> None:
    authorization_detail = "private unknown authorization result"
    cleanup_detail = "private cleanup timeout"
    control = IndeterminateAuthorizationControl(
        RuntimeError(authorization_detail),
        release_error=TimeoutError(cleanup_detail),
    )
    service = EnabledResearchService(
        core=QueryCore(),
        control_plane=control,
        timeout_seconds=7.5,
    )

    with pytest.raises(ResearchUnavailableError) as caught:
        await service.authorize_research_reservation(
            digest("single-owner"),
            daily_limit=100,
            window_seconds=3600,
        )

    assert caught.value.args == ("research service is unavailable",)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert authorization_detail not in str(caught.value)
    assert cleanup_detail not in str(caught.value)
    assert [name for name, _ in control.calls] == ["reserve", "release"]


@pytest.mark.asyncio
async def test_outer_timeout_waits_for_indeterminate_authorization_cleanup() -> None:
    control = BlockingIndeterminateAuthorizationControl()
    service = EnabledResearchService(
        core=QueryCore(),
        control_plane=control,
        timeout_seconds=7.5,
    )

    async def allow_cleanup_after_outer_timeout() -> None:
        await asyncio.sleep(0.03)
        control.allow_cleanup.set()

    async def invoke_with_outer_timeout() -> None:
        try:
            async with asyncio.timeout(0.01):
                await service.authorize_research_reservation(
                    digest("single-owner"),
                    daily_limit=100,
                    window_seconds=3600,
                )
        except TimeoutError:
            assert control.cleanup_completed.is_set()
            raise

    unblock = asyncio.create_task(allow_cleanup_after_outer_timeout())
    try:
        with pytest.raises(TimeoutError):
            await invoke_with_outer_timeout()
    finally:
        control.allow_cleanup.set()
        await unblock

    assert [name for name, _ in control.calls] == ["reserve", "release"]
    assert control.current is not None
    assert control.current.state is ReservationState.RELEASED


@pytest.mark.asyncio
async def test_outer_timeout_during_authorization_releases_applied_reservation() -> None:
    control = AppliedThenCancelledAuthorizationControl()
    service = EnabledResearchService(
        core=QueryCore(),
        control_plane=control,
        timeout_seconds=7.5,
    )

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await service.authorize_research_reservation(
                digest("single-owner"),
                daily_limit=100,
                window_seconds=3600,
            )

    assert control.authorization_started.is_set()
    assert control.cleanup_completed.is_set()
    assert [name for name, _ in control.calls] == ["reserve", "release"]
    assert control.current is not None
    assert control.current.state is ReservationState.RELEASED


@pytest.mark.asyncio
async def test_failed_reservation_commit_prevents_paid_query() -> None:
    control = ReservationControl(commit_result=False)
    core = QueryCore()
    service = EnabledResearchService(core=core, control_plane=control, timeout_seconds=7.5)

    with pytest.raises(ResearchUnavailableError, match="research service is unavailable"):
        await service.query_research(
            "AAPL",
            "What risk was disclosed?",
            reservation=control.reservation,
        )

    assert core.events == ["preflight"]
    assert [name for name, _ in control.calls] == ["commit"]


@pytest.mark.asyncio
async def test_missing_active_corpus_is_sanitized_and_never_committed() -> None:
    control = ReservationControl()
    core = QueryCore(missing=True)
    service = EnabledResearchService(core=core, control_plane=control, timeout_seconds=7.5)

    with pytest.raises(ResearchUnavailableError, match="research service is unavailable") as error:
        await service.query_research(
            "AAPL",
            "What risk was disclosed?",
            reservation=control.reservation,
        )

    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert core.events == ["preflight"]
    assert control.calls == []


@pytest.mark.asyncio
async def test_research_runtime_rethrow_detaches_question_and_internal_error() -> None:
    question = "private runtime research question"
    internal_detail = "private core failure detail"
    control = ReservationControl()

    class FailingCore:
        async def query_research(self, *_: object, **__: object) -> ResearchAnswer:
            raise RuntimeError(internal_detail)

    service = EnabledResearchService(
        core=FailingCore(),
        control_plane=control,
        timeout_seconds=7.5,
    )

    with pytest.raises(ResearchUnavailableError) as caught:
        await service.query_research(
            "AAPL",
            question,
            reservation=control.reservation,
        )

    rendered = repr(caught.value) + str(caught.value)
    traceback = caught.tb
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("services\\research.py"):
            rendered += "".join(
                repr(value)
                for value in traceback.tb_frame.f_locals.values()
                if not inspect.iscoroutine(value)
            )
        traceback = traceback.tb_next
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert question not in rendered
    assert internal_detail not in rendered


@pytest.mark.asyncio
async def test_release_delegates_to_atomic_commit_safe_control_transition() -> None:
    control = ReservationControl()
    service = EnabledResearchService(
        core=QueryCore(),
        control_plane=control,
        timeout_seconds=7.5,
    )

    released = await service.release_research_reservation(control.reservation)

    assert released is True
    assert [name for name, _ in control.calls] == ["release"]
