from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

import app.main as main_module
from app.cache.memory import MemoryCache
from app.errors import (
    CacheUnavailableError,
    ProviderAccessRestrictedError,
    ProviderAuthenticationError,
    ProviderNotFoundError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from app.main import create_app
from app.providers.marketdata import MarketDataAppProvider
from app.research.control import Reservation, ReservationState
from app.research.deadline import RequestDeadline
from app.research.domain import ResearchAnswer
from app.services.dashboard import DashboardService
from app.services.market_data import MarketDataService
from app.services.research import DisabledResearchService, EnabledResearchService
from app.services.symbols import SymbolSearchService


async def _csrf(client: httpx.AsyncClient) -> str:
    page = await client.get("/")
    assert page.status_code == 200
    match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert match is not None
    return match.group(1)


async def _login_and_dashboard_csrf(client: httpx.AsyncClient) -> str:
    login_csrf = await _csrf(client)
    login = await client.post(
        "/login",
        data={"password": "correct horse battery staple", "csrf_token": login_csrf},
        headers={"Origin": "https://dashboard.test"},
        follow_redirects=False,
    )
    assert login.status_code == 303
    dashboard = await client.get("/dashboard")
    csrf_cookie = dashboard.headers["set-cookie"].lower()
    assert "marketdata_csrf=" in csrf_cookie
    assert "httponly" in csrf_cookie
    assert "samesite=strict" in csrf_cookie
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', dashboard.text)
    assert match is not None
    return match.group(1)


@pytest.mark.asyncio
async def test_health_is_public_local_and_has_security_headers(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["data"] == {"status": "ok"}
    assert response.json()["meta"]["pagination"] is None
    assert response.headers["x-request-id"] == response.json()["meta"]["request_id"]
    assert response.headers["content-security-policy"].startswith("default-src 'self'")
    assert "strict-transport-security" not in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "same-origin"


def test_application_uses_provider_neutral_title(settings, service, cache) -> None:
    assert create_app(settings=settings, service=service, cache=cache).title == (
        "Market Data Dashboard API"
    )


def test_secret_session_setting_is_unwrapped_only_for_signer_construction(
    settings,
    service,
    cache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_secret = "session-secret-visible-only-to-signer"
    captured_secret: str | None = None

    class RecordingSigner:
        def __init__(self, secret: str, *, max_age_seconds: int) -> None:
            nonlocal captured_secret
            del max_age_seconds
            captured_secret = secret

    monkeypatch.setattr(main_module, "SessionSigner", RecordingSigner)
    protected = replace(settings, session_secret=SecretStr(session_secret))

    application = create_app(settings=protected, service=service, cache=cache)

    assert application.state.settings is protected
    assert captured_secret == session_secret
    assert session_secret not in repr(application.state.settings)


@pytest.mark.asyncio
async def test_session_secret_is_absent_from_responses_and_logs(
    settings,
    cache,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_secret = "session-secret-must-never-leak"

    class FailingService:
        async def latest_eod(self, symbol: str):
            del symbol
            raise ProviderUnavailableError(f"failure containing {session_secret}")

    protected = replace(settings, session_secret=SecretStr(session_secret))
    caplog.set_level(logging.WARNING, logger="app.main")
    transport = httpx.ASGITransport(
        app=create_app(settings=protected, service=FailingService(), cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=protected.allowed_origin) as local:
        token = await _csrf(local)
        login = await local.post(
            "/login",
            data={"password": "correct horse battery staple", "csrf_token": token},
            headers={"Origin": protected.allowed_origin},
            follow_redirects=False,
        )
        response = await local.get("/api/v1/eod/latest/AAPL")

    assert login.status_code == 303
    assert response.status_code == 502
    combined = f"{login.text}{login.headers}{response.text}{response.headers}{caplog.text}"
    assert session_secret not in combined


@pytest.mark.asyncio
async def test_unsupported_market_data_routes_are_absent_without_service_calls(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    paths = (
        "/api/v1/tickers",
        "/api/v1/exchanges",
        "/api/v1/splits/AAPL",
        "/api/v1/dividends/AAPL",
    )

    responses = [await client.get(path, headers=api_headers) for path in paths]
    schema = (await client.get("/openapi.json", headers=api_headers)).json()

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
    assert not (
        {
            "/api/v1/tickers",
            "/api/v1/exchanges",
            "/api/v1/splits/{symbol}",
            "/api/v1/dividends/{symbol}",
        }
        & set(schema["paths"])
    )
    assert service.calls == []


@pytest.mark.asyncio
async def test_symbol_search_is_authenticated_validated_ranked_and_enveloped(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    unauthorized = await client.get("/api/v1/symbols/search?q=ap")
    too_short = await client.get("/api/v1/symbols/search?q=a&limit=8", headers=api_headers)
    invalid_limit = await client.get("/api/v1/symbols/search?q=app&limit=9", headers=api_headers)
    invalid_characters = await client.get(
        "/api/v1/symbols/search?q=ap%3Cscript%3E", headers=api_headers
    )
    response = await client.get("/api/v1/symbols/search?q=app&limit=2", headers=api_headers)

    assert unauthorized.status_code == 401
    assert too_short.status_code == 422
    assert invalid_limit.status_code == 422
    assert invalid_characters.status_code == 422
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"] == [
        {"symbol": "APP", "name": "Applovin Corporation", "exchange": "Nasdaq"},
        {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "Nasdaq"},
    ]
    assert payload["meta"]["pagination"] == {"next_cursor": None, "total": 2, "limit": 2}
    assert payload["meta"]["source"] == "test-fixture"
    assert service.calls == [("search_symbols", {"query": "APP", "limit": 2})]


@pytest.mark.asyncio
async def test_symbol_search_has_a_separate_rate_limit(
    settings,
    service,
    cache,
) -> None:
    isolated = SimpleNamespace(**{**vars(settings), "symbol_search_rate_limit": 1})
    transport = httpx.ASGITransport(
        app=create_app(settings=isolated, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=isolated.allowed_origin) as local:
        first = await local.get(
            "/api/v1/symbols/search?q=ap&limit=2",
            headers={"X-App-Key": "correct horse battery staple"},
        )
        limited = await local.get(
            "/api/v1/symbols/search?q=ms&limit=2",
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert first.status_code == 200
    assert limited.status_code == 429
    assert cache.counts["rate:symbol-search:127.0.0.1"] == 2
    assert all("marketdata" not in key and "quota" not in key for key in cache.counts)
    assert service.calls == [("search_symbols", {"query": "AP", "limit": 2})]


@pytest.mark.asyncio
async def test_unseeded_symbol_directory_returns_specific_sanitized_503(
    settings,
    cache,
) -> None:
    symbol_search = SymbolSearchService(None, MemoryCache())
    service = SimpleNamespace(search_symbols=symbol_search.search_symbols)
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.get(
            "/api/v1/symbols/search?q=ap&limit=8",
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "SYMBOL_DIRECTORY_UNAVAILABLE",
        "message": "the symbol directory is unavailable",
    }


class _AtomicResearchControl:
    def __init__(
        self,
        *,
        timeline: list[str],
        exhausted: bool = False,
        unavailable: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.timeline = timeline
        self.exhausted = exhausted
        self.unavailable = unavailable
        self.current: Reservation | None = None
        self.principal_digests: list[str] = []
        self.reservations: list[Reservation] = []
        self.authorized_units = 0

    async def authorize_reservation(
        self,
        *,
        reservation: Reservation,
        limit: int,
        window_seconds: int,
        deadline: RequestDeadline,
    ) -> bool:
        del window_seconds, deadline
        self.events.append("authorize")
        self.timeline.append("authorize")
        self.principal_digests.append(reservation.principal_digest)
        self.reservations.append(reservation)
        if self.unavailable:
            raise RuntimeError("private redis control detail")
        if self.exhausted or self.authorized_units + reservation.units > limit:
            return False
        self.authorized_units += reservation.units
        self.current = reservation
        return True

    async def commit_reservation(
        self,
        *,
        reservation: Reservation,
        deadline: RequestDeadline,
    ) -> bool:
        del deadline
        self.events.append("commit")
        self.timeline.append("commit")
        if self.current != reservation or self.current.state is not ReservationState.AUTHORIZED:
            return False
        self.current = replace(reservation, state=ReservationState.COMMITTED)
        return True

    async def release_reservation(
        self,
        *,
        reservation: Reservation,
        deadline: RequestDeadline,
    ) -> bool:
        del deadline
        if self.current == reservation and self.current.state is ReservationState.AUTHORIZED:
            self.events.append("release_authorized")
            self.timeline.append("release_authorized")
            self.authorized_units = max(0, self.authorized_units - reservation.units)
            self.current = replace(reservation, state=ReservationState.RELEASED)
            return True
        self.events.append("retain_committed")
        self.timeline.append("retain_committed")
        return False


class _AtomicResearchCore:
    def __init__(
        self,
        behavior: str,
        *,
        timeline: list[str],
        error_detail: str | None = None,
    ) -> None:
        self.behavior = behavior
        self.error_detail = error_detail
        self.events: list[str] = []
        self.timeline = timeline
        self.query_started = asyncio.Event()
        self.paid_call_started = asyncio.Event()

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        deadline: RequestDeadline | None = None,
        before_paid_call: Callable[[], Awaitable[None]] | None = None,
    ) -> ResearchAnswer:
        del question, deadline
        self.events.append("query")
        self.timeline.append("query")
        self.query_started.set()
        if self.behavior == "refuse":
            return ResearchAnswer.refusal(symbol)
        if self.behavior == "fail_precommit":
            raise RuntimeError("private pre-commit failure")
        if self.behavior == "cancel_precommit":
            await asyncio.Event().wait()
        assert before_paid_call is not None
        await before_paid_call()
        self.events.append("embed")
        self.timeline.append("embed")
        self.paid_call_started.set()
        if self.behavior == "malformed":
            return {
                "answer": "unsupported",
                "citations": [{"url": self.error_detail}],
            }  # type: ignore[return-value]
        if self.behavior == "cross_symbol":
            return ResearchAnswer.insufficient("MSFT")
        if self.behavior in {"provider_error", "vector_error", "generation_error"}:
            raise RuntimeError(self.error_detail or f"private {self.behavior}")
        if self.behavior in {"timeout", "cancel"}:
            await asyncio.Event().wait()
        return ResearchAnswer.insufficient(symbol)


def _atomic_research_service(
    behavior: str,
    *,
    exhausted: bool = False,
    unavailable: bool = False,
    error_detail: str | None = None,
) -> tuple[EnabledResearchService, _AtomicResearchCore, _AtomicResearchControl]:
    timeline: list[str] = []
    core = _AtomicResearchCore(behavior, timeline=timeline, error_detail=error_detail)
    control = _AtomicResearchControl(
        timeline=timeline,
        exhausted=exhausted,
        unavailable=unavailable,
    )
    service = EnabledResearchService(
        core=core,
        control_plane=control,
        timeout_seconds=0.5,
    )
    return service, core, control


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("behavior", "expected_status"),
    [("refuse", 200), ("fail_precommit", 503)],
)
async def test_research_precommit_outcomes_release_authorized_spend(
    settings,
    cache,
    behavior: str,
    expected_status: int,
) -> None:
    service, core, control = _atomic_research_service(behavior)
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "Should I buy this stock?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == expected_status
    assert control.current is not None
    assert control.current.state is ReservationState.RELEASED
    assert control.events == ["authorize", "release_authorized"]
    assert core.events == ["query"]
    assert all(
        "provider_quota" not in key and not key.startswith("marketdata:") for key in cache.counts
    )


@pytest.mark.asyncio
async def test_research_commits_before_first_embedding_and_never_refunds_commit(
    settings,
    cache,
) -> None:
    service, core, control = _atomic_research_service("success")
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 200
    assert control.current is not None
    assert control.current.state is ReservationState.COMMITTED
    assert control.events == ["authorize", "commit", "retain_committed"]
    assert core.events == ["query", "embed"]
    assert control.timeline == ["authorize", "query", "commit", "embed", "retain_committed"]
    assert control.principal_digests == [hashlib.sha256(b"127.0.0.1").hexdigest()]
    assert re.fullmatch(r"[0-9a-f]{64}", control.principal_digests[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["provider_error", "vector_error", "generation_error"])
async def test_research_committed_downstream_errors_retain_charge(
    settings,
    cache,
    behavior: str,
) -> None:
    service, core, control = _atomic_research_service(behavior)
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "RESEARCH_UNAVAILABLE",
        "message": "research is not configured",
    }
    assert core.events == ["query", "embed"]
    assert control.current is not None
    assert control.current.state is ReservationState.COMMITTED
    assert control.events[-1] == "retain_committed"


@pytest.mark.asyncio
async def test_research_timeout_after_commit_retains_charge(settings, cache) -> None:
    service, _, control = _atomic_research_service("timeout")
    isolated = SimpleNamespace(**{**vars(settings), "research_timeout_seconds": 0.01})
    transport = httpx.ASGITransport(
        app=create_app(settings=isolated, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=isolated.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 504
    assert control.current is not None
    assert control.current.state is ReservationState.COMMITTED
    assert control.events[-1] == "retain_committed"


@pytest.mark.asyncio
async def test_research_client_cancellation_after_commit_retains_charge(settings, cache) -> None:
    service, core, control = _atomic_research_service("cancel")
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=True,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        request = asyncio.create_task(
            local.post(
                "/api/v1/research/query",
                json={"symbol": "AAPL", "question": "What risks are described?"},
                headers={"X-App-Key": "correct horse battery staple"},
            )
        )
        await asyncio.wait_for(core.paid_call_started.wait(), timeout=1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.sleep(0)

    assert control.current is not None
    assert control.current.state is ReservationState.COMMITTED
    assert control.events[-1] == "retain_committed"


@pytest.mark.asyncio
async def test_research_client_cancellation_before_commit_releases_charge(settings, cache) -> None:
    service, core, control = _atomic_research_service("cancel_precommit")
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=True,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        request = asyncio.create_task(
            local.post(
                "/api/v1/research/query",
                json={"symbol": "AAPL", "question": "What risks are described?"},
                headers={"X-App-Key": "correct horse battery staple"},
            )
        )
        await asyncio.wait_for(core.query_started.wait(), timeout=1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

    assert control.current is not None
    assert control.current.state is ReservationState.RELEASED
    assert control.events == ["authorize", "release_authorized"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exhausted", "unavailable", "expected_status"),
    [(True, False, 429), (False, True, 503)],
)
async def test_research_reservation_fails_closed_before_paid_call(
    settings,
    cache,
    exhausted: bool,
    unavailable: bool,
    expected_status: int,
) -> None:
    service, core, control = _atomic_research_service(
        "success",
        exhausted=exhausted,
        unavailable=unavailable,
    )
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == expected_status
    assert core.events == []
    # An indeterminate exception triggers an exact-record release attempt. This fake records
    # every release CAS miss, including a missing record, as "retain_committed".
    assert control.events == (["authorize", "retain_committed"] if unavailable else ["authorize"])
    assert not any("provider_quota" in key for key in cache.counts)


@pytest.mark.asyncio
async def test_research_query_is_authenticated_validated_cited_and_sanitized(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    unauthorized = await client.post(
        "/api/v1/research/query",
        json={"symbol": "AAPL", "question": "What risks are described?"},
    )
    invalid = await client.post(
        "/api/v1/research/query",
        json={"symbol": "AAPL", "question": "bad\u0001question"},
        headers=api_headers,
    )
    arbitrary_url = await client.post(
        "/api/v1/research/query",
        json={
            "symbol": "AAPL",
            "question": "What risks are described?",
            "url": "https://evil.test/instructions",
        },
        headers=api_headers,
    )
    response = await client.post(
        "/api/v1/research/query",
        json={"symbol": "aapl", "question": "What risks are described?"},
        headers=api_headers,
    )
    insufficient = await client.post(
        "/api/v1/research/query",
        json={"symbol": "MSFT", "question": "What risks are described?"},
        headers=api_headers,
    )

    assert unauthorized.status_code == 401
    assert invalid.status_code == 422
    assert arbitrary_url.status_code == 422
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["symbol"] == "AAPL"
    assert payload["data"]["insufficient_evidence"] is False
    assert payload["data"]["citations"][0]["url"].startswith("https://www.sec.gov/Archives/")
    assert "investment advice" in payload["data"]["disclaimer"]
    assert insufficient.status_code == 200
    assert insufficient.json()["data"]["insufficient_evidence"] is True
    assert service.calls == [
        ("query_research", {"symbol": "AAPL", "question": "What risks are described?"}),
        ("query_research", {"symbol": "MSFT", "question": "What risks are described?"}),
    ]


@pytest.mark.asyncio
async def test_session_research_post_requires_same_origin_double_submit_csrf(
    client: httpx.AsyncClient,
    service,
) -> None:
    csrf = await _login_and_dashboard_csrf(client)
    payload = {"symbol": "AAPL", "question": "What risks are described?"}

    missing = await client.post("/api/v1/research/query", json=payload)
    wrong_origin = await client.post(
        "/api/v1/research/query",
        json=payload,
        headers={"Origin": "https://evil.test", "X-CSRF-Token": csrf},
    )
    wrong_token = await client.post(
        "/api/v1/research/query",
        json=payload,
        headers={"Origin": "https://dashboard.test", "X-CSRF-Token": "wrong"},
    )
    accepted = await client.post(
        "/api/v1/research/query",
        json=payload,
        headers={"Origin": "https://dashboard.test", "X-CSRF-Token": csrf},
    )

    assert [missing.status_code, wrong_origin.status_code, wrong_token.status_code] == [
        403,
        403,
        403,
    ]
    assert all(
        response.json()["error"]["code"] == "forbidden"
        for response in (missing, wrong_origin, wrong_token)
    )
    assert accepted.status_code == 200
    assert service.calls == [
        ("query_research", {"symbol": "AAPL", "question": "What risks are described?"})
    ]


@pytest.mark.asyncio
async def test_programmatic_research_post_uses_app_key_without_cookie_csrf(
    client: httpx.AsyncClient,
    service,
) -> None:
    await _login_and_dashboard_csrf(client)
    response = await client.post(
        "/api/v1/research/query",
        json={"symbol": "aapl", "question": "  What risks are described?  "},
        headers={"X-App-Key": "correct horse battery staple"},
    )

    assert response.status_code == 200
    assert service.calls == [
        ("query_research", {"symbol": "AAPL", "question": "What risks are described?"})
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"symbol": "AAPL!", "question": "What risks are described?"},
        {"symbol": "AAPL", "question": "   "},
        {"symbol": "AAPL", "question": "x" * 501},
        {"symbol": "AAPL", "question": 42},
    ],
)
async def test_research_request_model_is_strict(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
    payload: dict[str, object],
) -> None:
    response = await client.post(
        "/api/v1/research/query",
        json=payload,
        headers=api_headers,
    )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "VALIDATION_ERROR",
        "message": "request validation failed",
    }
    assert service.calls == []


@pytest.mark.asyncio
async def test_research_request_has_a_bounded_raw_body(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    response = await client.post(
        "/api/v1/research/query",
        json={
            "symbol": "AAPL",
            "question": "What risks are described?",
            "padding": "x" * 5000,
        },
        headers=api_headers,
    )
    understated = await client.post(
        "/api/v1/research/query",
        content=json.dumps(
            {
                "symbol": "AAPL",
                "question": "What risks are described?",
                "padding": "x" * 5000,
            }
        ),
        headers={
            **api_headers,
            "Content-Length": "1",
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == understated.status_code == 413
    for rejected in (response, understated):
        assert rejected.json()["error"] == {
            "code": "request_too_large",
            "message": "research request body is too large",
        }
    assert service.calls == []


@pytest.mark.asyncio
async def test_research_has_separate_client_rate_and_global_budget_buckets(
    settings,
    cache,
) -> None:
    service, core, control = _atomic_research_service("success")
    isolated = SimpleNamespace(
        **{
            **vars(settings),
            "research_rate_limit": 1,
            "research_daily_global_limit": 2,
        }
    )
    transport = httpx.ASGITransport(
        app=create_app(settings=isolated, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=isolated.allowed_origin) as local:
        first = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )
        limited = await local.post(
            "/api/v1/research/query",
            json={"symbol": "MSFT", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert first.status_code == 200
    assert limited.status_code == 429
    assert cache.counts["rate:research:127.0.0.1"] == 2
    assert not any(key.startswith("budget:research:") for key in cache.counts)
    assert all("marketdata" not in key and "quota" not in key for key in cache.counts)
    assert core.events == ["query", "embed"]
    assert control.events == ["authorize", "commit", "retain_committed"]


@pytest.mark.asyncio
async def test_research_global_budget_is_shared_across_clients(
    settings,
    cache,
) -> None:
    service, core, control = _atomic_research_service("success")
    isolated = SimpleNamespace(
        **{
            **vars(settings),
            "deployment_platform": "vercel",
            "research_rate_limit": 10,
            "research_daily_global_limit": 1,
        }
    )
    transport = httpx.ASGITransport(
        app=create_app(settings=isolated, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=isolated.allowed_origin) as local:
        first = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={
                "X-App-Key": "correct horse battery staple",
                "X-Forwarded-For": "203.0.113.10",
            },
        )
        exhausted = await local.post(
            "/api/v1/research/query",
            json={"symbol": "MSFT", "question": "What risks are described?"},
            headers={
                "X-App-Key": "correct horse battery staple",
                "X-Forwarded-For": "203.0.113.11",
            },
        )

    assert first.status_code == 200
    assert exhausted.status_code == 429
    assert cache.counts["rate:research:203.0.113.10"] == 1
    assert cache.counts["rate:research:203.0.113.11"] == 1
    assert not any(key.startswith("budget:research:") for key in cache.counts)
    assert core.events == ["query", "embed"]
    assert control.events == ["authorize", "commit", "retain_committed", "authorize"]
    assert len(control.reservations) == 2
    assert control.reservations[0].budget_digest == control.reservations[1].budget_digest
    assert control.principal_digests[0] != control.principal_digests[1]


@pytest.mark.asyncio
async def test_research_internal_failure_does_not_leak_secrets(
    settings,
    cache,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider_sentinel = "private-embedding-token-and-vector-index"

    service, _, control = _atomic_research_service(
        "provider_error",
        error_detail=provider_sentinel,
    )

    caplog.set_level(logging.WARNING)
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "RESEARCH_UNAVAILABLE",
        "message": "research is not configured",
    }
    assert provider_sentinel not in response.text
    assert provider_sentinel not in caplog.text
    assert control.current is not None
    assert control.current.state is ReservationState.COMMITTED


@pytest.mark.asyncio
async def test_research_query_has_a_bounded_server_timeout(settings, cache) -> None:
    service, _, control = _atomic_research_service("timeout")

    isolated = SimpleNamespace(
        **{
            **vars(settings),
            "research_enabled": True,
            "research_timeout_seconds": 0.001,
        }
    )
    transport = httpx.ASGITransport(
        app=create_app(settings=isolated, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=isolated.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 504
    assert response.json()["error"] == {
        "code": "RESEARCH_TIMEOUT",
        "message": "the research request timed out",
    }
    assert control.current is not None
    # A deliberately tiny outer deadline may expire on either side of the atomic
    # commit. It must never strand an authorized reservation: pre-commit timeouts
    # release it, while post-commit timeouts retain the charge.
    assert control.current.state in {
        ReservationState.RELEASED,
        ReservationState.COMMITTED,
    }
    expected_event = (
        "release_authorized"
        if control.current.state is ReservationState.RELEASED
        else "retain_committed"
    )
    assert control.events[-1] == expected_event


@pytest.mark.asyncio
async def test_research_is_fail_closed_when_settings_omit_the_enable_flag(
    settings,
    service,
    cache,
) -> None:
    omitted = SimpleNamespace(
        **{name: value for name, value in vars(settings).items() if name != "research_enabled"}
    )
    transport = httpx.ASGITransport(
        app=create_app(settings=omitted, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=omitted.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 503
    assert service.calls == []
    assert not any(key.startswith("budget:research:") for key in cache.counts)


@pytest.mark.asyncio
async def test_research_rejects_unvalidated_adapter_output_without_refunding_committed_spend(
    settings,
    cache,
) -> None:
    malicious_url = "https://evil.example/private-research-source"
    service, _, control = _atomic_research_service(
        "malformed",
        error_detail=malicious_url,
    )

    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "RESEARCH_UNAVAILABLE",
        "message": "research is not configured",
    }
    assert malicious_url not in response.text
    assert control.current is not None
    assert control.current.state is ReservationState.COMMITTED
    assert control.events[-1] == "retain_committed"


@pytest.mark.asyncio
async def test_research_rejects_cross_symbol_typed_adapter_output(settings, cache) -> None:
    service, _, control = _atomic_research_service("cross_symbol")

    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 503
    assert control.current is not None
    assert control.current.state is ReservationState.COMMITTED
    assert control.events[-1] == "retain_committed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/v1/symbols/search?q=AA", "GET"),
        ("/api/v1/research/query", "POST"),
    ],
)
async def test_unconfigured_optional_services_return_sanitized_503(
    settings,
    cache,
    path: str,
    method: str,
) -> None:
    private_configuration = "private-vector-token-and-index-name"
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=object(), cache=cache),
        raise_app_exceptions=False,
    )
    kwargs = {"headers": {"X-App-Key": "correct horse battery staple"}}
    if method == "POST":
        kwargs["json"] = {
            "symbol": "AAPL",
            "question": "What risks are described?",
        }
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.request(method, path, **kwargs)

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "service_unavailable",
        "message": (
            "research service is unavailable"
            if method == "POST"
            else "symbol search service is unavailable"
        ),
    }
    assert private_configuration not in response.text


@pytest.mark.asyncio
async def test_disabled_research_returns_503_without_spending_research_budget(
    settings,
    service,
    cache,
) -> None:
    isolated = SimpleNamespace(**{**vars(settings), "research_enabled": False})
    transport = httpx.ASGITransport(
        app=create_app(settings=isolated, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=isolated.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "service_unavailable",
        "message": "research service is unavailable",
    }
    assert not any(key.startswith("rate:research:") for key in cache.counts)
    assert not any(key.startswith("budget:research:") for key in cache.counts)
    assert service.calls == []


@pytest.mark.asyncio
async def test_disabled_session_research_still_enforces_csrf_before_feature_state(
    settings,
    service,
    cache,
) -> None:
    isolated = SimpleNamespace(**{**vars(settings), "research_enabled": False})
    transport = httpx.ASGITransport(
        app=create_app(settings=isolated, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=isolated.allowed_origin) as local:
        await _login_and_dashboard_csrf(local)
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
        )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    assert service.calls == []


@pytest.mark.asyncio
async def test_missing_production_research_adapters_fail_with_specific_sanitized_503(
    settings,
    service,
    cache,
) -> None:
    isolated = SimpleNamespace(**{**vars(settings), "research_enabled": True})
    dashboard = DashboardService(service, object(), DisabledResearchService())
    transport = httpx.ASGITransport(
        app=create_app(settings=isolated, service=dashboard, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=isolated.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "What risks are described?"},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "RESEARCH_UNAVAILABLE",
        "message": "research is not configured",
    }


@pytest.mark.asyncio
async def test_openapi_research_contract_contains_no_secret_or_arbitrary_url_fields(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
) -> None:
    schema = (await client.get("/openapi.json", headers=api_headers)).json()
    serialized = str(schema).lower()
    request_schema = schema["components"]["schemas"]["ResearchQueryRequest"]

    assert set(request_schema["properties"]) == {"symbol", "question"}
    assert request_schema["additionalProperties"] is False
    assert all(
        secret not in serialized
        for secret in (
            "marketdata_token",
            "openai_api_key",
            "upstash_redis_rest_token",
            "vector_token",
            "correct horse battery staple",
        )
    )


@pytest.mark.asyncio
async def test_api_rejects_missing_or_wrong_app_key_without_service_calls(
    client: httpx.AsyncClient,
    service,
) -> None:
    assert (await client.get("/api/v1/eod/latest/AAPL")).status_code == 401
    assert (
        await client.get("/api/v1/eod/latest/AAPL", headers={"X-App-Key": "wrong"})
    ).status_code == 401
    assert service.calls == []


@pytest.mark.asyncio
async def test_provider_token_cannot_authorize_the_application_boundary(
    settings,
    service,
    cache,
) -> None:
    isolated = replace(
        settings,
        app_access_key_sha256=hashlib.sha256(b"application-only-key").hexdigest(),
    )
    transport = httpx.ASGITransport(
        app=create_app(settings=isolated, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=isolated.allowed_origin) as local:
        response = await local.get(
            "/api/v1/eod/latest/AAPL",
            headers={"X-App-Key": "provider-only-token"},
        )

    assert response.status_code == 401
    assert service.calls == []


@pytest.mark.asyncio
async def test_api_envelope_request_id_and_supported_service_delegation(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    request_id = str(uuid.uuid4())
    service.last_metadata = {
        "source": "cache",
        "as_of": "2026-08-07T00:00:00Z",
        "cached": True,
        "stale": False,
    }

    response = await client.get(
        "/api/v1/eod/latest/aapl",
        headers={**api_headers, "X-Request-ID": request_id},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {"symbol": "AAPL", "close": 201.0},
        "meta": {
            "request_id": request_id,
            "pagination": None,
            "source": "cache",
            "as_of": "2026-08-07T00:00:00Z",
            "cached": True,
            "stale": False,
        },
        "error": None,
    }
    assert response.headers["X-Request-ID"] == request_id
    assert service.calls == [("latest_eod", {"symbol": "AAPL"})]


@pytest.mark.asyncio
async def test_latest_history_and_usage_routes_are_available(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
) -> None:
    today = date.today()
    paths = (
        "/api/v1/eod/latest/AAPL",
        f"/api/v1/eod/history/AAPL?date_from={today - timedelta(days=30)}&date_to={today}",
        "/api/v1/usage",
    )

    responses = [await client.get(path, headers=api_headers) for path in paths]

    assert [response.status_code for response in responses] == [200, 200, 200]


@pytest.mark.asyncio
async def test_latest_route_maps_the_real_market_data_candles_boundary(settings) -> None:
    upstream_requests: list[httpx.Request] = []
    candle_time = int(datetime(2026, 8, 7, tzinfo=UTC).timestamp())

    async def upstream(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(
            203,
            json={
                "s": "ok",
                "o": [201.0],
                "h": [205.0],
                "l": [199.0],
                "c": [204.5],
                "v": [12_345],
                "t": [candle_time],
            },
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    provider = MarketDataAppProvider("provider-only-secret", client=upstream_client)
    shared_cache = MemoryCache()
    service = MarketDataService(provider, shared_cache)
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=shared_cache),
        raise_app_exceptions=False,
    )
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url=settings.allowed_origin
        ) as local:
            response = await local.get(
                "/api/v1/eod/latest/aapl",
                headers={"X-App-Key": "correct horse battery staple"},
            )
            usage_response = await local.get(
                "/api/v1/usage",
                headers={"X-App-Key": "correct horse battery staple"},
            )
    finally:
        await upstream_client.aclose()

    assert response.status_code == 200
    assert response.json()["data"] == {
        "symbol": "AAPL",
        "date": "2026-08-07",
        "open": "201.0",
        "high": "205.0",
        "low": "199.0",
        "close": "204.5",
        "volume": 12_345,
        "adjusted_open": None,
        "adjusted_high": None,
        "adjusted_low": None,
        "adjusted_close": None,
        "adjusted_volume": None,
    }
    assert len(upstream_requests) == 1
    request = upstream_requests[0]
    assert request.url.path == "/v1/stocks/candles/D/AAPL/"
    assert dict(request.url.params) == {
        "countback": "1",
        "adjustsplits": "false",
        "adjustdividends": "false",
    }
    assert "to" not in request.url.params
    assert not {"token", "access_key", "api_key"} & set(request.url.params)
    assert request.headers.get_list("Authorization") == ["Bearer provider-only-secret"]
    assert "x-app-key" not in request.headers
    assert await shared_cache.current_count(service.quota_key) == 1
    assert usage_response.status_code == 200
    assert usage_response.json()["data"]["requests_used"] == 1
    assert usage_response.json()["meta"]["source"] == "local"
    assert len(upstream_requests) == 1


@pytest.mark.asyncio
async def test_exact_reported_history_range_reaches_documented_provider_boundary(settings) -> None:
    upstream_requests: list[httpx.Request] = []
    candle_time = int(datetime(2026, 8, 7, tzinfo=UTC).timestamp())

    async def upstream(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(
            200,
            json={
                "s": "ok",
                "o": [201.0],
                "h": [205.0],
                "l": [199.0],
                "c": [204.5],
                "v": [12_345],
                "t": [candle_time],
            },
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    provider = MarketDataAppProvider("provider-only-secret", client=upstream_client)
    shared_cache = MemoryCache()
    service = MarketDataService(provider, shared_cache)
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=shared_cache),
        raise_app_exceptions=False,
    )
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url=settings.allowed_origin
        ) as local:
            response = await local.get(
                "/api/v1/eod/history/AAPL?date_from=2025-08-08&date_to=2026-08-08&limit=1000",
                headers={"X-App-Key": "correct horse battery staple"},
            )
    finally:
        await upstream_client.aclose()

    assert response.status_code == 200
    assert response.json()["data"][0]["date"] == "2026-08-07"
    assert response.json()["meta"]["pagination"] == {
        "next_cursor": None,
        "total": 1,
    }
    assert len(upstream_requests) == 1
    assert dict(upstream_requests[0].url.params) == {
        "from": "2025-08-08",
        "to": "2026-08-08",
        "adjustsplits": "false",
        "adjustdividends": "false",
    }
    assert not {"limit", "cursor", "offset", "token", "access_key", "api_key"} & set(
        upstream_requests[0].url.params
    )
    assert upstream_requests[0].headers.get_list("Authorization") == ["Bearer provider-only-secret"]
    assert "x-app-key" not in upstream_requests[0].headers
    assert await shared_cache.current_count(service.quota_key) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream_status", [400, 413, 422])
async def test_upstream_request_rejection_is_502_not_cached_and_refunds_quota(
    settings,
    upstream_status: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream_calls = 0

    async def upstream(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return httpx.Response(
            upstream_status,
            json={"s": "error", "errmsg": "private-provider-body"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    provider = MarketDataAppProvider("provider-only-secret", client=upstream_client)
    shared_cache = MemoryCache()
    service = MarketDataService(provider, shared_cache)
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=shared_cache),
        raise_app_exceptions=False,
    )
    caplog.set_level(logging.WARNING, logger="app.main")
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url=settings.allowed_origin
        ) as local:
            responses = [
                await local.get(
                    "/api/v1/eod/history/AAPL?date_from=2025-08-08&date_to=2026-08-08&limit=1000",
                    headers={"X-App-Key": "correct horse battery staple"},
                )
                for _ in range(2)
            ]
    finally:
        await upstream_client.aclose()

    assert [response.status_code for response in responses] == [502, 502]
    assert [response.json()["error"]["code"] for response in responses] == [
        "UPSTREAM_REQUEST_REJECTED",
        "UPSTREAM_REQUEST_REJECTED",
    ]
    assert all("private-provider-body" not in response.text for response in responses)
    assert upstream_calls == 2
    assert await shared_cache.current_count(service.quota_key) == 0
    records = [
        record for record in caplog.records if record.message == "market_data_request_failed"
    ]
    assert len(records) == 2
    assert all(record.upstream_status == upstream_status for record in records)
    assert all(record.semantic_code == "request_rejected" for record in records)
    assert all(record.cache_outcome == "miss" for record in records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream_status", "public_status", "public_code"),
    [
        (401, 502, "UPSTREAM_AUTHENTICATION_FAILED"),
        (402, 502, "UPSTREAM_ACCESS_RESTRICTED"),
        (403, 502, "UPSTREAM_ACCESS_RESTRICTED"),
        (429, 429, "UPSTREAM_QUOTA_EXHAUSTED"),
    ],
)
async def test_real_provider_failures_are_sanitized_and_refund_local_credit(
    settings,
    upstream_status: int,
    public_status: int,
    public_code: str,
) -> None:
    upstream_calls = 0

    async def upstream(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return httpx.Response(upstream_status, json={"s": "error", "errmsg": "private"})

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    provider = MarketDataAppProvider("provider-only-secret", client=upstream_client)
    shared_cache = MemoryCache()
    service = MarketDataService(provider, shared_cache)
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=shared_cache),
        raise_app_exceptions=False,
    )
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url=settings.allowed_origin
        ) as local:
            response = await local.get(
                "/api/v1/eod/latest/AAPL",
                headers={"X-App-Key": "correct horse battery staple"},
            )
    finally:
        await upstream_client.aclose()

    assert response.status_code == public_status
    assert response.json()["error"]["code"] == public_code
    assert "private" not in response.text
    assert "provider-only-secret" not in response.text
    assert upstream_calls == 1
    assert await shared_cache.current_count(service.quota_key) == 0


@pytest.mark.asyncio
async def test_history_cursor_takes_precedence_over_offset(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    today = date.today()
    response = await client.get(
        f"/api/v1/eod/history/aapl?date_from={today - timedelta(days=30)}"
        f"&date_to={today}&limit=25&cursor=9&offset=4",
        headers=api_headers,
    )

    assert response.status_code == 200
    assert service.calls[-1] == (
        "eod_history",
        {
            "symbol": "AAPL",
            "date_from": today - timedelta(days=30),
            "date_to": today,
            "limit": 25,
            "cursor": "9",
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["_AAPL", "AAPL:US", "A B", "AAPL!", "A" * 33])
async def test_invalid_symbols_return_422_without_service_calls(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
    symbol: str,
) -> None:
    response = await client.get(f"/api/v1/eod/latest/{symbol}", headers=api_headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "date_from=2026-01-01",
        "date_to=2026-01-02",
        "date_from=not-a-date&date_to=2026-01-02",
        "date_from=2026-01-02&date_to=2026-01-01",
        "date_from=2025-01-01&date_to=2026-01-02",
    ],
)
async def test_invalid_history_ranges_return_422_without_service_calls(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
    query: str,
) -> None:
    response = await client.get(
        f"/api/v1/eod/history/AAPL?{query}",
        headers=api_headers,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert service.calls == []


@pytest.mark.asyncio
async def test_usage_returns_local_daily_shape_without_provider_headers(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    response = await client.get("/api/v1/usage", headers=api_headers)

    assert response.status_code == 200
    assert response.json()["data"] == {
        "requests_used": 7,
        "requests_limit": 90,
        "requests_remaining": 83,
        "reset_at": "2026-08-09T00:00:00Z",
    }
    assert service.calls == [("usage", {})]
    assert "authorization" not in {key.lower() for key in response.headers}


@pytest.mark.asyncio
async def test_rejected_provider_token_returns_sanitized_diagnostics(
    settings,
    cache,
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaked_token = "provider-token-must-not-leak"

    class RejectingService:
        async def latest_eod(self, symbol: str):
            del symbol
            failure = ProviderAuthenticationError(f"rejected Bearer {leaked_token}")
            failure.upstream_status = 401
            failure.semantic_code = "authentication_failed"
            raise failure

    caplog.set_level(logging.WARNING, logger="app.main")
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=RejectingService(), cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.get(
            "/api/v1/eod/latest/AAPL",
            headers={
                "X-App-Key": "correct horse battery staple",
                "X-Request-ID": "7afef2d7-902d-4298-a6ab-82c86bc57474",
            },
        )

    assert response.status_code == 502
    assert response.json()["error"] == {
        "code": "UPSTREAM_AUTHENTICATION_FAILED",
        "message": "the market data provider rejected its credentials",
    }
    assert leaked_token not in response.text
    records = [
        record for record in caplog.records if record.message == "market_data_request_failed"
    ]
    assert len(records) == 1
    assert records[0].request_id == "7afef2d7-902d-4298-a6ab-82c86bc57474"
    assert records[0].exception_category == "ProviderAuthenticationError"
    assert records[0].upstream_status == 401
    assert records[0].semantic_code == "authentication_failed"
    assert leaked_token not in caplog.text


@pytest.mark.asyncio
async def test_provider_plan_or_ip_restriction_is_sanitized(settings, cache) -> None:
    private_detail = "private-provider-plan-ip-and-account-detail"

    class RestrictedService:
        async def latest_eod(self, symbol: str):
            del symbol
            raise ProviderAccessRestrictedError(private_detail)

    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=RestrictedService(), cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        response = await local.get(
            "/api/v1/eod/latest/AAPL",
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 502
    assert response.json()["error"] == {
        "code": "UPSTREAM_ACCESS_RESTRICTED",
        "message": "the market data provider does not permit this request",
    }
    assert private_detail not in response.text


@pytest.mark.asyncio
async def test_dashboard_exposes_accessible_symbol_lookup_without_removed_sections(
    client: httpx.AsyncClient,
) -> None:
    token = await _csrf(client)
    await client.post(
        "/login",
        data={"password": "correct horse battery staple", "csrf_token": token},
        headers={"Origin": "https://dashboard.test"},
    )

    response = await client.get("/dashboard")

    assert response.status_code == 200
    assert 'name="symbol"' in response.text
    assert 'pattern="[A-Za-z0-9][A-Za-z0-9.\\-]{0,31}"' in response.text
    assert 'role="combobox"' in response.text
    assert 'role="listbox"' in response.text
    assert 'aria-autocomplete="list"' in response.text
    assert "company-metadata" not in response.text
    assert "splits-table" not in response.text
    assert "dividends-table" not in response.text
    assert "Daily API usage" in response.text
    assert "Unadjusted close" in response.text


@pytest.mark.asyncio
async def test_docs_are_protected_by_the_app_key(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert (await client.get(path)).status_code == 401
        assert (await client.get(path, headers=api_headers)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/docs", "/redoc"])
async def test_docs_assets_use_the_per_request_csp_nonce(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    path: str,
) -> None:
    response = await client.get(path, headers=api_headers)
    csp = response.headers["content-security-policy"]
    nonce_match = re.search(r"'nonce-([^']+)'", csp)

    assert response.status_code == 200
    assert nonce_match is not None
    nonce = nonce_match.group(1)
    assert "unsafe-inline" not in csp
    assert f"script-src 'self' https://cdn.jsdelivr.net 'nonce-{nonce}'" in csp
    assert f"style-src 'self' https://cdn.jsdelivr.net 'nonce-{nonce}'" in csp
    assert all(f'nonce="{nonce}"' in tag for tag in re.findall(r"<script[^>]*>", response.text))
    assert all(f'nonce="{nonce}"' in tag for tag in re.findall(r"<style[^>]*>", response.text))


@pytest.mark.asyncio
async def test_same_origin_cors_is_narrow(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
) -> None:
    allowed = await client.options(
        "/api/v1/eod/latest/AAPL",
        headers={
            "Origin": api_headers["Origin"],
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-App-Key",
        },
    )
    denied = await client.options(
        "/api/v1/eod/latest/AAPL",
        headers={"Origin": "https://evil.test", "Access-Control-Request-Method": "GET"},
    )

    assert allowed.headers["access-control-allow-origin"] == api_headers["Origin"]
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.asyncio
async def test_untrusted_host_is_rejected(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health", headers={"Host": "evil.test"})).status_code == 400


@pytest.mark.asyncio
async def test_invalid_request_id_is_replaced_with_uuid(client: httpx.AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "not-a-uuid"})

    assert str(uuid.UUID(response.headers["X-Request-ID"])) == response.headers["X-Request-ID"]


@pytest.mark.asyncio
async def test_local_rate_limits_ignore_forwarded_headers(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    cache,
) -> None:
    response = await client.get(
        "/api/v1/eod/latest/AAPL",
        headers={**api_headers, "X-Forwarded-For": "203.0.113.7"},
    )

    assert response.status_code == 200
    assert "rate:api:127.0.0.1" in cache.counts
    assert "rate:api:203.0.113.7" not in cache.counts


@pytest.mark.asyncio
async def test_vercel_rate_limits_use_official_forwarded_ip(
    settings,
    service,
    cache,
    api_headers: dict[str, str],
) -> None:
    deployed = replace(settings, deployment_platform="vercel")
    transport = httpx.ASGITransport(app=create_app(settings=deployed, service=service, cache=cache))
    async with httpx.AsyncClient(transport=transport, base_url=deployed.allowed_origin) as local:
        response = await local.get(
            "/api/v1/eod/latest/AAPL",
            headers={
                **api_headers,
                "X-Vercel-Forwarded-For": "203.0.113.8",
                "X-Forwarded-For": "198.51.100.1",
            },
        )

    assert response.status_code == 200
    assert "rate:api:203.0.113.8" not in cache.counts
    assert "rate:api:198.51.100.1" in cache.counts


@pytest.mark.asyncio
async def test_lifespan_closes_only_service_when_it_owns_cache(settings) -> None:
    class ClosingResource:
        def __init__(self) -> None:
            self.close_count = 0

        async def aclose(self) -> None:
            self.close_count += 1

    service = ClosingResource()
    cache = ClosingResource()
    application = create_app(settings=settings, service=service, cache=cache)

    async with application.router.lifespan_context(application):
        pass

    assert service.close_count == 1
    assert cache.close_count == 0


@pytest.mark.asyncio
async def test_login_session_authorizes_supported_api_and_logout(client: httpx.AsyncClient) -> None:
    token = await _csrf(client)
    login = await client.post(
        "/login",
        data={"password": "correct horse battery staple", "csrf_token": token},
        headers={"Origin": "https://dashboard.test"},
        follow_redirects=False,
    )

    assert login.status_code == 303
    cookie = login.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert (await client.get("/api/v1/eod/latest/AAPL")).status_code == 200
    dashboard = await client.get("/dashboard")
    csrf = re.search(r'<meta name="csrf-token" content="([^"]+)"', dashboard.text)
    assert csrf is not None

    logout = await client.post(
        "/logout",
        headers={"Origin": "https://dashboard.test", "X-CSRF-Token": csrf.group(1)},
        follow_redirects=False,
    )
    assert logout.status_code == 303
    assert (await client.get("/dashboard", follow_redirects=False)).status_code == 303


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (CacheUnavailableError("redis token leaked"), 503, "CACHE_UNAVAILABLE"),
        (QuotaExceededError("provider token leaked"), 429, "UPSTREAM_QUOTA_EXHAUSTED"),
        (ProviderNotFoundError("provider body leaked"), 404, "NOT_FOUND"),
        (ProviderUnavailableError("provider private host leaked"), 502, "UPSTREAM_UNAVAILABLE"),
        (ProviderTimeoutError("provider timeout detail leaked"), 504, "UPSTREAM_TIMEOUT"),
    ],
)
async def test_domain_failures_are_sanitized(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    status: int,
    code: str,
) -> None:
    async def fail(symbol: str):
        del symbol
        raise failure

    monkeypatch.setattr(service, "latest_eod", fail)
    response = await client.get("/api/v1/eod/latest/AAPL", headers=api_headers)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert str(failure) not in response.text


@pytest.mark.asyncio
async def test_api_rate_limit_fails_closed_when_cache_is_down(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    cache,
) -> None:
    cache.available = False
    response = await client.get("/api/v1/eod/latest/AAPL", headers=api_headers)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rate_limiter_unavailable"


@pytest.mark.asyncio
async def test_login_rate_limit_fails_closed_when_cache_is_down(
    client: httpx.AsyncClient,
    cache,
) -> None:
    cache.available = False
    token = await _csrf(client)
    response = await client.post(
        "/login",
        data={"password": "wrong", "csrf_token": token},
        headers={"Origin": "https://dashboard.test"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rate_limiter_unavailable"
