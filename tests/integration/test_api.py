from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace

import httpx
import pytest

from app.main import create_app


async def _csrf(client: httpx.AsyncClient) -> str:
    page = await client.get("/")
    match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert match is not None
    return match.group(1)


async def _login(client: httpx.AsyncClient, origin: str) -> str:
    token = await _csrf(client)
    response = await client.post(
        "/login",
        data={"password": "correct horse battery staple", "csrf_token": token},
        headers={"Origin": origin},
        follow_redirects=False,
    )
    assert response.status_code == 303
    dashboard = await client.get("/dashboard")
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', dashboard.text)
    assert match is not None
    return match.group(1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_is_public_and_security_headers_are_present(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["data"] == {"status": "ok"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["content-security-policy"].startswith("default-src 'self'")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_market_routes_are_authenticated_and_enveloped(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    assert (await client.get("/api/v1/eod/latest/AAPL")).status_code == 401
    latest = await client.get("/api/v1/eod/latest/aapl", headers=api_headers)
    history = await client.get(
        "/api/v1/eod/history/AAPL?date_from=2026-01-01&date_to=2026-01-31&limit=20",
        headers=api_headers,
    )
    usage = await client.get("/api/v1/usage", headers=api_headers)

    assert latest.status_code == history.status_code == usage.status_code == 200
    assert latest.json()["data"] == {"symbol": "AAPL", "close": 201.0}
    assert history.json()["meta"]["pagination"]["total"] == 0
    assert usage.json()["data"]["requests_remaining"] == 83
    assert [call[0] for call in service.calls] == ["latest_eod", "eod_history", "usage"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_marketview_analysis_uses_existing_endpoint_and_openai_shaped_answer(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    response = await client.post(
        "/api/v1/research/query",
        json={"symbol": "aapl", "question": "  What is the latest close?  "},
        headers=api_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["symbol"] == "AAPL"
    assert body["data"]["status"] == "answered"
    assert body["data"]["provider"] == "marketdata.app"
    assert body["data"]["evidence_count"] == 0
    assert body["data"]["disclaimer"].startswith("AI-assisted analysis")
    assert service.calls == [
        ("query_research", {"symbol": "AAPL", "question": "What is the latest close?"})
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_analysis_requires_double_submit_csrf(settings, service, cache) -> None:
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=settings.allowed_origin) as local:
        csrf = await _login(local, settings.allowed_origin)
        missing = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "Analyze the latest close."},
        )
        accepted = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "Analyze the latest close."},
            headers={"Origin": settings.allowed_origin, "X-CSRF-Token": csrf},
        )

    assert missing.status_code == 403
    assert accepted.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"symbol": "AAPL!", "question": "Analyze."},
        {"symbol": "AAPL", "question": "   "},
        {"symbol": "AAPL", "question": "x" * 501},
        {"symbol": "AAPL", "question": 42},
    ],
)
async def test_analysis_request_is_strict(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
    payload: dict[str, object],
) -> None:
    response = await client.post("/api/v1/research/query", json=payload, headers=api_headers)

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analysis_body_rate_and_daily_budgets_are_bounded(settings, service, cache) -> None:
    limited = replace(settings, research_rate_limit=1, research_daily_global_limit=1)
    transport = httpx.ASGITransport(
        app=create_app(settings=limited, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    headers = {"X-App-Key": "correct horse battery staple", "Origin": limited.allowed_origin}
    async with httpx.AsyncClient(transport=transport, base_url=limited.allowed_origin) as local:
        oversized = await local.post(
            "/api/v1/research/query",
            content=json.dumps({"symbol": "AAPL", "question": "Analyze.", "padding": "x" * 5000}),
            headers={**headers, "Content-Type": "application/json"},
        )
        first = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "Analyze the latest close."},
            headers=headers,
        )
        limited_response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "MSFT", "question": "Analyze the latest close."},
            headers=headers,
        )

    assert oversized.status_code == 413
    assert first.status_code == 200
    assert limited_response.status_code == 429
    assert any(key.startswith("budget:market-analysis:") for key in cache.counts)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disabled_analysis_fails_closed_without_service_call(
    settings, service, cache
) -> None:
    disabled = replace(settings, research_enabled=False)
    transport = httpx.ASGITransport(
        app=create_app(settings=disabled, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=disabled.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "Analyze the latest close."},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 503
    assert service.calls == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analysis_timeout_is_sanitized(settings, cache) -> None:
    class SlowService:
        async def query_research(self, symbol: str, question: str, **kwargs):
            del kwargs
            del symbol, question
            await asyncio.sleep(1)

    bounded = replace(settings, research_timeout_seconds=0.001)
    transport = httpx.ASGITransport(
        app=create_app(settings=bounded, service=SlowService(), cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(transport=transport, base_url=bounded.allowed_origin) as local:
        response = await local.post(
            "/api/v1/research/query",
            json={"symbol": "AAPL", "question": "Analyze the latest close."},
            headers={"X-App-Key": "correct horse battery staple"},
        )

    assert response.status_code == 504
    assert response.json()["error"] == {
        "code": "RESEARCH_TIMEOUT",
        "message": "the research request timed out",
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openapi_contract_contains_no_rag_or_secret_fields(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
) -> None:
    schema = (await client.get("/openapi.json", headers=api_headers)).json()
    serialized = json.dumps(schema).lower()
    request_schema = schema["components"]["schemas"]["ResearchQueryRequest"]

    assert set(request_schema["properties"]) == {"symbol", "question"}
    assert request_schema["additionalProperties"] is False
    for forbidden in ("openai_api_key", "vector", "embedding", "sec filing"):
        assert forbidden not in serialized
