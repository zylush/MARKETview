from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr

import app.main as main_module
from app.cache.memory import MemoryCache
from app.errors import (
    CacheUnavailableError,
    ProviderAccessRestrictedError,
    ProviderAuthenticationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from app.main import create_app
from app.providers.marketdata import MarketDataAppProvider
from app.services.market_data import MarketDataService


async def _csrf(client: httpx.AsyncClient) -> str:
    page = await client.get("/")
    assert page.status_code == 200
    match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
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
    }
    assert "to" not in request.url.params
    assert request.headers.get_list("Authorization") == ["Bearer provider-only-secret"]
    assert "x-app-key" not in request.headers
    assert await shared_cache.current_count(service.quota_key) == 1
    assert usage_response.status_code == 200
    assert usage_response.json()["data"]["requests_used"] == 1
    assert usage_response.json()["meta"]["source"] == "local"
    assert len(upstream_requests) == 1


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
    assert service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "date_from=2026-01-01",
        "date_to=2026-01-02",
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
async def test_dashboard_exposes_only_direct_symbol_lookup_sections(
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
    assert 'role="combobox"' not in response.text
    assert 'role="listbox"' not in response.text
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
        headers={**api_headers, "X-Vercel-Forwarded-For": "203.0.113.7"},
    )

    assert response.status_code == 200
    assert "rate:api:127.0.0.1" in cache.counts
    assert "rate:api:203.0.113.7" not in cache.counts


@pytest.mark.asyncio
async def test_vercel_rate_limits_use_first_valid_forwarded_ip(
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
                "X-Vercel-Forwarded-For": "invalid, 203.0.113.8, 203.0.113.9",
                "X-Forwarded-For": "198.51.100.1",
            },
        )

    assert response.status_code == 200
    assert "rate:api:203.0.113.8" in cache.counts
    assert "rate:api:198.51.100.1" not in cache.counts


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
        (ProviderUnavailableError("provider private host leaked"), 502, "UPSTREAM_UNAVAILABLE"),
        (ProviderTimeoutError("provider timeout detail leaked"), 504, "UPSTREAM_UNAVAILABLE"),
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
