from __future__ import annotations

import re
import uuid
from dataclasses import replace
from datetime import date, timedelta

import httpx
import pytest

from api.index import app as serverless_app
from app.errors import (
    CacheUnavailableError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from app.main import create_app


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


def test_serverless_entrypoint_builds_a_fastapi_application() -> None:
    assert serverless_app.title == "Marketstack Dashboard API"


@pytest.mark.asyncio
async def test_api_rejects_missing_or_wrong_app_key(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/v1/tickers")).status_code == 401
    assert (await client.get("/api/v1/tickers", headers={"X-App-Key": "wrong"})).status_code == 401


@pytest.mark.asyncio
async def test_api_envelope_request_id_and_service_delegation(
    client: httpx.AsyncClient, api_headers: dict[str, str], service
) -> None:
    request_id = str(uuid.uuid4())
    service.last_metadata = {
        "source": "cache",
        "as_of": "2026-08-07T00:00:00Z",
        "cached": True,
        "stale": False,
    }
    response = await client.get(
        "/api/v1/tickers?search=apple&limit=10&offset=2",
        headers={**api_headers, "X-Request-ID": request_id},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": [{"symbol": "AAPL"}],
        "meta": {
            "request_id": request_id,
            "pagination": {"next_cursor": "10", "total": 20},
            "source": "cache",
            "as_of": "2026-08-07T00:00:00Z",
            "cached": True,
            "stale": False,
        },
        "error": None,
    }
    assert response.headers["X-Request-ID"] == request_id
    assert service.calls[-1] == (
        "tickers",
        {"search": "apple", "limit": 10, "offset": 2},
    )


@pytest.mark.asyncio
async def test_all_scripted_endpoints_are_available(
    client: httpx.AsyncClient, api_headers: dict[str, str]
) -> None:
    today = date.today()
    paths = [
        "/api/v1/exchanges",
        "/api/v1/eod/latest/AAPL",
        f"/api/v1/eod/history/AAPL?date_from={today - timedelta(days=30)}&date_to={today}",
        "/api/v1/splits/AAPL",
        "/api/v1/dividends/AAPL",
        "/api/v1/usage",
    ]

    responses = [await client.get(path, headers=api_headers) for path in paths]
    assert [response.status_code for response in responses] == [200] * len(paths)


@pytest.mark.asyncio
async def test_cursor_takes_precedence_over_offset_on_all_list_routes(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
) -> None:
    today = date.today()
    paths = [
        "/api/v1/exchanges?limit=25&cursor=next-page&offset=9",
        (
            f"/api/v1/eod/history/AAPL?date_from={today - timedelta(days=30)}"
            f"&date_to={today}&limit=25&cursor=next-page&offset=9"
        ),
        "/api/v1/splits/AAPL?limit=25&cursor=next-page&offset=9",
        "/api/v1/dividends/AAPL?limit=25&cursor=next-page&offset=9",
    ]

    responses = [await client.get(path, headers=api_headers) for path in paths]

    assert [response.status_code for response in responses] == [200] * len(paths)
    assert service.calls[-4:] == [
        ("exchanges", {"limit": 25, "cursor": "next-page"}),
        (
            "eod_history",
            {
                "symbol": "AAPL",
                "date_from": today - timedelta(days=30),
                "date_to": today,
                "limit": 25,
                "cursor": "next-page",
            },
        ),
        ("splits", {"symbol": "AAPL", "limit": 25, "cursor": "next-page"}),
        ("dividends", {"symbol": "AAPL", "limit": 25, "cursor": "next-page"}),
    ]


@pytest.mark.asyncio
async def test_history_rejects_more_than_one_year(
    client: httpx.AsyncClient, api_headers: dict[str, str]
) -> None:
    today = date.today()
    response = await client.get(
        f"/api/v1/eod/history/AAPL?date_from={today - timedelta(days=366)}&date_to={today}",
        headers=api_headers,
    )

    assert response.status_code == 422
    assert response.json()["success"] is False
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_docs_are_protected_by_the_app_key(
    client: httpx.AsyncClient, api_headers: dict[str, str]
) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert (await client.get(path)).status_code == 401
        assert (await client.get(path, headers=api_headers)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/docs", "/redoc"])
async def test_docs_scripts_and_styles_use_the_per_request_csp_nonce(
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
    client: httpx.AsyncClient, api_headers: dict[str, str]
) -> None:
    allowed = await client.options(
        "/api/v1/tickers",
        headers={
            "Origin": api_headers["Origin"],
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-App-Key",
        },
    )
    denied = await client.options(
        "/api/v1/tickers",
        headers={"Origin": "https://evil.test", "Access-Control-Request-Method": "GET"},
    )

    assert allowed.headers["access-control-allow-origin"] == api_headers["Origin"]
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.asyncio
async def test_untrusted_host_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.get("/health", headers={"Host": "evil.test"})

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_invalid_request_id_is_replaced_with_a_uuid(client: httpx.AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "not-a-uuid"})

    assert str(uuid.UUID(response.headers["X-Request-ID"])) == response.headers["X-Request-ID"]


@pytest.mark.asyncio
async def test_local_rate_limits_ignore_forwarded_ip_headers(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    cache,
) -> None:
    response = await client.get(
        "/api/v1/tickers",
        headers={**api_headers, "X-Vercel-Forwarded-For": "203.0.113.7"},
    )

    assert response.status_code == 200
    assert "rate:api:127.0.0.1" in cache.counts
    assert "rate:api:203.0.113.7" not in cache.counts


@pytest.mark.asyncio
async def test_vercel_rate_limits_use_first_valid_platform_forwarded_ip(
    settings,
    service,
    cache,
    api_headers: dict[str, str],
) -> None:
    deployed = replace(settings, deployment_platform="vercel")
    transport = httpx.ASGITransport(app=create_app(settings=deployed, service=service, cache=cache))
    async with httpx.AsyncClient(
        transport=transport, base_url=deployed.allowed_origin
    ) as deployed_client:
        response = await deployed_client.get(
            "/api/v1/tickers",
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
async def test_malformed_deployed_forwarding_headers_fall_back_to_socket_ip(
    settings,
    service,
    cache,
    api_headers: dict[str, str],
) -> None:
    deployed = replace(settings, deployment_platform="vercel")
    transport = httpx.ASGITransport(app=create_app(settings=deployed, service=service, cache=cache))
    async with httpx.AsyncClient(
        transport=transport, base_url=deployed.allowed_origin
    ) as deployed_client:
        response = await deployed_client.get(
            "/api/v1/tickers",
            headers={
                **api_headers,
                "X-Vercel-Forwarded-For": "not-an-ip",
                "X-Forwarded-For": "also-not-an-ip",
            },
        )

    assert response.status_code == 200
    assert "rate:api:127.0.0.1" in cache.counts


@pytest.mark.asyncio
async def test_lifespan_closes_only_the_service_when_it_owns_the_cache(settings) -> None:
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
async def test_lifespan_closes_cache_when_service_has_no_close(settings) -> None:
    class ClosingCache:
        def __init__(self) -> None:
            self.close_count = 0

        async def aclose(self) -> None:
            self.close_count += 1

    cache = ClosingCache()
    application = create_app(settings=settings, service=object(), cache=cache)

    async with application.router.lifespan_context(application):
        pass

    assert cache.close_count == 1


@pytest.mark.asyncio
async def test_login_requires_origin_and_csrf_then_sets_hardened_cookie(
    client: httpx.AsyncClient,
) -> None:
    token = await _csrf(client)
    denied = await client.post(
        "/auth/login",
        data={
            "username": "analyst",
            "password": "correct horse battery staple",
            "csrf_token": token,
        },
        headers={"Origin": "https://evil.test"},
    )
    assert denied.status_code == 403

    token = await _csrf(client)
    response = await client.post(
        "/auth/login",
        data={
            "username": "analyst",
            "password": "correct horse battery staple",
            "csrf_token": token,
        },
        headers={"Origin": "https://dashboard.test"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert (await client.get("/dashboard")).status_code == 200


@pytest.mark.asyncio
async def test_logout_accepts_the_dashboard_csrf_header(client: httpx.AsyncClient) -> None:
    token = await _csrf(client)
    await client.post(
        "/login",
        data={"password": "correct horse battery staple", "csrf_token": token},
        headers={"Origin": "https://dashboard.test"},
    )
    dashboard = await client.get("/dashboard")
    csrf = re.search(r'<meta name="csrf-token" content="([^"]+)"', dashboard.text)
    assert csrf is not None

    logout = await client.post(
        "/logout",
        headers={
            "Origin": "https://dashboard.test",
            "X-CSRF-Token": csrf.group(1),
        },
        follow_redirects=False,
    )

    assert logout.status_code == 303
    assert (await client.get("/dashboard", follow_redirects=False)).status_code == 303


@pytest.mark.asyncio
async def test_secure_production_responses_emit_hsts(settings, service, cache) -> None:
    production = replace(settings, environment="production", cookie_secure=True)
    transport = httpx.ASGITransport(
        app=create_app(settings=production, service=service, cache=cache)
    )
    async with httpx.AsyncClient(transport=transport, base_url=production.allowed_origin) as secure:
        response = await secure.get("/health")

    assert response.headers["strict-transport-security"].startswith("max-age=")


def test_non_secure_production_cookie_configuration_is_rejected(settings, service, cache) -> None:
    production = replace(settings, environment="production", cookie_secure=False)

    with pytest.raises(ValueError, match="non-secure cookies"):
        create_app(settings=production, service=service, cache=cache)


@pytest.mark.asyncio
async def test_login_rate_limit_fails_closed_when_cache_is_down(
    client: httpx.AsyncClient, cache
) -> None:
    cache.available = False
    token = await _csrf(client)
    response = await client.post(
        "/auth/login",
        data={"username": "analyst", "password": "wrong", "csrf_token": token},
        headers={"Origin": "https://dashboard.test"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rate_limiter_unavailable"


@pytest.mark.asyncio
async def test_signed_browser_session_authorizes_dashboard_api(client: httpx.AsyncClient) -> None:
    token = await _csrf(client)
    login = await client.post(
        "/login",
        data={"password": "correct horse battery staple", "csrf_token": token},
        headers={"Origin": "https://dashboard.test"},
        follow_redirects=False,
    )

    assert login.status_code == 303
    assert (await client.get("/api/v1/tickers")).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (CacheUnavailableError("redis token leaked"), 503, "CACHE_UNAVAILABLE"),
        (QuotaExceededError("upstream key leaked"), 429, "UPSTREAM_QUOTA_EXHAUSTED"),
        (ProviderUnavailableError("provider private host leaked"), 502, "UPSTREAM_UNAVAILABLE"),
        (ProviderTimeoutError("provider timeout detail leaked"), 504, "UPSTREAM_UNAVAILABLE"),
    ],
)
async def test_domain_failures_are_sanitized(
    client: httpx.AsyncClient,
    api_headers: dict[str, str],
    service,
    monkeypatch,
    failure: Exception,
    status: int,
    code: str,
) -> None:
    async def fail(**params):
        del params
        raise failure

    monkeypatch.setattr(service, "tickers", fail)
    response = await client.get("/api/v1/tickers", headers=api_headers)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert str(failure) not in response.text


@pytest.mark.asyncio
async def test_api_rate_limit_fails_closed_when_cache_is_down(
    client: httpx.AsyncClient, api_headers: dict[str, str], cache
) -> None:
    cache.available = False
    response = await client.get("/api/v1/tickers", headers=api_headers)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rate_limiter_unavailable"
