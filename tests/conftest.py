from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import pytest_asyncio

from app.main import create_app


@dataclass(frozen=True)
class TestSettings:
    session_secret: str = "test-session-secret-that-is-long-enough"
    session_cookie_name: str = "marketstack_session"
    session_max_age_seconds: int = 3600
    cookie_secure: bool = False
    environment: str = "test"
    app_access_key_sha256: str = hashlib.sha256(b"correct horse battery staple").hexdigest()
    allowed_origin: str = "https://dashboard.test"
    login_rate_limit: int = 5
    api_rate_limit: int = 20
    rate_limit_window_seconds: int = 60
    deployment_platform: str = "local"


class FakeCache:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.available = True

    async def increment(self, key: str, ttl: int) -> int:
        if not self.available:
            raise ConnectionError("cache unavailable")
        self.counts = {**self.counts, key: self.counts.get(key, 0) + 1}
        return self.counts[key]


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def tickers(self, **params: Any) -> dict[str, Any]:
        self.calls = [*self.calls, ("tickers", params)]
        return {"items": [{"symbol": "AAPL"}], "next_cursor": "10", "total": 20}

    async def exchanges(self, **params: Any) -> dict[str, Any]:
        self.calls = [*self.calls, ("exchanges", params)]
        return {"items": [{"mic": "XNAS"}], "pagination": {"count": 1}}

    async def latest_eod(self, symbol: str) -> dict[str, Any]:
        self.calls = [*self.calls, ("latest_eod", {"symbol": symbol})]
        return {"symbol": symbol, "close": 201.0}

    async def eod_history(self, symbol: str, **params: Any) -> dict[str, Any]:
        self.calls = [*self.calls, ("eod_history", {"symbol": symbol, **params})]
        return {"items": [], "pagination": {"count": 0}}

    async def splits(self, **params: Any) -> dict[str, Any]:
        self.calls = [*self.calls, ("splits", params)]
        return {"items": [], "pagination": {"count": 0}}

    async def dividends(self, **params: Any) -> dict[str, Any]:
        self.calls = [*self.calls, ("dividends", params)]
        return {"items": [], "pagination": {"count": 0}}

    async def usage(self) -> dict[str, int]:
        self.calls = [*self.calls, ("usage", {})]
        return {"requests": 7}

    async def health(self) -> dict[str, bool]:
        raise AssertionError("public health must not call the upstream service")


@pytest.fixture
def settings() -> TestSettings:
    return TestSettings()


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def cache() -> FakeCache:
    return FakeCache()


@pytest_asyncio.fixture
async def client(settings: TestSettings, service: FakeService, cache: FakeCache):
    transport = httpx.ASGITransport(
        app=create_app(settings=settings, service=service, cache=cache),
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(
        transport=transport, base_url=settings.allowed_origin
    ) as test_client:
        yield test_client


@pytest.fixture
def api_headers(settings: TestSettings) -> dict[str, str]:
    return {"X-App-Key": "correct horse battery staple", "Origin": settings.allowed_origin}
