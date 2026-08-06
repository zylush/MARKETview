"""Deterministic loopback application and browser fixtures for opt-in E2E tests."""

from __future__ import annotations

import hashlib
import os
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx
import pytest
import uvicorn

from app.main import create_app

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    PlaywrightError = RuntimeError  # type: ignore[misc,assignment]
    sync_playwright = None

E2E_PASSWORD = "test-password"


@dataclass(frozen=True)
class E2ESettings:
    session_secret: str
    session_cookie_name: str
    session_max_age_seconds: int
    cookie_secure: bool
    environment: str
    app_access_key_sha256: str
    allowed_origin: str
    allowed_hosts: tuple[str, ...]
    login_rate_limit: int
    api_rate_limit: int
    rate_limit_window_seconds: int


@dataclass(frozen=True)
class RunningServer:
    base_url: str
    app_key: str


class FakeCache:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    async def increment(self, key: str, ttl: int) -> int:
        del ttl
        next_count = self._counts.get(key, 0) + 1
        self._counts = {**self._counts, key: next_count}
        return next_count


class FakeMarketService:
    @property
    def last_metadata(self) -> dict[str, Any]:
        return {
            "source": "e2e-fixture",
            "as_of": "2026-08-07T00:00:00Z",
            "cached": False,
            "stale": False,
        }

    @staticmethod
    def _tickers() -> tuple[dict[str, str], ...]:
        return (
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "mic": "XNAS",
                "country": "United States",
                "currency": "USD",
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft Corporation",
                "exchange": "NASDAQ",
                "mic": "XNAS",
                "country": "United States",
                "currency": "USD",
            },
        )

    async def tickers(self, **params: Any) -> dict[str, Any]:
        search = str(params.get("search", "")).casefold()
        items = [
            dict(ticker)
            for ticker in self._tickers()
            if not search
            or search in ticker["symbol"].casefold()
            or search in ticker["name"].casefold()
        ]
        return {"items": items, "next_cursor": None, "total": len(items)}

    async def exchanges(self, **params: Any) -> dict[str, Any]:
        del params
        items = [
            {
                "name": "Nasdaq Stock Market",
                "acronym": "NASDAQ",
                "mic": "XNAS",
                "country": "United States",
                "currency": "USD",
            }
        ]
        return {"items": items, "next_cursor": None, "total": len(items)}

    async def latest_eod(self, symbol: str) -> dict[str, Any]:
        close = 423.46 if symbol == "MSFT" else 229.35
        return {
            "symbol": symbol,
            "date": "2026-08-06T00:00:00Z",
            "open": close - 2.1,
            "high": close + 1.4,
            "low": close - 3.2,
            "close": close,
            "volume": 24_318_742,
        }

    async def history(
        self,
        symbol: str,
        date_from: date,
        date_to: date,
        **params: Any,
    ) -> dict[str, Any]:
        del date_from, params
        baseline = 420.0 if symbol == "MSFT" else 225.0
        items = [
            {
                "symbol": symbol,
                "date": (date_to - timedelta(days=offset)).isoformat(),
                "open": baseline + index,
                "high": baseline + index + 2.0,
                "low": baseline + index - 1.0,
                "close": baseline + index + 1.0,
                "volume": 20_000_000 + index * 100_000,
            }
            for index, offset in enumerate((4, 3, 2, 1, 0))
        ]
        return {"items": items, "next_cursor": None, "total": len(items)}

    async def splits(self, **params: Any) -> dict[str, Any]:
        item = {
            "symbol": params["symbol"],
            "date": "2022-06-06",
            "split_factor": "2:1",
        }
        return {"items": [item], "next_cursor": None, "total": 1}

    async def dividends(self, **params: Any) -> dict[str, Any]:
        item = {
            "symbol": params["symbol"],
            "date": "2026-06-12",
            "dividend": 0.83,
            "currency": "USD",
        }
        return {"items": [item], "next_cursor": None, "total": 1}

    async def usage(self) -> dict[str, Any]:
        return {
            "used": 17,
            "limit": 100,
            "remaining": 83,
            "reset_at": "2026-09-01T00:00:00Z",
        }


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.getenv("RUN_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_E2E=1 to execute loopback browser tests")
    for item in items:
        if item.get_closest_marker("e2e"):
            item.add_marker(skip)


def _wait_until_ready(base_url: str, thread: threading.Thread) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("the E2E uvicorn server stopped during startup")
        try:
            response = httpx.get(f"{base_url}/health", timeout=0.25)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.05)
    raise RuntimeError("the E2E uvicorn server did not become ready")


@pytest.fixture(scope="session")
def e2e_server() -> Iterator[RunningServer]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = int(listener.getsockname()[1])
    base_url = f"http://127.0.0.1:{port}"
    settings = E2ESettings(
        session_secret="e2e-session-secret-at-least-sixteen-chars",
        session_cookie_name="marketstack_e2e_session",
        session_max_age_seconds=600,
        cookie_secure=False,
        environment="test",
        app_access_key_sha256=hashlib.sha256(E2E_PASSWORD.encode()).hexdigest(),
        allowed_origin=base_url,
        allowed_hosts=("127.0.0.1", "localhost"),
        login_rate_limit=20,
        api_rate_limit=100,
        rate_limit_window_seconds=60,
    )
    application = create_app(
        settings=settings,
        service=FakeMarketService(),
        cache=FakeCache(),
    )
    config = uvicorn.Config(application, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        name="marketstack-e2e-server",
        daemon=True,
    )
    thread.start()
    try:
        _wait_until_ready(base_url, thread)
        yield RunningServer(base_url=base_url, app_key=E2E_PASSWORD)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        listener.close()


@pytest.fixture(scope="session")
def base_url(e2e_server: RunningServer) -> str:
    return e2e_server.base_url


@pytest.fixture(scope="session")
def app_key(e2e_server: RunningServer) -> str:
    return e2e_server.app_key


@pytest.fixture
def page():
    if sync_playwright is None:
        pytest.skip("the Playwright Python package is not installed")
    with sync_playwright() as driver:
        try:
            browser = driver.chromium.launch()
        except PlaywrightError as error:
            pytest.skip(f"Playwright Chromium is not installed: {error}")
        context = browser.new_context(reduced_motion="reduce")
        context.add_init_script(
            """
            window.Chart = class ChartStub {
              constructor() { this.destroy = () => {}; }
            };
            """
        )
        active_page = context.new_page()
        active_page.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
        yield active_page
        context.close()
        browser.close()
