"""Deterministic loopback application and browser fixtures for opt-in E2E tests."""

from __future__ import annotations

import hashlib
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pytest
import uvicorn

from app.main import create_app
from app.market_analysis.domain import MarketAnalysisAnswer, MarketAnswerStatus, MarketPeriod

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
    research_enabled: bool
    research_max_question_chars: int
    research_max_request_bytes: int
    research_timeout_seconds: float
    research_rate_limit: int
    research_daily_global_limit: int


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

    async def reserve_quota(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> int | None:
        del window_seconds
        next_count = self._counts.get(key, 0) + 1
        if next_count > limit:
            return None
        self._counts = {**self._counts, key: next_count}
        return next_count

    async def release_quota(self, key: str) -> int:
        next_count = max(0, self._counts.get(key, 0) - 1)
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

    async def usage(self) -> dict[str, Any]:
        return {
            "requests_used": 17,
            "requests_limit": 90,
            "requests_remaining": 73,
            "reset_at": "2026-08-09T00:00:00Z",
        }

    async def search_symbols(self, query: str, *, limit: int = 8) -> dict[str, Any]:
        records = (
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "Nasdaq"},
            {"symbol": "APP", "name": "Applovin Corporation", "exchange": "Nasdaq"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "Nasdaq"},
        )
        normalized = query.upper()
        items = [
            record
            for record in records
            if record["symbol"].startswith(normalized) or normalized in record["name"].upper()
        ][:limit]
        return {
            "items": items,
            "next_cursor": None,
            "total": len(items),
            "limit": limit,
            "source": "e2e-fixture",
            "as_of": "2026-08-09T00:00:00Z",
        }

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        before_generation=None,
    ) -> MarketAnalysisAnswer:
        if before_generation is not None:
            await before_generation()
        if "buy" in question.lower() or "sell" in question.lower():
            return MarketAnalysisAnswer(
                symbol=symbol,
                status=MarketAnswerStatus.INSUFFICIENT_EVIDENCE,
                answer="MarketView cannot provide personalized investment advice.",
            )
        return MarketAnalysisAnswer(
            symbol=symbol,
            status=MarketAnswerStatus.ANSWERED,
            answer=f"AI-assisted analysis of collected {symbol} market data.",
            provider="marketdata.app",
            as_of=datetime(2026, 8, 9, tzinfo=UTC),
            periods=(MarketPeriod(start=date(2026, 7, 10), end=date(2026, 8, 9)),),
        )


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
        session_cookie_name="marketdata_e2e_session",
        session_max_age_seconds=600,
        cookie_secure=False,
        environment="test",
        app_access_key_sha256=hashlib.sha256(E2E_PASSWORD.encode()).hexdigest(),
        allowed_origin=base_url,
        allowed_hosts=("127.0.0.1", "localhost"),
        login_rate_limit=20,
        api_rate_limit=100,
        rate_limit_window_seconds=60,
        research_enabled=True,
        research_max_question_chars=500,
        research_max_request_bytes=4096,
        research_timeout_seconds=2.0,
        research_rate_limit=100,
        research_daily_global_limit=100,
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
        name="marketdata-e2e-server",
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
