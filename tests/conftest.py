from __future__ import annotations

import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx
import pytest
import pytest_asyncio

from app.main import create_app
from app.research.control import Reservation, ReservationState
from app.research.deadline import RequestDeadline
from app.research.domain import (
    EvidenceQuote,
    GeneratedClaim,
    ResearchAnswer,
    ResearchCitation,
)


def _is_loopback_host(host: object) -> bool:
    if not isinstance(host, str):
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail tests before any non-loopback DNS or socket connection can leave the host."""

    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def checked_getaddrinfo(host: object, *args: Any, **kwargs: Any):
        if not _is_loopback_host(host):
            raise RuntimeError("external network access is disabled in tests")
        return original_getaddrinfo(host, *args, **kwargs)

    def checked_connect(sock: socket.socket, address: object):
        if not isinstance(address, tuple) or not _is_loopback_host(address[0]):
            raise RuntimeError("external network access is disabled in tests")
        return original_connect(sock, address)

    def checked_connect_ex(sock: socket.socket, address: object):
        if not isinstance(address, tuple) or not _is_loopback_host(address[0]):
            raise RuntimeError("external network access is disabled in tests")
        return original_connect_ex(sock, address)

    def checked_create_connection(address: object, *args: Any, **kwargs: Any):
        if not isinstance(address, tuple) or not _is_loopback_host(address[0]):
            raise RuntimeError("external network access is disabled in tests")
        return original_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", checked_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", checked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", checked_connect_ex)
    monkeypatch.setattr(socket, "create_connection", checked_create_connection)


@dataclass(frozen=True)
class TestSettings:
    session_secret: str = "test-session-secret-that-is-long-enough"
    session_cookie_name: str = "marketdata_session"
    session_max_age_seconds: int = 3600
    cookie_secure: bool = False
    environment: str = "test"
    app_access_key_sha256: str = hashlib.sha256(b"correct horse battery staple").hexdigest()
    allowed_origin: str = "https://dashboard.test"
    login_rate_limit: int = 5
    api_rate_limit: int = 20
    rate_limit_window_seconds: int = 60
    deployment_platform: str = "local"
    research_enabled: bool = True
    research_max_question_chars: int = 500
    research_max_request_bytes: int = 4096
    research_timeout_seconds: float = 8.0
    research_rate_limit: int = 10
    research_daily_global_limit: int = 100


class FakeCache:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.available = True

    async def increment(self, key: str, ttl: int) -> int:
        if not self.available:
            raise ConnectionError("cache unavailable")
        self.counts = {**self.counts, key: self.counts.get(key, 0) + 1}
        return self.counts[key]

    async def reserve_quota(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> int | None:
        del window_seconds
        if not self.available:
            raise ConnectionError("cache unavailable")
        current = self.counts.get(key, 0)
        if current >= limit:
            return None
        reserved = current + 1
        self.counts = {**self.counts, key: reserved}
        return reserved

    async def release_quota(self, key: str) -> int:
        current = self.counts.get(key, 0)
        released = max(0, current - 1)
        self.counts = {**self.counts, key: released}
        return released


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.symbols = (
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "Nasdaq"},
            {"symbol": "APP", "name": "Applovin Corporation", "exchange": "Nasdaq"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "Nasdaq"},
            {"symbol": "MAPS", "name": "WM Technology Inc.", "exchange": "Nasdaq"},
        )

    async def latest_eod(self, symbol: str) -> dict[str, Any]:
        self.calls = [*self.calls, ("latest_eod", {"symbol": symbol})]
        return {"symbol": symbol, "close": 201.0}

    async def eod_history(self, symbol: str, **params: Any) -> dict[str, Any]:
        self.calls = [*self.calls, ("eod_history", {"symbol": symbol, **params})]
        return {"items": [], "next_cursor": None, "total": 0}

    async def usage(self) -> dict[str, int | str]:
        self.calls = [*self.calls, ("usage", {})]
        return {
            "requests_used": 7,
            "requests_limit": 90,
            "requests_remaining": 83,
            "reset_at": "2026-08-09T00:00:00Z",
        }

    async def search_symbols(self, query: str, *, limit: int = 8) -> dict[str, Any]:
        self.calls = [*self.calls, ("search_symbols", {"query": query, "limit": limit})]
        normalized = query.upper()
        matches = sorted(
            (
                item
                for item in self.symbols
                if item["symbol"].startswith(normalized) or normalized in item["name"].upper()
            ),
            key=lambda item: (
                0 if item["symbol"].startswith(normalized) else 1,
                item["symbol"],
            ),
        )[:limit]
        return {
            "items": matches,
            "next_cursor": None,
            "total": len(matches),
            "limit": limit,
            "source": "test-fixture",
            "as_of": "2026-08-09T00:00:00Z",
        }

    async def authorize_research_reservation(
        self,
        principal_digest: str,
        *,
        daily_limit: int,
        window_seconds: int,
        deadline: RequestDeadline | None = None,
    ) -> Reservation | None:
        del daily_limit, window_seconds
        if deadline is not None:
            deadline.raise_if_expired()
        return Reservation(
            reservation_digest="1" * 64,
            budget_digest="2" * 64,
            principal_digest=principal_digest,
            units=1,
            state=ReservationState.AUTHORIZED,
        )

    async def query_research(
        self,
        symbol: str,
        question: str,
        *,
        reservation: Reservation | None = None,
        deadline: RequestDeadline | None = None,
    ) -> ResearchAnswer:
        if reservation is None or reservation.state is not ReservationState.AUTHORIZED:
            raise RuntimeError("research reservation is required")
        if deadline is not None:
            deadline.raise_if_expired()
        self.calls = [*self.calls, ("query_research", {"symbol": symbol, "question": question})]
        if symbol == "AAPL" and "risk" in question.lower():
            quote = EvidenceQuote(
                chunk_id="chunk-" + "a" * 64,
                quote="supply constraints could affect results",
            )
            claim = GeneratedClaim(
                text="Apple identifies supply constraints as a risk.",
                supporting_chunk_ids=(quote.chunk_id,),
                evidence_quotes=(quote,),
            )
            citation = ResearchCitation(
                chunk_id=quote.chunk_id,
                title="Apple 2025 Form 10-K",
                url=(
                    "https://www.sec.gov/Archives/edgar/data/320193/"
                    "000032019325000001/aapl-20250927.htm"
                ),
                filed_date=date(2025, 10, 31),
                filing_type="10-K",
                accession_number="0000320193-25-000001",
                snippet="supply constraints could affect results",
            )
            return ResearchAnswer.answered(symbol, (claim,), (citation,))
        return ResearchAnswer.insufficient(symbol)

    async def release_research_reservation(
        self,
        reservation: Reservation,
        *,
        deadline: RequestDeadline | None = None,
    ) -> bool:
        if deadline is not None:
            deadline.raise_if_expired()
        return reservation.state is ReservationState.AUTHORIZED

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
