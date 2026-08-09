from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from app.errors import ProviderUnavailableError
from app.providers.sec_symbols import SEC_SYMBOL_DIRECTORY_URL, SecSymbolDirectoryProvider


def run(coro):
    return asyncio.run(coro)


def test_provider_uses_fixed_sec_url_descriptive_user_agent_and_bounded_timeout() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SecSymbolDirectoryProvider(
        user_agent="MarketView/1.0 operations@example.com",
        client=client,
        timeout_seconds=4.0,
        clock=lambda: datetime(2026, 8, 9, tzinfo=UTC),
    )

    result = run(provider.load())
    run(client.aclose())

    assert len(requests) == 1
    assert str(requests[0].url) == SEC_SYMBOL_DIRECTORY_URL
    assert requests[0].headers["user-agent"] == "MarketView/1.0 operations@example.com"
    assert requests[0].headers["accept"] == "application/json"
    assert max(requests[0].extensions["timeout"].values()) == 4.0
    assert result.source == "sec-company-tickers-exchange"
    assert result.as_of == datetime(2026, 8, 9, tzinfo=UTC)
    assert result.records[0].symbol == "AAPL"


def test_provider_owned_client_ignores_environment_proxies() -> None:
    provider = SecSymbolDirectoryProvider(user_agent="MarketView/1.0 operations@example.com")

    try:
        assert provider._client._trust_env is False
    finally:
        run(provider.aclose())


def test_provider_maps_columns_skips_malformed_rows_and_deduplicates_deterministically() -> None:
    payload = {
        "fields": ["exchange", "ticker", "cik", "name"],
        "data": [
            ["NYSE", "DUP", 2, "Zulu Duplicate"],
            ["Nasdaq", "AAPL", 320193, "Apple Inc."],
            ["NYSE", "DUP", 1, "Alpha Duplicate"],
            ["NYSE", "BAD<script>", 3, "Unsafe"],
            ["NYSE", "OK", 4],
            "not-a-row",
            ["", "EMPTY", 5, "Missing Exchange"],
        ],
    }

    provider = SecSymbolDirectoryProvider(
        user_agent="MarketView/1.0 operations@example.com",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
        ),
    )

    result = run(provider.load())
    run(provider.aclose())

    assert [(item.symbol, item.name, item.exchange) for item in result.records] == [
        ("AAPL", "Apple Inc.", "Nasdaq"),
        ("DUP", "Alpha Duplicate", "NYSE"),
    ]


@pytest.mark.parametrize(
    ("response_factory", "expected"),
    [
        (lambda: httpx.Response(503, text="secret payload marker"), "unavailable"),
        (lambda: httpx.Response(200, content=b"not-json secret payload marker"), "invalid"),
        (lambda: httpx.Response(200, json={"fields": [], "data": []}), "invalid"),
    ],
)
def test_provider_failures_are_sanitized_and_never_retried(response_factory, expected) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response_factory()

    provider = SecSymbolDirectoryProvider(
        user_agent="MarketView/1.0 operations@example.com",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderUnavailableError, match=expected) as captured:
        run(provider.load())
    run(provider.aclose())

    assert calls == 1
    assert "secret payload marker" not in str(captured.value)


def test_provider_timeout_is_sanitized_and_never_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("secret timeout payload", request=request)

    provider = SecSymbolDirectoryProvider(
        user_agent="MarketView/1.0 operations@example.com",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderUnavailableError, match="timed out") as captured:
        run(provider.load())
    run(provider.aclose())

    assert calls == 1
    assert "secret timeout payload" not in str(captured.value)


def test_provider_stops_streaming_when_directory_exceeds_byte_limit() -> None:
    class OversizedStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.chunks_read = 0

        async def __aiter__(self):
            for _ in range(4):
                self.chunks_read += 1
                yield b"x" * (6 * 1024 * 1024)

        async def aclose(self) -> None:
            return None

    stream = OversizedStream()
    provider = SecSymbolDirectoryProvider(
        user_agent="MarketView/1.0 operations@example.com",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        ),
    )

    with pytest.raises(ProviderUnavailableError, match="invalid"):
        run(provider.load())
    run(provider.aclose())

    assert stream.chunks_read == 3


def test_provider_error_traceback_retains_no_request_or_payload_details() -> None:
    contact = "sentinel-contact-7291@example.com"
    payload = "sentinel-malformed-payload-4168"
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload.encode()))
    )
    provider = SecSymbolDirectoryProvider(
        user_agent=f"MarketView/1.0 {contact}",
        client=client,
    )

    with pytest.raises(ProviderUnavailableError) as captured:
        run(provider.load())
    run(client.aclose())

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    retained: list[str] = []
    traceback = captured.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("sec_symbols.py"):
            retained.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    joined = "\n".join(retained)
    assert contact not in joined
    assert payload not in joined
    assert SEC_SYMBOL_DIRECTORY_URL not in joined


@pytest.mark.parametrize(
    "user_agent",
    ["", "python-httpx", "MarketView", "MarketView/1.0\r\nInjected: yes"],
)
def test_provider_rejects_non_descriptive_or_unsafe_user_agent(user_agent: str) -> None:
    with pytest.raises(ValueError, match="descriptive"):
        SecSymbolDirectoryProvider(user_agent=user_agent)


@pytest.mark.parametrize("timeout", [0, -1, 10.01, float("inf")])
def test_provider_rejects_unbounded_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="between"):
        SecSymbolDirectoryProvider(
            user_agent="MarketView/1.0 operations@example.com",
            timeout_seconds=timeout,
        )
