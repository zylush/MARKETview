import asyncio
import logging
from typing import Any

import httpx
import pytest

from app.cache.memory import MemoryCache
from app.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.marketstack import MarketstackProvider
from app.services.market_data import MarketDataService, ServiceTTLs


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_adapter_maps_v2_tickers_and_pagination_without_leaking_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"symbol": "MSFT", "name": "Microsoft", "stock_exchange": {"mic": "XNAS"}}
                ],
                "pagination": {"offset": 0, "limit": 1, "count": 1, "total": 2},
            },
        )

    async def scenario():  # type: ignore[no-untyped-def]
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("super-secret", client=client)
            return await provider.list_tickers(limit=1, cursor="0")
        finally:
            await client.aclose()

    page = run(scenario())

    assert page.items[0].symbol == "MSFT"
    assert page.next_cursor == "1"
    assert seen[0].url.params["access_key"] == "super-secret"
    assert "access_key" not in seen[0].headers


def test_adapter_redacts_query_credentials_from_http_client_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [], "pagination": {"offset": 0, "limit": 1, "count": 0}},
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("super-secret", client=client)
            await provider.list_tickers(limit=1)
        finally:
            await client.aclose()

    caplog.set_level(logging.INFO, logger="httpx")
    run(scenario())

    assert "super-secret" not in caplog.text
    assert "access_key=REDACTED" in caplog.text
    assert "limit=1" in caplog.text


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, ProviderAuthenticationError), (429, ProviderRateLimitError)],
)
def test_adapter_maps_errors_to_sanitized_domain_errors(
    status: int, expected: type[Exception]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"message": f"bad super-secret {request.url}"}}
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("super-secret", client=client)
            with pytest.raises(expected) as caught:
                await provider.latest_eod("MSFT")
            assert "super-secret" not in str(caught.value)
            assert "access_key" not in str(caught.value)
        finally:
            await client.aclose()

    run(scenario())


def test_provider_error_statuses_distinguish_bad_gateway_from_timeout() -> None:
    assert ProviderUnavailableError.status_code == 502
    assert ProviderTimeoutError.status_code == 504


@pytest.mark.parametrize(
    ("failure_type", "expected"),
    [
        (httpx.ReadTimeout, ProviderTimeoutError),
        (httpx.ConnectError, ProviderUnavailableError),
    ],
)
def test_adapter_maps_transport_failures_to_sanitized_domain_errors(
    failure_type: type[httpx.HTTPError], expected: type[Exception]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure_type("transport failed", request=request)

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("super-secret", client=client)
            with pytest.raises(expected) as caught:
                await provider.latest_eod("MSFT")
            assert "super-secret" not in str(caught.value)
            assert "access_key" not in str(caught.value)
        finally:
            await client.aclose()

    run(scenario())


def test_adapter_maps_history_and_corporate_actions() -> None:
    responses = {
        "/v2/eod": {
            "data": [
                {
                    "symbol": "MSFT",
                    "date": "2025-01-02T00:00:00+0000",
                    "open": 1,
                    "high": 3,
                    "low": 1,
                    "close": 2,
                    "volume": 4,
                    "adj_close": 2.5,
                }
            ],
            "pagination": {"offset": 0, "limit": 100, "count": 1, "total": 1},
        },
        "/v2/splits": {"data": [{"symbol": "MSFT", "date": "2025-01-02", "split_factor": "2.0"}]},
        "/v2/dividends": {"data": [{"symbol": "MSFT", "date": "2025-01-02", "dividend": "0.75"}]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    async def scenario():  # type: ignore[no-untyped-def]
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("key", client=client)
            history = await provider.eod_history(
                "MSFT", start_date="2025-01-01", end_date="2025-01-02"
            )
            splits = await provider.splits("MSFT")
            dividends = await provider.dividends("MSFT")
            return history, splits, dividends
        finally:
            await client.aclose()

    history, splits, dividends = run(scenario())

    assert history.items[0].close == 2
    assert history.items[0].adjusted_close == 2.5
    assert splits.items[0].ratio == 2
    assert dividends.items[0].amount == 0.75


def test_adapter_validates_history_range_before_network_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": []})

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("key", client=client)
            with pytest.raises(ValueError, match="one year"):
                await provider.eod_history("MSFT", start_date="2024-01-01", end_date="2025-01-02")
        finally:
            await client.aclose()

    run(scenario())

    assert calls == 0


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        ("tickers", {"data": [], "pagination": {"offset": "private-value"}}),
        ("tickers", {"data": ["private-value"]}),
        ("exchanges", {"data": [{"mic": "XNAS"}]}),
        ("latest", {"data": [{"symbol": "MSFT", "date": "private-value"}]}),
        ("splits", {"data": [{"date": "2026-01-01", "split_factor": 2}]}),
        (
            "dividends",
            {"data": [{"symbol": "MSFT", "date": "2026-01-01", "dividend": "bad"}]},
        ),
        ("usage", {"data": {"requests": "private-value"}}),
    ],
)
def test_adapter_sanitizes_all_malformed_upstream_payloads(
    operation: str, payload: dict[str, Any]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def invoke(provider: MarketstackProvider) -> object:
        if operation == "tickers":
            return await provider.list_tickers()
        if operation == "exchanges":
            return await provider.list_exchanges()
        if operation == "latest":
            return await provider.latest_eod("MSFT")
        if operation == "splits":
            return await provider.splits("MSFT")
        if operation == "dividends":
            return await provider.dividends("MSFT")
        return await provider.usage()

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("key", client=client)
            with pytest.raises(ProviderUnavailableError) as caught:
                await invoke(provider)
            assert "private-value" not in str(caught.value)
        finally:
            await client.aclose()

    run(scenario())


def test_service_uses_stale_data_when_upstream_normalization_fails() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={"data": [{"symbol": "MSFT", "date": "2026-01-01", "close": 100}]},
            )
        return httpx.Response(200, json={"data": [{"symbol": "MSFT", "date": "malformed-private"}]})

    async def scenario() -> None:
        clock = [1000.0]
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("key", client=client)
            service = MarketDataService(provider, MemoryCache(clock=lambda: clock[0]))
            fresh = await service.latest_eod("MSFT")
            clock[0] += ServiceTTLs().latest + 1
            stale = await service.latest_eod("MSFT")
            assert stale == fresh
            assert service.last_metadata is not None
            assert service.last_metadata.stale
        finally:
            await client.aclose()

    run(scenario())


def test_service_propagates_sanitized_normalization_error_without_stale_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"pagination": {"offset": "private-value"}})

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            service = MarketDataService(MarketstackProvider("key", client=client), MemoryCache())
            with pytest.raises(ProviderUnavailableError) as caught:
                await service.tickers()
            assert "private-value" not in str(caught.value)
        finally:
            await client.aclose()

    run(scenario())
