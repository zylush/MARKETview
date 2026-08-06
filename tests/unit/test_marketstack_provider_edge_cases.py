from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pytest

from app.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderValidationError,
)
from app.providers.marketstack import MarketstackProvider


async def _provider_for(handler: Any) -> tuple[MarketstackProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return MarketstackProvider("secret", client=client), client


def test_provider_requires_an_access_key() -> None:
    with pytest.raises(ValueError, match="access key"):
        MarketstackProvider("")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (403, "", ProviderAuthenticationError),
        (404, "", ProviderNotFoundError),
        (400, "", ProviderValidationError),
        (500, "", ProviderUnavailableError),
        (418, "", ProviderError),
        (200, "invalid_access_key", ProviderAuthenticationError),
        (200, "usage_limit_reached", ProviderRateLimitError),
        (200, "invalid_api_function", ProviderNotFoundError),
        (200, "validation_error", ProviderValidationError),
        (200, "internal_error", ProviderUnavailableError),
    ],
)
async def test_provider_maps_http_and_payload_error_taxonomy(
    status: int, code: str, expected: type[Exception]
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        payload = {"error": {"code": code, "message": "secret details"}}
        return httpx.Response(status, json=payload)

    provider, client = await _provider_for(handler)
    try:
        with pytest.raises(expected) as caught:
            await provider.latest_eod("MSFT")
    finally:
        await client.aclose()

    assert "secret details" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["network", "invalid-json", "list-payload"])
async def test_provider_sanitizes_transport_and_malformed_payloads(failure: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectError("secret host", request=request)
        if failure == "invalid-json":
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(200, json=[])

    provider, client = await _provider_for(handler)
    try:
        with pytest.raises(ProviderUnavailableError) as caught:
            await provider.latest_eod("MSFT")
    finally:
        await client.aclose()

    assert "secret host" not in str(caught.value)


@pytest.mark.asyncio
async def test_provider_maps_exchange_search_latest_and_usage_shapes() -> None:
    seen: list[httpx.Request] = []
    payloads = {
        "/v2/exchanges": {
            "data": {"name": "NASDAQ", "mic": "XNAS"},
            "pagination": {"offset": 0, "limit": 10, "count": 1, "total": 1},
        },
        "/v2/eod/latest": {"data": [{"symbol": "MSFT", "date": "2026-08-07", "close": "12.50"}]},
        "/v2/usage": {"data": {"current": 4, "limit": 90, "remaining": 86}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payloads[request.url.path])

    provider, client = await _provider_for(handler)
    try:
        exchanges = await provider.list_exchanges(search="  nasdaq  ", limit=10)
        latest = await provider.latest_eod("msft")
        usage = await provider.usage()
    finally:
        await client.aclose()

    assert exchanges.items[0].mic == "XNAS"
    assert latest.symbol == "MSFT"
    assert usage.requests_used == 4
    assert usage.requests_limit == 90
    assert seen[0].url.params["search"] == "nasdaq"


@pytest.mark.asyncio
async def test_provider_reports_empty_latest_and_invalid_usage() -> None:
    replies: Any = iter([{"data": []}, {"data": []}])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(replies))

    provider, client = await _provider_for(handler)
    try:
        with pytest.raises(ProviderNotFoundError):
            await provider.latest_eod("MSFT")
        with pytest.raises(ProviderUnavailableError):
            await provider.usage()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_provider_validates_action_dates_before_network_and_maps_fallback_fields() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path.endswith("splits"):
            return httpx.Response(
                200, json={"data": [{"symbol": "MSFT", "date": "2026-01-02", "ratio": "2"}]}
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"symbol": "MSFT", "date": "2026-01-02", "amount": "0.5", "currency": "USD"}
                ]
            },
        )

    provider, client = await _provider_for(handler)
    try:
        split = await provider.splits("MSFT", start_date=date(2026, 1, 1), end_date="2026-01-03")
        dividend = await provider.dividends("MSFT")
        with pytest.raises(ProviderValidationError, match="provided together"):
            await provider.splits("MSFT", start_date="2026-01-01")
        with pytest.raises(ProviderValidationError, match="ISO"):
            await provider.dividends("MSFT", start_date="bad", end_date="2026-01-02")
    finally:
        await client.aclose()

    assert split.items[0].ratio == 2
    assert dividend.items[0].amount == 0.5
    assert calls == 2


@pytest.mark.asyncio
async def test_provider_rejects_invalid_corporate_action_decimal() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"symbol": "MSFT", "date": "2026-01-02", "split_factor": "bad"}]},
        )

    provider, client = await _provider_for(handler)
    try:
        with pytest.raises(ProviderUnavailableError, match="invalid response"):
            await provider.splits("MSFT")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_provider_context_manager_closes_owned_client() -> None:
    provider = MarketstackProvider("secret")
    async with provider as entered:
        assert entered is provider
    assert provider._client.is_closed
