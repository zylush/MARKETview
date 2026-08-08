from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import httpx
import pytest

from app.errors import (
    ProviderAccessRestrictedError,
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


@pytest.mark.parametrize(
    "access_key",
    [
        "   ",
        "<your-marketstack-api-key>",
        '"<your-marketstack-api-key>"',
        "replace-me",
        "key\nwith-control",
        "key\n",
    ],
)
def test_provider_rejects_invalid_access_keys_before_network(access_key: str) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError, match="access key") as caught:
            MarketstackProvider(access_key, client=client)
    finally:
        asyncio.run(client.aclose())

    assert access_key not in str(caught.value)
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        (401, {}, ProviderAuthenticationError),
        (403, {}, ProviderAccessRestrictedError),
        (404, {}, ProviderUnavailableError),
        (400, {}, ProviderUnavailableError),
        (500, {}, ProviderUnavailableError),
        (418, {}, ProviderError),
        (200, {"code": "missing_access_key"}, ProviderAuthenticationError),
        (200, {"code": "invalid_access_key"}, ProviderAuthenticationError),
        (200, {"type": "inactive_user"}, ProviderAuthenticationError),
        (200, {"code": "usage_limit_reached"}, ProviderRateLimitError),
        (200, {"type": "rate_limit_reached"}, ProviderRateLimitError),
        (200, {"type": "too_many_requests"}, ProviderRateLimitError),
        (
            403,
            {"code": 104, "type": "function_access_restricted"},
            ProviderAccessRestrictedError,
        ),
        (200, {"type": "https_access_restricted"}, ProviderAccessRestrictedError),
        (200, {"code": "invalid_api_function"}, ProviderUnavailableError),
        (200, {"code": "404_not_found"}, ProviderUnavailableError),
        (200, {"code": "validation_error"}, ProviderValidationError),
        (200, {"code": "internal_error"}, ProviderUnavailableError),
    ],
)
async def test_provider_maps_http_and_payload_error_taxonomy(
    status: int, error: dict[str, object], expected: type[ProviderError]
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        payload = {"error": {**error, "message": "secret details"}}
        return httpx.Response(status, json=payload)

    provider, client = await _provider_for(handler)
    try:
        with pytest.raises(expected) as caught:
            await provider.latest_eod("MSFT")
    finally:
        await client.aclose()

    assert "secret details" not in str(caught.value)
    assert isinstance(caught.value, ProviderError)
    assert caught.value.upstream_status == status
    assert caught.value.status_code == expected.status_code
    assert caught.value.semantic_code not in {"secret", "secret details"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "response_kwargs", "expected"),
    [
        (401, {"content": b"<html>private</html>"}, ProviderAuthenticationError),
        (429, {"json": ["private"]}, ProviderRateLimitError),
        (422, {"content": b""}, ProviderValidationError),
    ],
)
async def test_http_status_taxonomy_survives_malformed_error_bodies(
    status: int,
    response_kwargs: dict[str, object],
    expected: type[ProviderError],
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, **response_kwargs)

    provider, client = await _provider_for(handler)
    try:
        with pytest.raises(expected) as caught:
            await provider.latest_eod("MSFT")
    finally:
        await client.aclose()

    assert caught.value.status_code == expected.status_code
    assert caught.value.upstream_status == status
    assert "private" not in str(caught.value)


@pytest.mark.asyncio
async def test_known_validation_semantic_wins_over_generic_http_400_fallback() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"type": "validation_error", "message": "private details"}},
        )

    provider, client = await _provider_for(handler)
    try:
        with pytest.raises(ProviderValidationError) as caught:
            await provider.latest_eod("MSFT")
    finally:
        await client.aclose()

    assert caught.value.upstream_status == 400
    assert caught.value.semantic_code == "validation_error"


@pytest.mark.asyncio
async def test_unknown_upstream_semantic_is_not_exposed_for_runtime_logging() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"type": "private_semantic_sentinel"}},
        )

    provider, client = await _provider_for(handler)
    try:
        with pytest.raises(ProviderUnavailableError) as caught:
            await provider.latest_eod("MSFT")
    finally:
        await client.aclose()

    assert caught.value.semantic_code == "unknown"


@pytest.mark.asyncio
async def test_provider_does_not_retry_transport_failures() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("private endpoint", request=request)

    provider, client = await _provider_for(handler)
    try:
        with pytest.raises(ProviderUnavailableError):
            await provider.latest_eod("MSFT")
    finally:
        await client.aclose()

    assert calls == 1


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
