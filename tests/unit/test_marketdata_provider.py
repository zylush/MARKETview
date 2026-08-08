from __future__ import annotations

import copy
import logging
from datetime import date
from decimal import Decimal
from typing import Any

import httpx
import pytest

from app.errors import (
    InputValidationError,
    ProviderAccessRestrictedError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderRequestRejectedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.marketdata import MarketDataAppProvider

PROVIDER_TEST_SENTINEL = "marketdata-provider-test-value"


def candles_payload(**overrides: Any) -> dict[str, Any]:
    return {
        "s": "ok",
        "o": [181.0, 182.25],
        "h": [184.0, 185.5],
        "l": [180.5, 181.75],
        "c": [183.5, 184.75],
        "v": [1_000_000, 1_100_000],
        "t": [1704067200, 1704153600],
        **overrides,
    }


def provider_for(
    handler: Any, *, token: str = PROVIDER_TEST_SENTINEL
) -> tuple[MarketDataAppProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = MarketDataAppProvider(
        token,
        base_url="https://api.marketdata.app/v1",
        client=client,
    )
    return provider, client


@pytest.mark.asyncio
async def test_latest_uses_exact_candles_path_query_and_bearer_header() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json=candles_payload(o=[181], h=[184], l=[180], c=[183], v=[9], t=[1704067200]),
        )

    provider, client = provider_for(handler)
    try:
        bar = await provider.latest_eod("aapl")
    finally:
        await client.aclose()

    assert bar.symbol == "AAPL"
    assert bar.date == date(2024, 1, 1)
    assert bar.open == Decimal("181")
    assert bar.high == Decimal("184")
    assert bar.low == Decimal("180")
    assert bar.close == Decimal("183")
    assert bar.volume == 9
    assert bar.adjusted_open is None
    assert bar.adjusted_high is None
    assert bar.adjusted_low is None
    assert bar.adjusted_close is None
    assert bar.adjusted_volume is None
    assert len(captured) == 1
    request = captured[0]
    assert request.method == "GET"
    assert request.url.path == "/v1/stocks/candles/D/AAPL/"
    assert dict(request.url.params) == {
        "countback": "1",
        "adjustsplits": "false",
        "adjustdividends": "false",
    }
    assert "to" not in request.url.params
    assert request.headers.get_list("authorization") == [f"Bearer {PROVIDER_TEST_SENTINEL}"]
    assert request.headers.get_list("accept") == ["application/json"]
    assert PROVIDER_TEST_SENTINEL not in str(request.url)
    assert not {"token", "access_key", "api_key"} & set(request.url.params.keys())


@pytest.mark.asyncio
async def test_history_uses_exact_query_and_applies_immutable_local_pagination() -> None:
    payload = candles_payload(
        o=[3, 1, 2],
        h=[4, 2, 3],
        l=[2, 0, 1],
        c=[3.5, 1.5, 2.5],
        v=[30, 10, 20],
        t=[1704240000, 1704067200, 1704153600],
    )
    original = copy.deepcopy(payload)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(203, json=payload)

    provider, client = provider_for(handler)
    try:
        page = await provider.eod_history(
            "aapl",
            start_date=date(2024, 1, 1),
            end_date="2024-01-03",
            limit=1,
            cursor="1",
        )
    finally:
        await client.aclose()

    assert payload == original
    assert [bar.date for bar in page.items] == [date(2024, 1, 2)]
    assert page.items[0].close == Decimal("2.5")
    assert page.total == 3
    assert page.next_cursor == "2"
    assert len(captured) == 1
    assert captured[0].url.path == "/v1/stocks/candles/D/AAPL/"
    assert dict(captured[0].url.params) == {
        "from": "2024-01-01",
        "to": "2024-01-03",
        "adjustsplits": "false",
        "adjustdividends": "false",
    }
    assert not {"limit", "cursor", "offset", "token", "access_key", "api_key"} & set(
        captured[0].url.params
    )
    assert captured[0].headers.get_list("authorization") == [f"Bearer {PROVIDER_TEST_SENTINEL}"]
    assert PROVIDER_TEST_SENTINEL not in str(captured[0].url)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 203])
async def test_200_and_203_are_success(status: int) -> None:
    provider, client = provider_for(
        lambda request: httpx.Response(
            status,
            json=candles_payload(o=[1], h=[2], l=[0], c=[1.5], v=[1], t=[1704067200]),
        )
    )
    try:
        result = await provider.latest_eod("AAPL")
    finally:
        await client.aclose()

    assert result.date == date(2024, 1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (204, None, ProviderNotFoundError),
        (200, {"s": "no_data"}, ProviderNotFoundError),
        (404, {"s": "error", "errmsg": "private symbol details"}, ProviderNotFoundError),
        (400, {"s": "error", "errmsg": "private query"}, ProviderRequestRejectedError),
        (413, {"s": "error"}, ProviderRequestRejectedError),
        (422, {"s": "error", "errmsg": "private query"}, ProviderRequestRejectedError),
        (401, {"s": "error", "errmsg": "private auth"}, ProviderAuthenticationError),
        (402, {"s": "error", "errmsg": "private plan"}, ProviderAccessRestrictedError),
        (403, {"s": "error", "errmsg": "private IP"}, ProviderAccessRestrictedError),
        (429, {"s": "error", "errmsg": "private credits"}, ProviderRateLimitError),
        (500, {"s": "error"}, ProviderUnavailableError),
        (502, {"s": "error"}, ProviderUnavailableError),
        (503, {"s": "error"}, ProviderUnavailableError),
        (504, {"s": "error"}, ProviderTimeoutError),
        (509, {"s": "error"}, ProviderUnavailableError),
        (524, {"s": "error"}, ProviderTimeoutError),
        (529, {"s": "error"}, ProviderUnavailableError),
        (530, {"s": "error"}, ProviderUnavailableError),
        (540, {"s": "error"}, ProviderUnavailableError),
        (598, {"s": "error"}, ProviderUnavailableError),
    ],
)
async def test_status_taxonomy_is_sanitized(
    status: int, payload: dict[str, Any] | None, expected: type[ProviderError]
) -> None:
    private = "private-provider-body-sentinel"
    body = payload if payload is None else {**payload, "private": private}
    provider, client = provider_for(
        lambda request: (
            httpx.Response(status, json=body) if body is not None else httpx.Response(status)
        )
    )
    try:
        with pytest.raises(expected) as captured:
            await provider.latest_eod("AAPL")
    finally:
        await client.aclose()

    assert captured.value.upstream_status == status
    assert private not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_success_status_with_error_semantic_is_sanitized_failure() -> None:
    provider, client = provider_for(
        lambda request: httpx.Response(
            200,
            json={"s": "error", "errmsg": "provider-private-message", "code": "private-code"},
        )
    )
    try:
        with pytest.raises(ProviderError) as captured:
            await provider.latest_eod("AAPL")
    finally:
        await client.aclose()

    assert str(captured.value) == "market data provider request failed"
    assert captured.value.upstream_status == 200
    assert captured.value.semantic_code == "provider_error"
    assert "provider-private-message" not in repr(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"s": "unexpected"},
        {"s": "ok"},
        candles_payload(o="not-an-array"),
        candles_payload(t=[1704067200]),
        candles_payload(o=[], h=[], l=[], c=[], v=[], t=[]),
        candles_payload(o=[True, 2]),
        candles_payload(c=["NaN", 2]),
        candles_payload(v=[-1, 2]),
        candles_payload(v=[1.5, 2]),
        candles_payload(t=["not-a-timestamp", 1704153600]),
    ],
)
async def test_malformed_parallel_arrays_are_rejected_without_payload_disclosure(
    payload: dict[str, Any],
) -> None:
    aliased = payload
    original = copy.deepcopy(payload)
    aliased["private"] = "private-array-payload-sentinel"
    original["private"] = "private-array-payload-sentinel"
    provider, client = provider_for(lambda request: httpx.Response(200, json=aliased))
    try:
        with pytest.raises(ProviderUnavailableError) as captured:
            await provider.latest_eod("AAPL")
    finally:
        await client.aclose()

    assert aliased == original
    assert "private-array-payload-sentinel" not in str(captured.value)
    assert captured.value.upstream_status == 200


@pytest.mark.asyncio
async def test_invalid_inputs_make_no_requests() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    provider, client = provider_for(handler)
    try:
        with pytest.raises(InputValidationError):
            await provider.latest_eod("bad symbol!")
        with pytest.raises(InputValidationError):
            await provider.eod_history("AAPL", start_date="2024-02-02", end_date="2024-01-01")
        with pytest.raises(InputValidationError):
            await provider.eod_history(
                "AAPL", start_date="2024-01-01", end_date="2024-01-02", cursor="x"
            )
    finally:
        await client.aclose()

    assert calls == 0


@pytest.mark.asyncio
async def test_transport_timeout_and_http_failure_are_not_retried() -> None:
    timeout_calls = 0
    failure_calls = 0

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        nonlocal timeout_calls
        timeout_calls += 1
        raise httpx.ReadTimeout("private-timeout", request=request)

    timeout_provider, timeout_client = provider_for(timeout_handler)
    try:
        with pytest.raises(ProviderTimeoutError) as timeout_error:
            await timeout_provider.latest_eod("AAPL")
    finally:
        await timeout_client.aclose()

    def failure_handler(request: httpx.Request) -> httpx.Response:
        nonlocal failure_calls
        failure_calls += 1
        raise httpx.ConnectError("private-connect", request=request)

    failure_provider, failure_client = provider_for(failure_handler)
    try:
        with pytest.raises(ProviderUnavailableError) as failure_error:
            await failure_provider.latest_eod("AAPL")
    finally:
        await failure_client.aclose()

    assert timeout_calls == 1
    assert failure_calls == 1
    assert "private-timeout" not in str(timeout_error.value)
    assert "private-connect" not in str(failure_error.value)
    assert timeout_error.value.__cause__ is None
    assert failure_error.value.__cause__ is None


@pytest.mark.asyncio
async def test_http_client_logs_are_suppressed_only_during_provider_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        logging.getLogger("httpx.security-test").warning(
            "Authorization: Bearer %s", PROVIDER_TEST_SENTINEL
        )
        return httpx.Response(
            200,
            json=candles_payload(o=[1], h=[2], l=[0], c=[1], v=[1], t=[1704067200]),
        )

    provider, client = provider_for(handler)
    with caplog.at_level(logging.WARNING):
        try:
            await provider.latest_eod("AAPL")
        finally:
            await client.aclose()
        logging.getLogger("httpx.security-test").warning("unrelated-safe-log")

    assert PROVIDER_TEST_SENTINEL not in caplog.text
    assert "unrelated-safe-log" in caplog.text


@pytest.mark.asyncio
async def test_provider_exception_graph_and_traceback_locals_are_secret_free() -> None:
    payload_secret = "private-payload-traceback-sentinel"
    provider, client = provider_for(
        lambda request: httpx.Response(
            401,
            json={"s": "error", "errmsg": payload_secret, "token": PROVIDER_TEST_SENTINEL},
        )
    )
    try:
        with pytest.raises(ProviderAuthenticationError) as captured:
            await provider.latest_eod("AAPL")
    finally:
        await client.aclose()

    pending: list[BaseException] = [captured.value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert PROVIDER_TEST_SENTINEL not in str(current)
        assert payload_secret not in str(current)
        traceback = current.__traceback__
        while traceback is not None:
            if "app\\providers" in traceback.tb_frame.f_code.co_filename.lower():
                locals_surface = repr(traceback.tb_frame.f_locals)
                assert PROVIDER_TEST_SENTINEL not in locals_surface
                assert payload_secret not in locals_surface
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_injected_client_is_not_closed_but_owned_client_is_closed() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=candles_payload(o=[1], h=[2], l=[0], c=[1], v=[1], t=[1704067200]),
            )
        )
    )
    injected = MarketDataAppProvider(PROVIDER_TEST_SENTINEL, client=client)
    await injected.aclose()
    assert not client.is_closed
    await client.aclose()

    owned = MarketDataAppProvider(PROVIDER_TEST_SENTINEL)
    await owned.aclose()
    assert owned._client.is_closed


@pytest.mark.parametrize(
    "token",
    ["", "  ", "token\ncontrol", '"quoted-token"', "<your-marketdata-token>", "example"],
)
def test_provider_rejects_invalid_token_without_disclosure(token: str) -> None:
    with pytest.raises(ValueError, match="Market Data token is required") as captured:
        MarketDataAppProvider(token)

    assert token not in str(captured.value) or not token


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.marketdata.app",
        "https://api.marketdata.app/v2",
        "https://user:pass@api.marketdata.app/v1",
        "https://api.marketdata.app/v1?x=y",
        "https://api.marketdata.app/v1#x",
    ],
)
def test_provider_requires_a_well_formed_v1_root(base_url: str) -> None:
    with pytest.raises(ValueError, match="v1 API root"):
        MarketDataAppProvider(PROVIDER_TEST_SENTINEL, base_url=base_url)
