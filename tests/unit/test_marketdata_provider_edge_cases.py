from __future__ import annotations

import copy
import logging
from datetime import date
from typing import Any

import httpx
import pytest

from app.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from app.providers.marketdata import MarketDataAppProvider
from tests.unit.test_marketdata_provider import TOKEN, candles_payload, provider_for


class _AliasedResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _RaisingResponse:
    status_code = 200

    def json(self) -> object:
        raise RecursionError("private-decoder-recursion-sentinel")


class _InjectedClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []
        self.closed = False

    async def get(self, url: str, **kwargs: object) -> object:
        self.requests.append({"url": url, **kwargs})
        return self.response

    async def aclose(self) -> None:
        self.closed = True


class _UnexpectedFailureClient:
    calls = 0

    async def get(self, url: str, **kwargs: object) -> object:
        self.calls += 1
        raise RuntimeError("private-unexpected-transport-sentinel")

    async def aclose(self) -> None:
        return None


def _provider_with_client(client: Any) -> MarketDataAppProvider:
    return MarketDataAppProvider(
        TOKEN,
        base_url="https://api.marketdata.app/v1",
        client=client,
    )


def _provider_traceback_locals(error: BaseException) -> list[str]:
    surfaces: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if "app\\providers" in traceback.tb_frame.f_code.co_filename.lower():
            surfaces.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return surfaces


@pytest.mark.asyncio
async def test_deep_aliased_payload_is_not_mutated_or_retained_by_failure() -> None:
    payload: dict[str, Any] = candles_payload(o=[1], h=[2], l=[0], c=[1], v=[1], t=[1704067200])
    private_branch = {"token": TOKEN, "message": "private-deep-payload-sentinel"}
    payload["private"] = {"left": private_branch, "right": private_branch}
    original = copy.deepcopy(payload)
    client = _InjectedClient(_AliasedResponse(200, payload))
    provider = _provider_with_client(client)

    result = await provider.latest_eod("AAPL")

    assert result.date == date(2024, 1, 1)
    assert payload == original
    assert payload["private"]["left"] is payload["private"]["right"]
    assert TOKEN not in repr(provider.__dict__)


@pytest.mark.asyncio
async def test_malformed_aliased_payload_failure_does_not_mutate_or_retain_payload() -> None:
    payload: dict[str, Any] = candles_payload(t=[1704067200])
    payload["private"] = {"token": TOKEN, "message": "private-malformed-payload-sentinel"}
    original = copy.deepcopy(payload)
    client = _InjectedClient(_AliasedResponse(200, payload))
    provider = _provider_with_client(client)

    with pytest.raises(ProviderUnavailableError) as captured:
        await provider.latest_eod("AAPL")

    assert payload == original
    assert "private-malformed-payload-sentinel" not in str(captured.value)
    for surface in _provider_traceback_locals(captured.value):
        assert TOKEN not in surface
        assert "private-malformed-payload-sentinel" not in surface


@pytest.mark.asyncio
async def test_json_decoder_recursion_is_detached_without_response_retention() -> None:
    client = _InjectedClient(_RaisingResponse())
    provider = _provider_with_client(client)

    with pytest.raises(ProviderUnavailableError) as captured:
        await provider.latest_eod("AAPL")

    assert "private-decoder-recursion-sentinel" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    for surface in _provider_traceback_locals(captured.value):
        assert TOKEN not in surface
        assert "private-decoder-recursion-sentinel" not in surface


@pytest.mark.asyncio
async def test_unexpected_transport_exception_is_sanitized_without_retry_or_retention() -> None:
    client = _UnexpectedFailureClient()
    provider = _provider_with_client(client)

    with pytest.raises(ProviderUnavailableError) as captured:
        await provider.eod_history(
            "PRIVATE-SYMBOL",
            start_date="2001-02-03",
            end_date="2001-02-04",
            cursor="87654321",
        )

    assert client.calls == 1
    assert "private-unexpected-transport-sentinel" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    forbidden = (
        TOKEN,
        "PRIVATE-SYMBOL",
        "2001-02-03",
        "2001-02-04",
        "87654321",
        "Authorization",
        "adjustsplits",
        "private-unexpected-transport-sentinel",
    )
    for surface in _provider_traceback_locals(captured.value):
        assert all(value not in surface for value in forbidden)


@pytest.mark.asyncio
async def test_late_httpx_and_httpcore_child_loggers_cannot_bypass_scrubbing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unique = str(id(object()))

    def handler(request: httpx.Request) -> httpx.Response:
        logging.getLogger(f"httpx.late.{unique}").error(
            "url=%s Authorization=Bearer %s", request.url, TOKEN
        )
        logging.getLogger(f"httpcore.late.{unique}").error("headers Authorization=Bearer %s", TOKEN)
        logging.getLogger(f"application.safe.{unique}").warning("unrelated-log-preserved")
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

    assert TOKEN not in caplog.text
    assert "Authorization" not in caplog.text
    assert "unrelated-log-preserved" in caplog.text


@pytest.mark.asyncio
async def test_private_history_inputs_are_absent_from_provider_exception_graph() -> None:
    private_symbol = "PRIVATE-SYMBOL-987654321"
    private_cursor = "87654321"
    private_start = "2001-02-03"
    private_end = "2001-02-04"
    provider, client = provider_for(
        lambda request: httpx.Response(
            401,
            json={"s": "error", "errmsg": "private-body-sentinel", "token": TOKEN},
        )
    )
    try:
        with pytest.raises(ProviderAuthenticationError) as captured:
            await provider.eod_history(
                private_symbol,
                start_date=private_start,
                end_date=private_end,
                limit=997,
                cursor=private_cursor,
            )
    finally:
        await client.aclose()

    forbidden = (
        TOKEN,
        private_symbol,
        private_cursor,
        private_start,
        private_end,
        "private-body-sentinel",
        "Authorization",
        "adjustsplits",
    )
    pending: list[BaseException] = [captured.value]
    while pending:
        current = pending.pop()
        assert all(value not in str(current) and value not in repr(current) for value in forbidden)
        for surface in _provider_traceback_locals(current):
            assert all(value not in surface for value in forbidden)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        candles_payload(o=[float("inf"), 2]),
        candles_payload(h=[float("-inf"), 2]),
        candles_payload(l=[True, 2]),
        candles_payload(c=[False, 2]),
        candles_payload(v=[True, 2]),
        candles_payload(v=[float("inf"), 2]),
        candles_payload(t=[True, 1704153600]),
        candles_payload(t=[float("inf"), 1704153600]),
    ],
)
async def test_bool_and_non_finite_numeric_values_are_rejected(payload: dict[str, Any]) -> None:
    provider, client = provider_for(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(ProviderUnavailableError):
            await provider.latest_eod("AAPL")
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "response_kwargs", "expected"),
    [
        (401, {"content": b"<html>private auth body</html>"}, ProviderAuthenticationError),
        (429, {"json": ["private rate body"]}, ProviderRateLimitError),
        (500, {"content": b"private server body"}, ProviderUnavailableError),
    ],
)
async def test_status_mapping_does_not_depend_on_parsing_private_error_bodies(
    status: int,
    response_kwargs: dict[str, object],
    expected: type[ProviderError],
) -> None:
    provider, client = provider_for(lambda request: httpx.Response(status, **response_kwargs))
    try:
        with pytest.raises(expected) as captured:
            await provider.latest_eod("AAPL")
    finally:
        await client.aclose()

    assert "private" not in str(captured.value)
    assert captured.value.upstream_status == status


@pytest.mark.asyncio
async def test_redirect_is_not_followed_and_authorization_is_not_forwarded() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://evil.example/collect"})

    provider, client = provider_for(handler)
    try:
        with pytest.raises(ProviderError):
            await provider.latest_eod("AAPL")
    finally:
        await client.aclose()

    assert len(requests) == 1
    assert requests[0].url.host == "api.marketdata.app"


@pytest.mark.asyncio
async def test_outbound_headers_never_include_application_access_key() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json=candles_payload(o=[1], h=[2], l=[0], c=[1], v=[1], t=[1704067200]),
        )

    provider, client = provider_for(handler)
    try:
        await provider.latest_eod("AAPL")
    finally:
        await client.aclose()

    assert "x-app-key" not in captured[0].headers
    assert captured[0].headers.get_list("authorization") == [f"Bearer {TOKEN}"]


@pytest.mark.asyncio
async def test_context_manager_closes_owned_client() -> None:
    provider = MarketDataAppProvider(TOKEN)

    async with provider as entered:
        assert entered is provider

    assert provider._client.is_closed


@pytest.mark.parametrize(
    ("token", "base_url"),
    [
        ("private-invalid-token\n", "https://api.marketdata.app/v1"),
        (TOKEN, "https://private-url.example/not-v1?private=query"),
    ],
)
def test_constructor_validation_discards_private_inputs_from_traceback(
    token: str, base_url: str
) -> None:
    with pytest.raises(ValueError, match="Market Data") as captured:
        MarketDataAppProvider(token, base_url=base_url)

    for surface in _provider_traceback_locals(captured.value):
        assert token not in surface
        assert base_url not in surface
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
