import asyncio
import logging
from typing import Any

import httpx
import pytest

from app.cache.memory import MemoryCache
from app.errors import (
    InputValidationError,
    ProviderAccessRestrictedError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderValidationError,
)
from app.providers.marketstack import MarketstackProvider
from app.services.market_data import MarketDataService, ServiceTTLs


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def traceback_locals(error: BaseException) -> list[str]:
    rendered: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        rendered.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return rendered


def structure_fingerprint(root: object) -> tuple[tuple[object, ...], ...]:
    tokens: list[tuple[object, ...]] = []
    pending: list[object] = [root]
    identities: dict[int, int] = {}
    while pending:
        current = pending.pop()
        if isinstance(current, (dict, list, tuple)):
            identity = id(current)
            if identity in identities:
                tokens.append(("reference", identities[identity]))
                continue
            node_id = len(identities)
            identities[identity] = node_id
            if isinstance(current, dict):
                tokens.append(("dict", node_id, tuple(current.keys())))
                pending.extend(reversed(tuple(current.values())))
            else:
                tokens.append((type(current).__name__, node_id, len(current)))
                pending.extend(reversed(current))
        else:
            tokens.append(("value", type(current).__name__, current))
    return tuple(tokens)


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
    assert seen[0].url.params.get_list("access_key") == ["super-secret"]
    assert "api_key" not in seen[0].url.params
    assert "authorization" not in seen[0].headers
    assert "access_key" not in seen[0].headers
    assert "x-app-key" not in seen[0].headers


def test_adapter_encodes_special_key_without_query_parameter_injection() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"data": [], "pagination": {"offset": 0, "limit": 1, "count": 0}},
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("key&admin=true?access_key=other", client=client)
            await provider.list_tickers(limit=1)
        finally:
            await client.aclose()

    run(scenario())

    assert seen[0].url.path == "/v2/tickers"
    assert seen[0].url.params.get_list("access_key") == ["key&admin=true?access_key=other"]
    assert "admin" not in seen[0].url.params


def test_adapter_normalizes_trailing_slash_and_uses_exact_v2_paths() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/v2/eod/latest":
            return httpx.Response(
                200,
                json={"data": [{"symbol": "MSFT", "date": "2026-08-08"}]},
            )
        return httpx.Response(
            200,
            json={"data": [], "pagination": {"offset": 0, "limit": 1, "count": 0}},
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider(
                "key", base_url="https://api.marketstack.com/v2/", client=client
            )
            await provider.list_tickers(limit=1)
            await provider.list_exchanges(limit=1)
            await provider.eod_history(
                "MSFT", start_date="2026-08-07", end_date="2026-08-08", limit=1
            )
            await provider.latest_eod("MSFT")
        finally:
            await client.aclose()

    run(scenario())

    assert seen == ["/v2/tickers", "/v2/exchanges", "/v2/eod", "/v2/eod/latest"]
    assert all("/v2/v2" not in path and "//" not in path for path in seen)


def test_adapter_suppresses_all_http_client_request_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        logging.getLogger("httpcore.connection").debug(
            "connect https://api.marketstack.com/v2/tickers?limit=1"
        )
        logging.getLogger("app.test").warning("application diagnostic remains visible")
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

    caplog.set_level(logging.DEBUG, logger="httpx")
    caplog.set_level(logging.DEBUG, logger="httpcore")
    run(scenario())

    client_records = [
        record
        for record in caplog.records
        if record.name == "httpx"
        or record.name.startswith("httpx.")
        or record.name == "httpcore"
        or record.name.startswith("httpcore.")
    ]
    assert client_records == []
    assert "api.marketstack.com" not in caplog.text
    assert "/v2/tickers" not in caplog.text
    assert "access_key" not in caplog.text
    assert "limit=1" not in caplog.text
    assert "application diagnostic remains visible" in caplog.text

    logging.getLogger("httpx").info("unrelated HTTP client diagnostic")
    logging.getLogger("httpcore.connection").debug("unrelated HTTP core diagnostic")

    assert "unrelated HTTP client diagnostic" in caplog.text
    assert "unrelated HTTP core diagnostic" in caplog.text


def test_late_http_client_logger_cannot_bypass_request_log_scrubbing() -> None:
    captured: list[str] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    def handler(_: httpx.Request) -> httpx.Response:
        logging.getLogger("httpcore.late_child").warning(
            "connect https://api.marketstack.com/v2/tickers?access_key=super-secret"
        )
        return httpx.Response(
            200,
            json={"data": [], "pagination": {"offset": 0, "limit": 1, "count": 0}},
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = MarketstackProvider("super-secret", client=client)
        late_logger = logging.getLogger("httpcore.late_child")
        capture_handler = CaptureHandler()
        try:
            late_logger.addHandler(capture_handler)
            late_logger.propagate = False
            await provider.list_tickers(limit=1)
        finally:
            late_logger.removeHandler(capture_handler)
            late_logger.propagate = True
            await client.aclose()

    run(scenario())

    assert captured == ["HTTP client request details suppressed"]


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


def test_adapter_exposes_only_sanitized_error_metadata() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "error": {
                    "code": 104,
                    "type": "function_access_restricted",
                    "message": "private-body super-secret",
                }
            },
        )

    async def scenario() -> ProviderAccessRestrictedError:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("super-secret", client=client)
            with pytest.raises(ProviderAccessRestrictedError) as caught:
                await provider.latest_eod("MSFT")
            return caught.value
        finally:
            await client.aclose()

    caught = run(scenario())

    assert caught.upstream_status == 403
    assert caught.semantic_code == "function_access_restricted"
    assert "private-body" not in str(caught)
    assert "super-secret" not in str(caught)
    assert "access_key" not in str(caught)


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

    async def scenario() -> Exception:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("super-secret", client=client)
            with pytest.raises(expected) as caught:
                await provider.latest_eod("MSFT")
            return caught.value
        finally:
            await client.aclose()

    translated = run(scenario())
    graph: list[BaseException] = []
    pending: list[BaseException] = [translated]
    while pending:
        current = pending.pop()
        graph.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)

    assert graph == [translated]
    assert translated.__cause__ is None
    assert translated.__context__ is None
    assert "super-secret" not in repr(graph)
    assert "access_key" not in repr(graph)

    for rendered_locals in traceback_locals(translated):
        assert "super-secret" not in rendered_locals
        assert "access_key=super-secret" not in rendered_locals
        assert "'access_key': 'super-secret'" not in rendered_locals


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (401, {"error": {"type": "invalid_access_key"}}),
        (200, {"error": {"type": "invalid_access_key"}}),
    ],
)
def test_http_and_semantic_errors_do_not_retain_secret_response_in_traceback(
    status: int, payload: dict[str, Any]
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    async def scenario() -> Exception:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("super-secret", client=client)
            with pytest.raises(ProviderAuthenticationError) as caught:
                await provider.latest_eod("MSFT")
            return caught.value
        finally:
            await client.aclose()

    translated = run(scenario())

    for rendered_locals in traceback_locals(translated):
        assert "super-secret" not in rendered_locals
        assert "access_key=super-secret" not in rendered_locals
        assert "'access_key': 'super-secret'" not in rendered_locals


def test_json_decoder_failure_does_not_retain_secret_response_in_traceback() -> None:
    class FailingJsonResponse(httpx.Response):
        def json(self, **_: Any) -> Any:
            raise RecursionError("private-decoder-sentinel")

    def handler(request: httpx.Request) -> httpx.Response:
        return FailingJsonResponse(200, request=request)

    async def scenario() -> ProviderUnavailableError:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("super-secret", client=client)
            with pytest.raises(ProviderUnavailableError) as caught:
                await provider.latest_eod("MSFT")
            return caught.value
        finally:
            await client.aclose()

    translated = run(scenario())

    assert translated.__cause__ is None
    assert translated.__context__ is None
    assert "private-decoder-sentinel" not in repr(translated)
    for rendered_locals in traceback_locals(translated):
        assert "private-decoder-sentinel" not in rendered_locals
        assert "super-secret" not in rendered_locals
        assert "access_key=super-secret" not in rendered_locals


@pytest.mark.parametrize(
    "operation",
    ["tickers", "exchanges", "latest", "history", "splits", "dividends", "usage"],
)
@pytest.mark.parametrize("status", [401, 200])
def test_public_methods_discard_private_query_inputs_before_upstream_error(
    operation: str, status: int
) -> None:
    private_payload = "private-payload-sentinel"
    response_payload = {
        "error": {
            "type": "invalid_access_key",
            "message": private_payload,
        }
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=response_payload)

    async def invoke(provider: MarketstackProvider) -> object:
        if operation == "tickers":
            return await provider.list_tickers(
                search="private-search-sentinel", cursor="987654321012"
            )
        if operation == "exchanges":
            return await provider.list_exchanges(
                search="private-search-sentinel", cursor="987654321012"
            )
        if operation == "latest":
            return await provider.latest_eod("PRIVATE_SENTINEL")
        if operation == "history":
            return await provider.eod_history(
                "PRIVATE_SENTINEL",
                start_date="2099-11-29",
                end_date="2099-12-30",
                cursor="987654321012",
            )
        if operation == "splits":
            return await provider.splits(
                "PRIVATE_SENTINEL",
                start_date="2099-11-29",
                end_date="2099-12-30",
                cursor="987654321012",
            )
        if operation == "dividends":
            return await provider.dividends(
                "PRIVATE_SENTINEL",
                start_date="2099-11-29",
                end_date="2099-12-30",
                cursor="987654321012",
            )
        return await provider.usage()

    async def scenario() -> ProviderAuthenticationError:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("provider-key-sentinel", client=client)
            with pytest.raises(ProviderAuthenticationError) as caught:
                await invoke(provider)
            return caught.value
        finally:
            await client.aclose()

    translated = run(scenario())

    graph: list[BaseException] = []
    pending: list[BaseException] = [translated]
    while pending:
        current = pending.pop()
        graph.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)

    assert graph == [translated]
    forbidden = (
        "private-search-sentinel",
        "PRIVATE_SENTINEL",
        "987654321012",
        "2099-11-29",
        "2099-12-30",
        "provider-key-sentinel",
        "api.marketstack.com",
        "access_key",
        private_payload,
    )
    assert all(value not in repr(graph) for value in forbidden)
    for rendered_locals in traceback_locals(translated):
        assert all(value not in rendered_locals for value in forbidden)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("tickers", InputValidationError),
        ("exchanges", InputValidationError),
        ("latest", InputValidationError),
        ("history", ProviderValidationError),
        ("splits", ProviderValidationError),
        ("dividends", ProviderValidationError),
    ],
)
def test_public_methods_discard_private_inputs_before_validation_error(
    operation: str, expected: type[Exception]
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": []})

    async def invoke(provider: MarketstackProvider) -> object:
        if operation == "tickers":
            return await provider.list_tickers(
                search="private-search-sentinel", cursor="private-cursor-sentinel"
            )
        if operation == "exchanges":
            return await provider.list_exchanges(
                search="private-search-sentinel", cursor="private-cursor-sentinel"
            )
        if operation == "latest":
            return await provider.latest_eod("private symbol sentinel")
        if operation == "history":
            return await provider.eod_history(
                "PRIVATE_SENTINEL",
                start_date="private-date-sentinel",
                end_date="2099-12-30",
                cursor="987654321012",
            )
        if operation == "splits":
            return await provider.splits(
                "PRIVATE_SENTINEL",
                start_date="private-date-sentinel",
                end_date="2099-12-30",
                cursor="987654321012",
            )
        return await provider.dividends(
            "PRIVATE_SENTINEL",
            start_date="private-date-sentinel",
            end_date="2099-12-30",
            cursor="987654321012",
        )

    async def scenario() -> Exception:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("provider-key-sentinel", client=client)
            with pytest.raises(expected) as caught:
                await invoke(provider)
            return caught.value
        finally:
            await client.aclose()

    translated = run(scenario())

    assert calls == 0
    assert translated.__cause__ is None
    assert translated.__context__ is None
    forbidden = (
        "private-search-sentinel",
        "private-cursor-sentinel",
        "private symbol sentinel",
        "PRIVATE_SENTINEL",
        "private-date-sentinel",
        "2099-12-30",
        "987654321012",
        "provider-key-sentinel",
    )
    for rendered_locals in traceback_locals(translated):
        assert all(value not in rendered_locals for value in forbidden)


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
        (
            "tickers",
            {"data": [], "pagination": {"offset": {"nested": ["private-value"]}}},
        ),
        ("tickers", {"data": [{"nested": ["private-value"]}]}),
        ("exchanges", {"data": [{"mic": "XNAS", "nested": ["private-value"]}]}),
        (
            "latest",
            {
                "data": [
                    {
                        "symbol": "MSFT",
                        "date": {"nested": ["private-value"]},
                    }
                ]
            },
        ),
        (
            "splits",
            {
                "data": [
                    {
                        "symbol": "MSFT",
                        "date": "2026-01-01",
                        "split_factor": "private-value",
                        "nested": ["private-value"],
                    }
                ]
            },
        ),
        (
            "dividends",
            {
                "data": [
                    {
                        "symbol": "MSFT",
                        "date": "2026-01-01",
                        "dividend": "private-value",
                        "nested": ["private-value"],
                    }
                ]
            },
        ),
        (
            "usage",
            {
                "data": {
                    "requests": {"nested": ["private-value"]},
                    "nested": ["private-value"],
                }
            },
        ),
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

    async def scenario() -> ProviderUnavailableError:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("key", client=client)
            with pytest.raises(ProviderUnavailableError) as caught:
                await invoke(provider)
            return caught.value
        finally:
            await client.aclose()

    translated = run(scenario())
    graph: list[BaseException] = []
    pending: list[BaseException] = [translated]
    while pending:
        current = pending.pop()
        graph.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)

    assert graph == [translated]
    assert "private-value" not in repr(graph)
    for rendered_locals in traceback_locals(translated):
        assert "private-value" not in rendered_locals


def test_deep_aliased_payload_remains_unchanged_without_retention() -> None:
    nested: dict[str, Any] = {}
    cursor = nested
    for _ in range(1200):
        child: dict[str, Any] = {}
        cursor["nested"] = child
        cursor = child
    cursor["value"] = "private-deep-value"
    payload: dict[str, Any] = {
        "data": [],
        "pagination": {"offset": nested},
    }
    original_fingerprint = structure_fingerprint(payload)

    class AliasedJsonResponse(httpx.Response):
        def json(self, **_: Any) -> Any:
            return payload

    def handler(request: httpx.Request) -> httpx.Response:
        return AliasedJsonResponse(200, request=request)

    async def scenario() -> ProviderUnavailableError:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            provider = MarketstackProvider("key", client=client)
            with pytest.raises(ProviderUnavailableError) as caught:
                await provider.list_tickers()
            return caught.value
        finally:
            await client.aclose()

    translated = run(scenario())

    assert structure_fingerprint(payload) == original_fingerprint
    assert translated.__cause__ is None
    assert translated.__context__ is None
    assert "private-deep-value" not in repr(translated)
    for rendered_locals in traceback_locals(translated):
        assert "private-deep-value" not in rendered_locals


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
