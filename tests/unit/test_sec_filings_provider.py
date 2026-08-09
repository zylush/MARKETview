from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import date

import httpx
import pytest

from app.errors import ProviderUnavailableError
from app.providers.sec_filings import SecFairAccessThrottle, SecFilingSource
from app.research.deadline import RequestDeadline
from app.research.domain import (
    FilingDiscoveryCursor,
    FilingDiscoveryRequest,
    FilingReference,
)

USER_AGENT = "MarketView/1.0 operations@example.com"


def _provider_traceback_locals(error: BaseException) -> tuple[str, ...]:
    retained: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("sec_filings.py"):
            retained.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return tuple(retained)


def _deadline() -> RequestDeadline:
    return RequestDeadline.after(30.0)


class RecordingThrottle:
    def __init__(self) -> None:
        self.calls = 0

    async def wait(self) -> None:
        self.calls += 1


def _request(*, limit: int = 2, cursor: str | None = None) -> FilingDiscoveryRequest:
    return FilingDiscoveryRequest(
        symbol="aapl",
        cik="0000320193",
        filing_types=("10-K", "10-Q", "8-K"),
        date_from=date(2024, 1, 1),
        date_to=date(2025, 12, 31),
        limit=limit,
        cursor=FilingDiscoveryCursor(cursor) if cursor else None,
    )


def _recent_payload() -> dict[str, object]:
    return {
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-25-000003",
                    "0000320193-25-000002",
                    "0000320193-25-000001",
                    "0000320193-24-000010",
                ],
                "filingDate": ["2025-10-31", "2025-08-01", "2025-07-01", "2024-10-31"],
                "form": ["10-K", "8-K", "8-K", "10-Q"],
                "primaryDocument": ["aapl-20250927.htm", "aapl-8k.htm", "skip.htm", "q4.htm"],
                "primaryDocDescription": [
                    "Annual report",
                    "Results announcement",
                    "Unrelated current report",
                    "Quarterly report",
                ],
                "items": ["", "2.02,9.01", "3.01", ""],
            },
            "files": [
                {
                    "name": "CIK0000320193-submissions-001.json",
                    "filingFrom": "2024-01-01",
                    "filingTo": "2024-12-31",
                },
                {
                    "name": "https://evil.example/steal.json",
                    "filingFrom": "2024-01-01",
                    "filingTo": "2024-12-31",
                },
                {
                    "name": "CIK0000789019-submissions-002.json",
                    "filingFrom": "2024-01-01",
                    "filingTo": "2024-12-31",
                },
            ],
        }
    }


def _older_payload() -> dict[str, object]:
    return {
        "accessionNumber": ["0000320193-24-000010", "0000320193-24-000009"],
        "filingDate": ["2024-10-31", "2024-05-03"],
        "form": ["10-Q", "10-Q"],
        "primaryDocument": ["q4.htm", "q2.htm"],
        "primaryDocDescription": ["Duplicate", "Quarterly report"],
        "items": ["", ""],
    }


@pytest.mark.asyncio
async def test_discovery_is_bounded_sorted_deduplicated_and_cursor_paginated() -> None:
    requested: list[str] = []
    throttle = RecordingThrottle()

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        assert request.headers["User-Agent"] == USER_AGENT
        if request.url.path == "/submissions/CIK0000320193.json":
            return httpx.Response(200, json=_recent_payload())
        if request.url.path == "/submissions/CIK0000320193-submissions-001.json":
            return httpx.Response(200, json=_older_payload())
        raise AssertionError(f"unexpected request: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = SecFilingSource(user_agent=USER_AGENT, client=client, throttle=throttle)
    first = await source.discover(_request(), deadline=_deadline())
    assert first.next_cursor is not None
    second = await source.discover(
        _request(limit=2, cursor=first.next_cursor.value),
        deadline=_deadline(),
    )
    await client.aclose()

    assert [item.accession_number for item in first.references] == [
        "0000320193-25-000003",
        "0000320193-25-000002",
    ]
    assert first.references[1].filing_type == "8-K"
    assert first.next_cursor == FilingDiscoveryCursor("v1:2")
    assert [item.accession_number for item in second.references] == [
        "0000320193-24-000010",
        "0000320193-24-000009",
    ]
    assert second.next_cursor is None
    assert all(url.startswith("https://data.sec.gov/submissions/") for url in requested)
    assert not any("evil.example" in url for url in requested)
    assert throttle.calls == len(requested)


@pytest.mark.asyncio
async def test_discovery_applies_8k_item_relevance_to_amendments() -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-25-000002",
                    "0000320193-25-000001",
                ],
                "filingDate": ["2025-08-02", "2025-08-01"],
                "form": ["8-K/A", "8-K/A"],
                "primaryDocument": ["irrelevant-amendment.htm", "relevant-amendment.htm"],
                "primaryDocDescription": [
                    "Irrelevant current report amendment",
                    "Results announcement amendment",
                ],
                "items": ["3.01,9.01", "2.02,9.01"],
            },
            "files": [],
        }
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    )
    source = SecFilingSource(user_agent=USER_AGENT, client=client)
    request = FilingDiscoveryRequest(
        symbol="aapl",
        cik="0000320193",
        filing_types=("8-K/A",),
        date_from=date(2025, 1, 1),
        date_to=date(2025, 12, 31),
        limit=2,
    )

    page = await source.discover(request, deadline=_deadline())
    await client.aclose()

    assert [item.accession_number for item in page.references] == ["0000320193-25-000001"]
    assert page.references[0].filing_type == "8-K/A"


@pytest.mark.asyncio
async def test_discovery_rejects_redirects_malformed_payloads_and_unbounded_bodies() -> None:
    responses = iter(
        (
            httpx.Response(302, headers={"Location": "https://evil.example/"}),
            httpx.Response(200, content=b"{" + b"x" * 5000),
            httpx.Response(200, content=json.dumps({"filings": {}}).encode()),
        )
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return next(responses)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = SecFilingSource(
        user_agent=USER_AGENT,
        client=client,
        max_response_bytes=1024,
        throttle=RecordingThrottle(),
    )
    for _ in range(3):
        with pytest.raises(ProviderUnavailableError, match="SEC filing discovery is unavailable"):
            await source.discover(_request(), deadline=_deadline())
    await client.aclose()


@pytest.mark.asyncio
async def test_discovery_sanitizes_transport_timeouts_without_retrying() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("secret raw upstream details", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = SecFilingSource(user_agent=USER_AGENT, client=client, throttle=RecordingThrottle())
    with pytest.raises(
        ProviderUnavailableError,
        match="SEC filing discovery is unavailable",
    ) as error:
        await source.discover(_request(), deadline=_deadline())
    await client.aclose()
    assert calls == 1
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert "secret raw upstream details" not in str(error.value)


@pytest.mark.asyncio
async def test_absolute_deadline_bounds_throttle_wait_before_request() -> None:
    requests = 0

    class BlockingThrottle:
        async def wait(self) -> None:
            await asyncio.Event().wait()

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=_recent_payload())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = SecFilingSource(user_agent=USER_AGENT, client=client, throttle=BlockingThrottle())

    with pytest.raises(TimeoutError, match="deadline expired") as caught:
        await source.discover(_request(), deadline=RequestDeadline.after(0.01))
    await client.aclose()

    assert requests == 0
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_slow_stream_deadline_detaches_partial_payload_without_retrying() -> None:
    marker = "private-slow-stream-payload-sentinel"
    requests = 0

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield f'{{"private":"{marker}",'.encode()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            return None

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            stream=SlowStream(),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = SecFilingSource(
        user_agent=USER_AGENT,
        client=client,
        throttle=RecordingThrottle(),
    )

    with pytest.raises(TimeoutError, match="deadline expired") as caught:
        await source.discover(_request(), deadline=RequestDeadline.after(0.01))
    await client.aclose()

    assert requests == 1
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert marker not in "\n".join(_provider_traceback_locals(caught.value))


@pytest.mark.asyncio
async def test_malformed_discovery_detaches_raw_submissions_from_exception_graph() -> None:
    marker = "private-submissions-payload-sentinel"
    payload = {
        "filings": {
            "recent": {"private": marker},
            "files": [],
        }
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    )
    source = SecFilingSource(user_agent=USER_AGENT, client=client)

    with pytest.raises(ProviderUnavailableError) as caught:
        await source.discover(_request(), deadline=_deadline())
    await client.aclose()

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert marker not in "\n".join(_provider_traceback_locals(caught.value))


@pytest.mark.asyncio
async def test_fetch_revalidates_fixed_archive_url_content_type_and_size() -> None:
    reference = FilingReference(
        symbol="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000003",
        filing_type="10-K",
        title="Annual report",
        filed_date=date(2025, 10, 31),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000003/aapl-20250927.htm"
        ),
    )
    requested: list[str] = []

    def ok_handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=b"<html><body>Annual report</body></html>",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(ok_handler))
    source = SecFilingSource(user_agent=USER_AGENT, client=client, throttle=RecordingThrottle())
    filing = await source.fetch(reference, deadline=_deadline())
    await client.aclose()
    assert filing.reference is reference
    assert filing.media_type == "text/html"
    assert requested == [reference.source_url]

    for response in (
        httpx.Response(302, headers={"Location": "https://evil.example/x"}),
        httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=b"pdf"),
        httpx.Response(
            200,
            headers={"Content-Type": "text/html", "Content-Length": "99999999"},
            content=b"x",
        ),
        httpx.Response(
            200,
            headers={"Content-Type": "text/html", "Content-Length": "not-a-number"},
            content=b"x",
        ),
    ):
        bad_client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _, current=response: current),
        )
        bad_source = SecFilingSource(
            user_agent=USER_AGENT,
            client=bad_client,
            max_filing_bytes=128,
            throttle=RecordingThrottle(),
        )
        with pytest.raises(ProviderUnavailableError, match="SEC filing document is unavailable"):
            await bad_source.fetch(reference, deadline=_deadline())
        await bad_client.aclose()

    mismatched = object.__new__(FilingReference)
    for name, value in (
        ("symbol", "AAPL"),
        ("cik", "0000320193"),
        ("accession_number", "0000320193-25-000003"),
        ("filing_type", "10-K"),
        ("title", "Annual report"),
        ("filed_date", date(2025, 10, 31)),
        (
            "source_url",
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325999999/aapl-20250927.htm",
        ),
    ):
        object.__setattr__(mismatched, name, value)
    never_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("unexpected request"))
    )
    strict_source = SecFilingSource(user_agent=USER_AGENT, client=never_client)
    with pytest.raises(ValueError, match="canonical filing identity"):
        await strict_source.fetch(mismatched, deadline=_deadline())
    await never_client.aclose()


@pytest.mark.parametrize("rate", [0, 5.01, 6, True])
def test_fair_access_throttle_never_allows_more_than_five_requests_per_second(
    rate: object,
) -> None:
    with pytest.raises(ValueError, match="between zero and five"):
        SecFairAccessThrottle(max_requests_per_second=rate)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fair_access_throttle_serializes_requests_at_the_configured_interval() -> None:
    now = 10.0
    delays: tuple[float, ...] = ()

    async def sleep(delay: float) -> None:
        nonlocal now, delays
        delays = (*delays, delay)
        now += delay

    throttle = SecFairAccessThrottle(
        max_requests_per_second=5,
        clock=lambda: now,
        sleeper=sleep,
    )
    await throttle.wait()
    await throttle.wait()
    assert delays == pytest.approx((0.2,))


def test_source_rejects_nondescriptive_user_agents_and_unsafe_limits() -> None:
    with pytest.raises(ValueError, match="contact email"):
        SecFilingSource(user_agent="MarketView")
    with pytest.raises(ValueError, match="response byte limit"):
        SecFilingSource(user_agent=USER_AGENT, max_response_bytes=True)


@pytest.mark.asyncio
async def test_expired_deadline_prevents_any_sec_request() -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=_recent_payload())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = SecFilingSource(user_agent=USER_AGENT, client=client)
    deadline = RequestDeadline(expires_at=0.0, _clock=lambda: 1.0)
    with pytest.raises(TimeoutError, match="deadline expired"):
        await source.discover(_request(), deadline=deadline)
    await client.aclose()
    assert requests == 0


@pytest.mark.asyncio
async def test_source_closes_only_the_client_it_constructs() -> None:
    source = SecFilingSource(user_agent=USER_AGENT)
    assert source._client._trust_env is False
    await source.aclose()
    assert source._client.is_closed is True


@pytest.mark.asyncio
async def test_source_rejects_invalid_contract_objects_before_network() -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=_recent_payload())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = SecFilingSource(user_agent=USER_AGENT, client=client)
    with pytest.raises(ValueError, match="request is invalid"):
        await source.discover(object(), deadline=_deadline())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="deadline is invalid"):
        await source.discover(_request(), deadline=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cursor is invalid"):
        await source.discover(
            _request(cursor="opaque"),
            deadline=_deadline(),
        )
    with pytest.raises(ValueError, match="trusted SEC Archives"):
        await source.fetch(object(), deadline=_deadline())  # type: ignore[arg-type]
    await client.aclose()
    assert requests == 0


@pytest.mark.asyncio
async def test_discovery_skips_malformed_rows_but_requires_a_valid_table_shape() -> None:
    malformed_rows = {
        "filings": {
            "recent": {
                "accessionNumber": ["bad", "0000789019-25-000001"],
                "filingDate": ["not-a-date", "2025-01-01"],
                "form": ["10-K", "10-K"],
                "primaryDocument": ["../bad.htm", "valid.htm"],
                "primaryDocDescription": ["Bad", "Wrong CIK"],
                "items": ["", ""],
            },
            "files": [],
        }
    }

    responses = iter(
        (
            httpx.Response(200, json=malformed_rows),
            httpx.Response(200, json={"filings": {"recent": {"form": []}, "files": []}}),
        )
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: next(responses)),
    )
    source = SecFilingSource(user_agent=USER_AGENT, client=client)
    empty = await source.discover(_request(), deadline=_deadline())
    assert empty.references == ()
    with pytest.raises(ProviderUnavailableError, match="discovery is unavailable"):
        await source.discover(_request(), deadline=_deadline())
    await client.aclose()
