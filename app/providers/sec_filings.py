from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, cast

import httpx

from app.errors import ProviderUnavailableError
from app.research.deadline import RequestDeadline
from app.research.domain import (
    FilingDiscoveryCursor,
    FilingDiscoveryPage,
    FilingDiscoveryRequest,
    FilingReference,
    RawFiling,
    is_trusted_sec_archive_url,
)

_SUBMISSIONS_ROOT = "https://data.sec.gov/submissions"
_ARCHIVES_ROOT = "https://www.sec.gov/Archives/edgar/data"
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_REFERENCED_FILE = re.compile(r"^CIK\d{10}-submissions-\d{3}\.json$")
_PRIMARY_DOCUMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}\.(?:htm|html)$", re.I)
_CURSOR = re.compile(r"^v1:(0|[1-9]\d{0,5})$")
_SUPPORTED_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_RELEVANT_8K_ITEMS = frozenset({"1.01", "2.01", "2.02", "4.02", "5.02", "7.01", "8.01"})
_MAX_REFERENCED_SUBMISSIONS = 8


@dataclass(frozen=True, slots=True)
class _DiscoveryOutcome:
    page: FilingDiscoveryPage | None = None


def _require_discovery_page(outcome: _DiscoveryOutcome) -> FilingDiscoveryPage:
    if outcome.page is None:
        raise ProviderUnavailableError("SEC filing discovery is unavailable") from None
    return outcome.page


class RequestThrottle(Protocol):
    async def wait(self) -> None: ...


class SecFairAccessThrottle:
    """Serialize SEC requests at a configurable rate no greater than five per second."""

    def __init__(
        self,
        *,
        max_requests_per_second: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if (
            isinstance(max_requests_per_second, bool)
            or not isinstance(max_requests_per_second, (int, float))
            or not math.isfinite(max_requests_per_second)
            or not 0 < max_requests_per_second <= 5
        ):
            raise ValueError("SEC request rate must be between zero and five per second")
        self._interval = 1.0 / float(max_requests_per_second)
        self._clock = clock
        self._sleep = sleeper
        self._lock = asyncio.Lock()
        self._next_request_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = self._clock()
            delay = self._next_request_at - now
            if delay > 0:
                await self._sleep(delay)
                now = self._clock()
            self._next_request_at = max(now, self._next_request_at) + self._interval


class SecFilingSource:
    """Bounded SEC submissions discovery and fixed-origin filing download adapter."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float = 8.0,
        max_response_bytes: int = 8 * 1024 * 1024,
        max_filing_bytes: int = 16 * 1024 * 1024,
        client: httpx.AsyncClient | None = None,
        throttle: RequestThrottle | None = None,
    ) -> None:
        self._user_agent = self._validated_user_agent(user_agent)
        self._timeout = self._validated_timeout(timeout_seconds)
        self._max_response_bytes = self._validated_byte_limit(
            max_response_bytes, "response byte limit"
        )
        self._max_filing_bytes = self._validated_byte_limit(max_filing_bytes, "filing byte limit")
        self._client = client or httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None
        self._throttle = throttle or SecFairAccessThrottle()
        self._request_slots = asyncio.Semaphore(2)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def discover(
        self,
        request: FilingDiscoveryRequest,
        *,
        deadline: RequestDeadline,
    ) -> FilingDiscoveryPage:
        if not isinstance(request, FilingDiscoveryRequest):
            raise ValueError("filing discovery request is invalid")
        if not isinstance(deadline, RequestDeadline):
            raise ValueError("filing discovery deadline is invalid")
        deadline.raise_if_expired()
        offset = self._cursor_offset(request.cursor)
        outcome = await self._discover_outcome(request, deadline=deadline, offset=offset)
        return _require_discovery_page(outcome)

    async def _discover_outcome(
        self,
        request: FilingDiscoveryRequest,
        *,
        deadline: RequestDeadline,
        offset: int,
    ) -> _DiscoveryOutcome:
        recent_url = f"{_SUBMISSIONS_ROOT}/CIK{request.cik}.json"
        recent = await self._json(recent_url, self._max_response_bytes, deadline)
        if recent is None:
            return _DiscoveryOutcome()
        try:
            filings = recent["filings"]
            if not isinstance(filings, Mapping):
                raise ValueError
            candidates = self._references_from_table(filings["recent"], request)
            referenced = self._referenced_files(filings.get("files"), request)
        except (KeyError, TypeError, ValueError):
            return _DiscoveryOutcome()

        for name in referenced:
            payload = await self._json(
                f"{_SUBMISSIONS_ROOT}/{name}",
                self._max_response_bytes,
                deadline,
            )
            if payload is None:
                return _DiscoveryOutcome()
            try:
                candidates = (*candidates, *self._references_from_table(payload, request))
            except (TypeError, ValueError):
                return _DiscoveryOutcome()

        unique = {
            (reference.cik, reference.accession_number): reference for reference in candidates
        }
        ordered = tuple(
            sorted(
                unique.values(),
                key=lambda reference: (reference.filed_date, reference.accession_number),
                reverse=True,
            )
        )
        page = ordered[offset : offset + request.limit]
        next_offset = offset + len(page)
        next_cursor = (
            FilingDiscoveryCursor(f"v1:{next_offset}") if next_offset < len(ordered) else None
        )
        return _DiscoveryOutcome(page=FilingDiscoveryPage(references=page, next_cursor=next_cursor))

    async def fetch(
        self,
        reference: FilingReference,
        *,
        deadline: RequestDeadline,
    ) -> RawFiling:
        if not isinstance(reference, FilingReference) or not is_trusted_sec_archive_url(
            reference.source_url
        ):
            raise ValueError("filing reference is not a trusted SEC Archives URL")
        if not reference.source_url.startswith(f"{_ARCHIVES_ROOT}/"):
            raise ValueError("filing reference is not a trusted SEC Archives URL")
        expected_directory = (
            f"{_ARCHIVES_ROOT}/{int(reference.cik)}/{reference.accession_number.replace('-', '')}/"
        )
        primary_document = reference.source_url.removeprefix(expected_directory)
        if reference.source_url == primary_document or not _PRIMARY_DOCUMENT.fullmatch(
            primary_document
        ):
            raise ValueError("filing URL does not match its canonical filing identity")
        if not isinstance(deadline, RequestDeadline):
            raise ValueError("filing download deadline is invalid")
        deadline.raise_if_expired()
        outcome = await self._bytes(
            reference.source_url,
            self._max_filing_bytes,
            _SUPPORTED_MEDIA_TYPES,
            deadline,
        )
        if outcome is None:
            raise ProviderUnavailableError("SEC filing document is unavailable") from None
        body, media_type = outcome
        if media_type not in _SUPPORTED_MEDIA_TYPES:
            raise ProviderUnavailableError("SEC filing document is unavailable") from None
        return RawFiling(reference=reference, media_type=media_type, body=body)

    async def _json(
        self,
        url: str,
        maximum: int,
        deadline: RequestDeadline,
    ) -> Mapping[str, Any] | None:
        outcome = await self._bytes(
            url,
            maximum,
            frozenset({"application/json"}),
            deadline,
        )
        if outcome is None:
            return None
        body, media_type = outcome
        if media_type != "application/json":
            return None
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, Mapping) else None

    async def _bytes(
        self,
        url: str,
        maximum: int,
        accepted_media_types: frozenset[str],
        deadline: RequestDeadline,
    ) -> tuple[bytes, str] | None:
        deadline.raise_if_expired()
        outcome: tuple[bytes, str] | None = None
        timed_out = False
        try:
            async with asyncio.timeout(deadline.remaining_seconds()):
                async with self._request_slots:
                    outcome = await self._bytes_once(
                        url,
                        maximum,
                        accepted_media_types,
                        deadline,
                    )
        except TimeoutError:
            timed_out = True
        if timed_out:
            raise TimeoutError("research request deadline expired") from None
        return outcome

    async def _bytes_once(
        self,
        url: str,
        maximum: int,
        accepted_media_types: frozenset[str],
        deadline: RequestDeadline,
    ) -> tuple[bytes, str] | None:
        deadline.raise_if_expired()
        await self._throttle.wait()
        deadline.raise_if_expired()
        request_timeout = min(self._timeout, deadline.remaining_seconds())
        if request_timeout <= 0:
            raise TimeoutError("research request deadline expired")
        try:
            async with self._client.stream(
                "GET",
                url,
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "application/json, text/html, application/xhtml+xml",
                },
                timeout=request_timeout,
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    return None
                media_type = (
                    response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                )
                if media_type not in accepted_media_types:
                    return None
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        if int(declared) < 0 or int(declared) > maximum:
                            return None
                    except ValueError:
                        return None
                chunks: list[bytes] = []
                received = 0
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    if received > maximum:
                        return None
                    chunks.append(chunk)
        except (httpx.HTTPError, RuntimeError):
            return None
        return b"".join(chunks), media_type

    @classmethod
    def _references_from_table(
        cls,
        table: object,
        request: FilingDiscoveryRequest,
    ) -> tuple[FilingReference, ...]:
        if not isinstance(table, Mapping):
            raise ValueError
        required_names = (
            "accessionNumber",
            "filingDate",
            "form",
            "primaryDocument",
            "primaryDocDescription",
        )
        columns = tuple(cls._column(table, name) for name in required_names)
        lengths = {len(column) for column in columns}
        if len(lengths) != 1:
            raise ValueError
        row_count = lengths.pop()
        items = table.get("items")
        if items is None:
            items = [""] * row_count
        if not cls._is_column(items) or len(items) != row_count:
            raise ValueError
        rows = zip(*columns, items, strict=True)
        references: list[FilingReference] = []
        for accession, filed, form, document, description, item_codes in rows:
            reference = cls._reference_from_row(
                request,
                accession,
                filed,
                form,
                document,
                description,
                item_codes,
            )
            if reference is not None:
                references.append(reference)
        return tuple(references)

    @classmethod
    def _reference_from_row(
        cls,
        request: FilingDiscoveryRequest,
        accession: object,
        filed: object,
        form: object,
        document: object,
        description: object,
        item_codes: object,
    ) -> FilingReference | None:
        if (
            not isinstance(accession, str)
            or not isinstance(filed, str)
            or not isinstance(form, str)
            or not isinstance(document, str)
        ):
            return None
        normalized_form = form.upper()
        if normalized_form not in request.filing_types:
            return None
        try:
            filed_date = date.fromisoformat(filed)
        except ValueError:
            return None
        if not request.date_from <= filed_date <= request.date_to:
            return None
        if normalized_form in ("8-K", "8-K/A") and not cls._relevant_8k(item_codes):
            return None
        if not _PRIMARY_DOCUMENT.fullmatch(document):
            return None
        compact_accession = accession.replace("-", "")
        if not re.fullmatch(r"\d{18}", compact_accession):
            return None
        if accession[:10] != request.cik:
            return None
        cik_path = str(int(request.cik))
        source_url = f"{_ARCHIVES_ROOT}/{cik_path}/{compact_accession}/{document}"
        title = description.strip() if isinstance(description, str) else ""
        if (
            not title
            or len(title) > 200
            or any(character in title for character in "<>")
            or any(ord(character) < 32 or ord(character) == 127 for character in title)
        ):
            title = f"{normalized_form} filing"
        try:
            return FilingReference(
                symbol=request.symbol,
                cik=request.cik,
                accession_number=accession,
                filing_type=normalized_form,
                title=title,
                filed_date=filed_date,
                source_url=source_url,
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _relevant_8k(value: object) -> bool:
        if not isinstance(value, str):
            return False
        codes = {item.strip() for item in value.split(",") if item.strip()}
        return bool(codes.intersection(_RELEVANT_8K_ITEMS))

    @staticmethod
    def _is_column(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))

    @classmethod
    def _column(cls, table: Mapping[str, Any], name: str) -> Sequence[object]:
        value = table.get(name)
        if not cls._is_column(value):
            raise ValueError
        return cast(Sequence[object], value)

    @staticmethod
    def _referenced_files(
        value: object,
        request: FilingDiscoveryRequest,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError
        names: list[str] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            from_value = item.get("filingFrom")
            to_value = item.get("filingTo")
            if (
                not isinstance(name, str)
                or not isinstance(from_value, str)
                or not isinstance(to_value, str)
            ):
                continue
            if not _REFERENCED_FILE.fullmatch(name):
                continue
            if not name.startswith(f"CIK{request.cik}-"):
                continue
            try:
                file_from = date.fromisoformat(from_value)
                file_to = date.fromisoformat(to_value)
            except ValueError:
                continue
            if file_from > file_to:
                continue
            if file_from <= request.date_to and file_to >= request.date_from:
                names.append(name)
            if len(names) >= _MAX_REFERENCED_SUBMISSIONS:
                break
        return tuple(names)

    @staticmethod
    def _cursor_offset(cursor: FilingDiscoveryCursor | None) -> int:
        if cursor is None:
            return 0
        match = _CURSOR.fullmatch(cursor.value)
        if match is None:
            raise ValueError("filing discovery cursor is invalid")
        return int(cursor.value.split(":", 1)[1])

    @staticmethod
    def _validated_user_agent(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("SEC User-Agent must identify the application and contact email")
        normalized = value.strip()
        if (
            not 10 <= len(normalized) <= 200
            or _EMAIL.search(normalized) is None
            or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
        ):
            raise ValueError("SEC User-Agent must identify the application and contact email")
        return normalized

    @staticmethod
    def _validated_timeout(value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 10
        ):
            raise ValueError("SEC timeout must be between zero and ten seconds")
        return float(value)

    @staticmethod
    def _validated_byte_limit(value: object, name: str) -> int:
        if type(value) is not int or not 1 <= value <= 64 * 1024 * 1024:
            raise ValueError(f"SEC {name} must be between 1 and 67108864")
        return value


__all__ = ["RequestThrottle", "SecFairAccessThrottle", "SecFilingSource"]
