from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import groupby
from typing import Literal, Never

import httpx
from pydantic import SecretStr, ValidationError

from app.errors import ProviderUnavailableError
from app.models import SymbolRecord
from app.services.symbols import SymbolDirectory

SEC_SYMBOL_DIRECTORY_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
_SOURCE_NAME = "sec-company-tickers-exchange"
_REQUIRED_FIELDS = frozenset({"name", "ticker", "exchange"})
_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class _LoadOutcome:
    directory: SymbolDirectory | None = None
    error: Literal["timeout", "unavailable", "invalid"] | None = None


def _raise_load_error(error: str | None) -> Never:
    if error == "timeout":
        raise ProviderUnavailableError("SEC symbol directory request timed out") from None
    if error == "invalid":
        raise ProviderUnavailableError("SEC symbol directory response is invalid") from None
    raise ProviderUnavailableError("SEC symbol directory is unavailable") from None


class SecSymbolDirectoryProvider:
    """Single-attempt adapter for the SEC company ticker directory."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        normalized_user_agent = self._validate_user_agent(user_agent)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 10
        ):
            raise ValueError("SEC timeout must be between 0 and 10 seconds")
        self._user_agent = SecretStr(normalized_user_agent)
        self._timeout = float(timeout_seconds)
        self._client = client or httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None
        self._clock = clock

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def load(self) -> SymbolDirectory:
        outcome = await self._load_raw()
        del self
        if outcome.directory is None:
            _raise_load_error(outcome.error)
        return outcome.directory

    async def _load_raw(self) -> _LoadOutcome:
        try:
            async with self._client.stream(
                "GET",
                SEC_SYMBOL_DIRECTORY_URL,
                headers={
                    "User-Agent": self._user_agent.get_secret_value(),
                    "Accept": "application/json",
                },
                timeout=self._timeout,
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    return _LoadOutcome(error="unavailable")
                chunks: tuple[bytes, ...] = ()
                received_bytes = 0
                async for chunk in response.aiter_bytes():
                    received_bytes += len(chunk)
                    if received_bytes > _MAX_RESPONSE_BYTES:
                        return _LoadOutcome(error="invalid")
                    chunks = (*chunks, chunk)
        except httpx.TimeoutException:
            return _LoadOutcome(error="timeout")
        except Exception:
            return _LoadOutcome(error="unavailable")
        try:
            payload = json.loads(b"".join(chunks))
            records = self._parse_payload(payload)
            as_of = self._clock()
            if as_of.tzinfo is None or as_of.utcoffset() is None:
                raise ValueError
        except Exception:
            return _LoadOutcome(error="invalid")
        return _LoadOutcome(
            directory=SymbolDirectory(
                records=records,
                source=_SOURCE_NAME,
                as_of=as_of.astimezone(UTC),
            )
        )

    @classmethod
    def _parse_payload(cls, payload: object) -> tuple[SymbolRecord, ...]:
        if not isinstance(payload, Mapping):
            raise ValueError
        fields = payload.get("fields")
        data = payload.get("data")
        if (
            not isinstance(fields, list)
            or not fields
            or not all(isinstance(field, str) for field in fields)
            or len(set(fields)) != len(fields)
            or not _REQUIRED_FIELDS.issubset(fields)
            or not isinstance(data, list)
            or not data
            or len(data) > _MAX_ROWS
        ):
            raise ValueError
        positions = {field: index for index, field in enumerate(fields)}
        candidates = tuple(
            record for row in data if (record := cls._parse_row(row, positions)) is not None
        )
        if not candidates:
            raise ValueError
        ordered = sorted(candidates, key=cls._record_order)
        return tuple(next(group) for _, group in groupby(ordered, key=lambda record: record.symbol))

    @staticmethod
    def _parse_row(
        row: object,
        positions: Mapping[str, int],
    ) -> SymbolRecord | None:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            return None
        try:
            name = row[positions["name"]]
            ticker = row[positions["ticker"]]
            exchange = row[positions["exchange"]]
        except (IndexError, TypeError):
            return None
        if not all(isinstance(value, str) and value.strip() for value in (name, ticker, exchange)):
            return None
        try:
            return SymbolRecord(symbol=ticker, name=name, exchange=exchange)
        except (ValidationError, ValueError, TypeError):
            return None

    @staticmethod
    def _record_order(record: SymbolRecord) -> tuple[str, str, str, str, str]:
        return (
            record.symbol,
            record.name.casefold(),
            record.exchange.casefold(),
            record.name,
            record.exchange,
        )

    @staticmethod
    def _validate_user_agent(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("SEC User-Agent must be descriptive and include contact email")
        normalized = value.strip()
        if (
            not 10 <= len(normalized) <= 200
            or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
            or _EMAIL_PATTERN.search(normalized) is None
        ):
            raise ValueError("SEC User-Agent must be descriptive and include contact email")
        return normalized
