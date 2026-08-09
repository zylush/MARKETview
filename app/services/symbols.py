from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import groupby
from typing import Any, Protocol

from app.cache.base import Cache, CacheEntry, CacheKeyBuilder
from app.errors import SymbolDirectoryUnavailableError
from app.models import SymbolRecord, SymbolSearchPage
from app.validation import validate_limit, validate_symbol_query

_DIRECTORY_CACHE_CONTRACT = "sec-symbol-directory-v3-prefix2"
_GENERATION_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_SOURCE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_COMPANY_TOKEN_PATTERN = re.compile(r"[A-Z0-9]+")
_BUCKET_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]$")
_MAX_DIRECTORY_RECORDS = 100_000
_MAX_BUCKETS = 2_000


@dataclass(frozen=True, slots=True)
class SymbolDirectory:
    records: tuple[SymbolRecord, ...]
    source: str
    as_of: datetime


@dataclass(frozen=True, slots=True)
class _DirectoryManifest:
    generation: str
    prefixes: tuple[str, ...]
    source: str
    as_of: datetime


class SymbolDirectorySource(Protocol):
    async def load(self) -> SymbolDirectory: ...


class StaticSymbolDirectorySource:
    """Deterministic source intended for tests, not production bootstrapping."""

    def __init__(
        self,
        *,
        records: Sequence[SymbolRecord],
        as_of: datetime | None = None,
        source: str = "static",
    ) -> None:
        self._directory = SymbolDirectory(
            records=tuple(records),
            source=source,
            as_of=as_of or datetime.now(UTC),
        )

    async def load(self) -> SymbolDirectory:
        return self._directory


class SymbolSearchService:
    """Search cached two-character buckets; never refresh on the request path."""

    def __init__(
        self,
        source: SymbolDirectorySource | None,
        cache: Cache,
        *,
        schema_version: str = "v2",
        directory_ttl_seconds: int = 24 * 3600,
        stale_seconds: int = 7 * 24 * 3600,
    ) -> None:
        if directory_ttl_seconds <= 0 or stale_seconds < 0:
            raise ValueError("symbol directory cache TTL values are invalid")
        self._source = source
        self._cache = cache
        self._keys = CacheKeyBuilder(schema_version=schema_version)
        self._ttl = directory_ttl_seconds
        self._stale = stale_seconds

    @property
    def manifest_cache_key(self) -> str:
        return self._keys.build(
            "symbol_directory_manifest",
            {"provider_contract": _DIRECTORY_CACHE_CONTRACT},
        )

    async def refresh_directory(self) -> SymbolDirectory:
        """Write every immutable generation bucket before switching the manifest."""
        try:
            if self._source is None:
                raise ValueError
            loaded = await self._source.load()
            directory = self._validated_directory(loaded)
            generation = self._generation(directory)
            buckets = self._build_buckets(directory.records)
            for prefix, records in buckets.items():
                await self._cache.set(
                    self._generation_bucket_cache_key(generation, prefix),
                    self._encode_bucket(records),
                    ttl_seconds=self._ttl,
                    stale_seconds=self._stale,
                )
            await self._cache.set(
                self.manifest_cache_key,
                {
                    "generation": generation,
                    "prefixes": tuple(buckets),
                    "source": directory.source,
                    "as_of": directory.as_of.isoformat(),
                },
                ttl_seconds=self._ttl,
                stale_seconds=self._stale,
            )
        except Exception:
            raise SymbolDirectoryUnavailableError("symbol directory refresh failed") from None
        return directory

    async def search_symbols(self, query: str, *, limit: int = 8) -> SymbolSearchPage:
        checked_query = validate_symbol_query(query)
        checked_limit = validate_limit(limit, maximum=8)
        manifest_entry = await self._read_manifest()
        manifest = self._decode_manifest(manifest_entry)
        records, buckets_stale = await self._read_relevant_buckets(manifest, checked_query)
        items = self._rank(records, checked_query)[:checked_limit]
        return SymbolSearchPage(
            items=items,
            total=len(items),
            next_cursor=None,
            limit=checked_limit,
            source=manifest.source,
            as_of=manifest.as_of,
            stale=manifest_entry.is_stale or buckets_stale,
        )

    async def _read_manifest(self) -> CacheEntry[Any]:
        try:
            entry = await self._cache.get(self.manifest_cache_key)
        except Exception:
            raise SymbolDirectoryUnavailableError("symbol directory is unavailable") from None
        if entry is None:
            raise SymbolDirectoryUnavailableError("symbol directory is unavailable")
        return entry

    async def _read_relevant_buckets(
        self,
        manifest: _DirectoryManifest,
        query: str,
    ) -> tuple[tuple[SymbolRecord, ...], bool]:
        available = frozenset(manifest.prefixes)
        relevant = tuple(prefix for prefix in self._query_prefixes(query) if prefix in available)
        if not relevant:
            return (), False
        collected: dict[str, SymbolRecord] = {}
        stale = False
        for prefix in relevant:
            try:
                entry = await self._cache.get(
                    self._generation_bucket_cache_key(manifest.generation, prefix)
                )
            except Exception:
                raise SymbolDirectoryUnavailableError("symbol directory is unavailable") from None
            if entry is None:
                raise SymbolDirectoryUnavailableError("symbol directory is unavailable")
            bucket = self._decode_bucket(entry.value, prefix)
            collected = {**collected, **{record.symbol: record for record in bucket}}
            stale = stale or entry.is_stale
        return tuple(
            sorted(
                collected.values(),
                key=lambda record: (
                    record.symbol,
                    record.name.casefold(),
                    record.exchange.casefold(),
                ),
            )
        ), stale

    def _generation_bucket_cache_key(self, generation: str, prefix: str) -> str:
        return self._keys.build(
            "symbol_directory_generation_bucket",
            {
                "provider_contract": _DIRECTORY_CACHE_CONTRACT,
                "generation": generation,
                "prefix": prefix,
            },
        )

    @classmethod
    def _decode_manifest(cls, entry: CacheEntry[Any]) -> _DirectoryManifest:
        value = entry.value
        if not isinstance(value, dict):
            raise SymbolDirectoryUnavailableError("symbol directory manifest is invalid")
        try:
            generation = value["generation"]
            prefixes = value["prefixes"]
            source = value["source"]
            raw_as_of = value["as_of"]
            if (
                not isinstance(generation, str)
                or not _GENERATION_PATTERN.fullmatch(generation)
                or not isinstance(prefixes, (list, tuple))
                or not prefixes
                or len(prefixes) > _MAX_BUCKETS
                or not all(isinstance(prefix, str) for prefix in prefixes)
                or tuple(sorted(set(prefixes))) != tuple(prefixes)
                or not all(_BUCKET_PATTERN.fullmatch(prefix) for prefix in prefixes)
                or not isinstance(source, str)
                or not _SOURCE_PATTERN.fullmatch(source)
                or not isinstance(raw_as_of, str)
            ):
                raise ValueError
            as_of = datetime.fromisoformat(raw_as_of)
            if as_of.tzinfo is None or as_of.utcoffset() is None:
                raise ValueError
            return _DirectoryManifest(
                generation=generation,
                prefixes=tuple(prefixes),
                source=source,
                as_of=as_of.astimezone(UTC),
            )
        except Exception:
            raise SymbolDirectoryUnavailableError("symbol directory manifest is invalid") from None

    @staticmethod
    def _encode_bucket(records: tuple[SymbolRecord, ...]) -> dict[str, object]:
        return {"records": tuple(record.model_dump(mode="json") for record in records)}

    @classmethod
    def _decode_bucket(cls, value: object, prefix: str) -> tuple[SymbolRecord, ...]:
        if not isinstance(value, dict):
            raise SymbolDirectoryUnavailableError("symbol directory cache entry is invalid")
        try:
            raw_records = value["records"]
            if (
                not isinstance(raw_records, (list, tuple))
                or not raw_records
                or len(raw_records) > _MAX_DIRECTORY_RECORDS
            ):
                raise ValueError
            records = tuple(SymbolRecord.model_validate(item) for item in raw_records)
            normalized = tuple(
                sorted(
                    records,
                    key=lambda item: (
                        item.symbol,
                        item.name.casefold(),
                        item.exchange.casefold(),
                    ),
                )
            )
            if len({record.symbol for record in normalized}) != len(normalized) or any(
                prefix not in cls._record_prefixes(record) for record in normalized
            ):
                raise ValueError
            return normalized
        except Exception:
            raise SymbolDirectoryUnavailableError(
                "symbol directory cache entry is invalid"
            ) from None

    @classmethod
    def _build_buckets(
        cls,
        records: tuple[SymbolRecord, ...],
    ) -> dict[str, tuple[SymbolRecord, ...]]:
        indexed = tuple(
            (prefix, record) for record in records for prefix in cls._record_prefixes(record)
        )
        ordered = sorted(
            indexed,
            key=lambda item: (
                item[0],
                item[1].symbol,
                item[1].name.casefold(),
                item[1].exchange.casefold(),
            ),
        )
        buckets = {
            prefix: tuple(item[1] for item in group)
            for prefix, group in groupby(ordered, key=lambda item: item[0])
        }
        if not buckets or len(buckets) > _MAX_BUCKETS:
            raise ValueError("invalid symbol directory")
        return buckets

    @staticmethod
    def _record_prefixes(record: SymbolRecord) -> tuple[str, ...]:
        prefixes = frozenset(
            (
                *((record.symbol[:2],) if len(record.symbol) >= 2 else ()),
                *(
                    token[:2]
                    for token in _COMPANY_TOKEN_PATTERN.findall(record.name.upper())
                    if len(token) >= 2
                ),
            )
        )
        return tuple(sorted(prefix for prefix in prefixes if _BUCKET_PATTERN.fullmatch(prefix)))

    @staticmethod
    def _query_prefixes(query: str) -> tuple[str, ...]:
        company_tokens = _COMPANY_TOKEN_PATTERN.findall(query)
        candidates = (
            *((query[:2],) if len(query) >= 2 else ()),
            *(token[:2] for token in company_tokens if len(token) >= 2),
        )
        return tuple(sorted({prefix for prefix in candidates if _BUCKET_PATTERN.fullmatch(prefix)}))

    @staticmethod
    def _validated_directory(directory: object) -> SymbolDirectory:
        if not isinstance(directory, SymbolDirectory):
            raise ValueError("invalid symbol directory")
        if not directory.records or len(directory.records) > _MAX_DIRECTORY_RECORDS:
            raise ValueError("invalid symbol directory")
        if not _SOURCE_PATTERN.fullmatch(directory.source):
            raise ValueError("invalid symbol directory")
        if directory.as_of.tzinfo is None or directory.as_of.utcoffset() is None:
            raise ValueError("invalid symbol directory")
        normalized_as_of = directory.as_of.astimezone(UTC)
        records = tuple(
            sorted(
                directory.records,
                key=lambda item: (item.symbol, item.name.casefold(), item.exchange.casefold()),
            )
        )
        if len({item.symbol for item in records}) != len(records):
            raise ValueError("invalid symbol directory")
        return SymbolDirectory(records=records, source=directory.source, as_of=normalized_as_of)

    @staticmethod
    def _generation(directory: SymbolDirectory) -> str:
        canonical = json.dumps(
            {
                "records": [record.model_dump(mode="json") for record in directory.records],
                "source": directory.source,
                "as_of": directory.as_of.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _rank(records: tuple[SymbolRecord, ...], query: str) -> tuple[SymbolRecord, ...]:
        query_tokens = tuple(_COMPANY_TOKEN_PATTERN.findall(query))

        def score(record: SymbolRecord) -> tuple[int, str, str, str]:
            name_tokens = tuple(_COMPANY_TOKEN_PATTERN.findall(record.name.upper()))
            if record.symbol.startswith(query):
                rank = 0
            elif query_tokens and all(
                any(name_token.startswith(query_token) for name_token in name_tokens)
                for query_token in query_tokens
            ):
                rank = 1
            else:
                rank = 9
            return (rank, record.symbol, record.name.casefold(), record.exchange.casefold())

        return tuple(record for record in sorted(records, key=score) if score(record)[0] < 9)
