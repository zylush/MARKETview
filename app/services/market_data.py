from __future__ import annotations

import asyncio
import inspect
import secrets
from collections.abc import Awaitable, Callable
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from app.cache.base import Cache, CacheEntry, CacheKeyBuilder
from app.errors import (
    CacheUnavailableError,
    MarketDataError,
    ProviderNotFoundError,
    ProviderValidationError,
    QuotaExceededError,
)
from app.models import Dividend, EODBar, Exchange, Page, Split, Ticker, Usage
from app.providers.base import MarketDataProvider
from app.validation import validate_cursor, validate_date_range, validate_limit, validate_symbol

ResultT = TypeVar("ResultT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ServiceTTLs:
    references: int = 7 * 86400
    completed_history: int = 30 * 86400
    latest: int = 6 * 3600
    corporate_actions: int = 24 * 3600
    invalid: int = 3600
    stale: int = 7 * 86400
    usage: int = 5 * 60


@dataclass(frozen=True, slots=True)
class ServiceMetadata:
    source: str
    as_of: datetime
    cached: bool
    stale: bool


class MarketDataService:
    """Provider-neutral, quota-aware cache-aside market data service."""

    def __init__(
        self,
        provider: MarketDataProvider,
        cache: Cache,
        *,
        monthly_budget: int = 90,
        schema_version: str = "v1",
        ttls: ServiceTTLs | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        lock_wait_attempts: int = 5,
        lock_wait_seconds: float = 0.05,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._monthly_budget = monthly_budget
        self._keys = CacheKeyBuilder(schema_version=schema_version)
        self._ttls = ttls or ServiceTTLs()
        self._singleflight: dict[str, asyncio.Lock] = {}
        self._now = now
        self._sleep = sleep
        self._lock_wait_attempts = lock_wait_attempts
        self._lock_wait_seconds = lock_wait_seconds
        self._metadata: ContextVar[ServiceMetadata | None] = ContextVar(
            f"market_data_metadata_{id(self)}", default=None
        )

    @property
    def quota_key(self) -> str:
        return self._keys.build(
            "provider_quota", {"month": self._now().astimezone(UTC).strftime("%Y-%m")}
        )

    @property
    def last_metadata(self) -> ServiceMetadata | None:
        """Metadata for the latest call in the current async task/context."""
        return self._metadata.get()

    def _quota_window_seconds(self) -> int:
        current = self._now().astimezone(UTC)
        if current.month == 12:
            following = datetime(current.year + 1, 1, 1, tzinfo=UTC)
        else:
            following = datetime(current.year, current.month + 1, 1, tzinfo=UTC)
        return max(1, int((following - current).total_seconds()))

    async def aclose(self) -> None:
        """Close provider/cache resources while respecting their own ownership semantics."""
        for resource in (self._provider, self._cache):
            close = getattr(resource, "aclose", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result

    def manages(self, resource: object) -> bool:
        """Return whether this service owns lifecycle cleanup for a resource."""
        return resource is self._provider or resource is self._cache

    async def tickers(
        self,
        *,
        search: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> Page[Ticker]:
        checked_limit = validate_limit(limit)
        checked_cursor = validate_cursor(offset if offset is not None else cursor)
        normalized_search = search.strip()[:100] if search and search.strip() else None
        key = self._keys.build(
            "tickers",
            {"search": normalized_search, "limit": checked_limit, "cursor": checked_cursor},
        )

        async def load() -> Page[Ticker]:
            kwargs: dict[str, Any] = {"limit": checked_limit, "cursor": checked_cursor}
            if normalized_search is not None:
                kwargs = {**kwargs, "search": normalized_search}
            return await self._provider.list_tickers(**kwargs)

        return await self._cached(key, self._ttls.references, Page[Ticker], load)

    async def exchanges(
        self,
        *,
        search: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> Page[Exchange]:
        checked_limit = validate_limit(limit)
        checked_cursor = validate_cursor(offset if offset is not None else cursor)
        normalized_search = search.strip()[:100] if search and search.strip() else None
        key = self._keys.build(
            "exchanges",
            {"search": normalized_search, "limit": checked_limit, "cursor": checked_cursor},
        )

        async def load() -> Page[Exchange]:
            kwargs: dict[str, Any] = {"limit": checked_limit, "cursor": checked_cursor}
            if normalized_search is not None:
                kwargs = {**kwargs, "search": normalized_search}
            return await self._provider.list_exchanges(**kwargs)

        return await self._cached(key, self._ttls.references, Page[Exchange], load)

    async def latest_eod(self, symbol: str) -> EODBar:
        checked_symbol = validate_symbol(symbol)
        key = self._keys.build("latest_eod", {"symbol": checked_symbol})
        return await self._cached(
            key,
            self._ttls.latest,
            EODBar,
            lambda: self._provider.latest_eod(checked_symbol),
        )

    async def history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        limit: int = 100,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> Page[EODBar]:
        checked_symbol = validate_symbol(symbol)
        checked_start, checked_end = validate_date_range(start_date, end_date)
        checked_limit = validate_limit(limit)
        checked_cursor = validate_cursor(offset if offset is not None else cursor)
        key = self._keys.build(
            "eod_history",
            {
                "symbol": checked_symbol,
                "start": checked_start,
                "end": checked_end,
                "limit": checked_limit,
                "cursor": checked_cursor,
            },
        )
        ttl = (
            self._ttls.completed_history
            if checked_end < self._now().astimezone(UTC).date()
            else self._ttls.latest
        )
        return await self._cached(
            key,
            ttl,
            Page[EODBar],
            lambda: self._provider.eod_history(
                checked_symbol,
                start_date=checked_start,
                end_date=checked_end,
                limit=checked_limit,
                cursor=checked_cursor,
            ),
        )

    async def eod_history(
        self,
        symbol: str,
        *,
        date_from: date,
        date_to: date,
        limit: int = 100,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> Page[EODBar]:
        return await self.history(
            symbol,
            date_from,
            date_to,
            limit=limit,
            cursor=cursor,
            offset=offset,
        )

    async def splits(
        self,
        symbol: str | None = None,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 100,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> Page[Split]:
        checked_symbol = validate_symbol(symbol) if symbol is not None else None
        checked_dates = self._optional_date_range(date_from, date_to)
        checked_limit = validate_limit(limit)
        checked_cursor = validate_cursor(offset if offset is not None else cursor)
        key = self._keys.build(
            "splits",
            {
                "symbol": checked_symbol,
                "start": checked_dates[0] if checked_dates else None,
                "end": checked_dates[1] if checked_dates else None,
                "limit": checked_limit,
                "cursor": checked_cursor,
            },
        )
        return await self._cached(
            key,
            self._ttls.corporate_actions,
            Page[Split],
            lambda: self._provider.splits(
                checked_symbol,
                start_date=checked_dates[0] if checked_dates else None,
                end_date=checked_dates[1] if checked_dates else None,
                limit=checked_limit,
                cursor=checked_cursor,
            ),
        )

    async def dividends(
        self,
        symbol: str | None = None,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 100,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> Page[Dividend]:
        checked_symbol = validate_symbol(symbol) if symbol is not None else None
        checked_dates = self._optional_date_range(date_from, date_to)
        checked_limit = validate_limit(limit)
        checked_cursor = validate_cursor(offset if offset is not None else cursor)
        key = self._keys.build(
            "dividends",
            {
                "symbol": checked_symbol,
                "start": checked_dates[0] if checked_dates else None,
                "end": checked_dates[1] if checked_dates else None,
                "limit": checked_limit,
                "cursor": checked_cursor,
            },
        )
        return await self._cached(
            key,
            self._ttls.corporate_actions,
            Page[Dividend],
            lambda: self._provider.dividends(
                checked_symbol,
                start_date=checked_dates[0] if checked_dates else None,
                end_date=checked_dates[1] if checked_dates else None,
                limit=checked_limit,
                cursor=checked_cursor,
            ),
        )

    @staticmethod
    def _optional_date_range(
        start_date: date | None, end_date: date | None
    ) -> tuple[date, date] | None:
        if (start_date is None) != (end_date is None):
            raise ValueError("start and end dates must be provided together")
        if start_date is None or end_date is None:
            return None
        return validate_date_range(start_date, end_date)

    async def usage(self) -> Usage:
        try:
            used = await self._cache.current_count(self.quota_key)
        except Exception as exc:
            if isinstance(exc, CacheUnavailableError):
                raise
            raise CacheUnavailableError("cache service is unavailable") from exc
        now = self._now().astimezone(UTC)
        self._metadata.set(ServiceMetadata(source="local", as_of=now, cached=False, stale=False))
        return Usage(
            requests_used=used,
            requests_limit=self._monthly_budget,
            requests_remaining=max(0, self._monthly_budget - used),
        )

    async def _cached(
        self,
        key: str,
        ttl: int,
        model: type[ResultT],
        loader: Callable[[], Awaitable[ResultT]],
    ) -> ResultT:
        entry = await self._read(key)
        if entry is not None and entry.is_fresh:
            return self._decode(entry, model, source="cache")
        stale = entry

        lock = self._singleflight.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._singleflight = {**self._singleflight, key: lock}
        distributed_key: str | None = None
        token: str | None = None
        acquired = False
        try:
            async with lock:
                second = await self._read(key)
                if second is not None and second.is_fresh:
                    return self._decode(second, model, source="cache")
                stale = second or stale
                distributed_key = self._keys.build("singleflight_lock", {"cache_key": key})
                token = secrets.token_urlsafe(24)
                try:
                    acquired = await self._cache.acquire_lock(
                        distributed_key, token, ttl_seconds=30
                    )
                except Exception as exc:
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                    if isinstance(exc, CacheUnavailableError):
                        raise
                    raise CacheUnavailableError("cache service is unavailable") from exc
                if not acquired:
                    for _ in range(self._lock_wait_attempts):
                        await self._sleep(self._lock_wait_seconds)
                        contender = await self._read(key)
                        if contender is not None and contender.is_fresh:
                            return self._decode(contender, model, source="cache")
                        stale = contender or stale
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                    raise CacheUnavailableError("market data refresh is already in progress")
                try:
                    reservation = await self._cache.reserve_quota(
                        self.quota_key,
                        limit=self._monthly_budget,
                        window_seconds=self._quota_window_seconds(),
                    )
                except Exception as exc:
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                    if isinstance(exc, CacheUnavailableError):
                        raise
                    raise CacheUnavailableError("cache service is unavailable") from exc
                if reservation is None:
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                    raise QuotaExceededError("monthly provider quota is exhausted")
                try:
                    result = await loader()
                except (ProviderNotFoundError, ProviderValidationError) as exc:
                    try:
                        await self._cache.set(
                            key,
                            {"status": "invalid", "code": exc.code},
                            ttl_seconds=self._ttls.invalid,
                        )
                    except Exception as cache_exc:
                        if stale is not None:
                            return self._decode(stale, model, source="stale")
                        raise CacheUnavailableError("cache service is unavailable") from cache_exc
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                    raise
                except MarketDataError:
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                    raise
                as_of = self._now().astimezone(UTC)
                try:
                    await self._cache.set(
                        key,
                        {
                            "status": "ok",
                            "payload": result.model_dump(mode="json"),
                            "as_of": as_of.isoformat(),
                        },
                        ttl_seconds=ttl,
                        stale_seconds=self._ttls.stale,
                    )
                except Exception as exc:
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                    if isinstance(exc, CacheUnavailableError):
                        raise
                    raise CacheUnavailableError("cache service is unavailable") from exc
                self._metadata.set(
                    ServiceMetadata(source="provider", as_of=as_of, cached=False, stale=False)
                )
                return result
        finally:
            if acquired and distributed_key is not None and token is not None:
                with suppress(Exception):
                    await self._cache.release_lock(distributed_key, token)
            if not lock.locked():
                self._singleflight = {
                    name: value for name, value in self._singleflight.items() if name != key
                }

    async def _read(self, key: str) -> CacheEntry[Any] | None:
        try:
            return await self._cache.get(key)
        except CacheUnavailableError:
            raise
        except Exception as exc:
            raise CacheUnavailableError("cache service is unavailable") from exc

    def _decode(self, entry: CacheEntry[Any], model: type[ResultT], *, source: str) -> ResultT:
        envelope = entry.value
        if not isinstance(envelope, dict):
            raise CacheUnavailableError("cache contained an invalid entry")
        if envelope.get("status") == "invalid":
            if envelope.get("code") == ProviderNotFoundError.code:
                raise ProviderNotFoundError("market data was not found")
            raise ProviderValidationError("market data request is invalid")
        try:
            result = model.model_validate(envelope["payload"])
        except (KeyError, ValueError, TypeError) as exc:
            raise CacheUnavailableError("cache contained an invalid entry") from exc
        try:
            as_of = datetime.fromisoformat(str(envelope["as_of"]))
        except (KeyError, ValueError, TypeError):
            as_of = self._now().astimezone(UTC)
        self._metadata.set(
            ServiceMetadata(
                source=source,
                as_of=as_of,
                cached=True,
                stale=source == "stale",
            )
        )
        return result
