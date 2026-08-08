from __future__ import annotations

import asyncio
import inspect
import secrets
from collections.abc import Awaitable, Callable
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
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
from app.models import EODBar, Page, Usage
from app.providers.base import MarketDataProvider
from app.validation import validate_cursor, validate_date_range, validate_limit, validate_symbol

ResultT = TypeVar("ResultT", bound=BaseModel)
_MAX_HISTORY_BARS = 366


@dataclass(frozen=True, slots=True)
class ServiceTTLs:
    completed_history: int = 30 * 86400
    latest: int = 6 * 3600
    invalid: int = 3600
    stale: int = 7 * 86400


@dataclass(frozen=True, slots=True)
class ServiceMetadata:
    source: str
    as_of: datetime
    cached: bool
    stale: bool


class MarketDataService:
    """Provider-neutral, cache-aside service with a local daily credit guard."""

    def __init__(
        self,
        provider: MarketDataProvider,
        cache: Cache,
        *,
        daily_credit_budget: int = 90,
        schema_version: str = "v2",
        ttls: ServiceTTLs | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        lock_wait_attempts: int = 5,
        lock_wait_seconds: float = 0.05,
    ) -> None:
        if daily_credit_budget <= 0:
            raise ValueError("daily credit budget must be positive")
        self._provider = provider
        self._cache = cache
        self._daily_credit_budget = daily_credit_budget
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

    def _utc_now(self) -> datetime:
        return self._now().astimezone(UTC)

    @staticmethod
    def _next_reset(current: datetime) -> datetime:
        return datetime.combine(current.date() + timedelta(days=1), datetime.min.time(), UTC)

    def _quota_key_for(self, current: datetime) -> str:
        return self._keys.build("provider_quota", {"day": current.date().isoformat()})

    @property
    def quota_key(self) -> str:
        return self._quota_key_for(self._utc_now())

    @property
    def last_metadata(self) -> ServiceMetadata | None:
        """Metadata for the latest call in the current async task/context."""
        return self._metadata.get()

    def _quota_window_seconds(self, current: datetime | None = None) -> int:
        snapshot = current or self._utc_now()
        return max(1, int((self._next_reset(snapshot) - snapshot).total_seconds()))

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
            },
        )
        ttl = (
            self._ttls.completed_history
            if checked_end < self._utc_now().date()
            else self._ttls.latest
        )
        complete = await self._cached(
            key,
            ttl,
            Page[EODBar],
            lambda: self._provider.eod_history(
                checked_symbol,
                start_date=checked_start,
                end_date=checked_end,
                limit=_MAX_HISTORY_BARS,
                cursor=None,
            ),
        )
        return self._paginate_history(complete, limit=checked_limit, cursor=checked_cursor)

    @staticmethod
    def _paginate_history(
        complete: Page[EODBar], *, limit: int, cursor: str | None
    ) -> Page[EODBar]:
        offset = int(cursor or 0)
        total = len(complete.items)
        items = complete.items[offset : offset + limit]
        following = offset + len(items)
        return Page(
            items=items,
            total=total,
            next_cursor=str(following) if following < total else None,
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

    async def usage(self) -> Usage:
        now = self._utc_now()
        quota_key = self._quota_key_for(now)
        failed = False
        try:
            used = await self._cache.current_count(quota_key)
        except CacheUnavailableError:
            raise
        except Exception:
            failed = True
        if failed:
            raise CacheUnavailableError("cache service is unavailable")
        self._metadata.set(ServiceMetadata(source="local", as_of=now, cached=False, stale=False))
        return Usage(
            requests_used=used,
            requests_limit=self._daily_credit_budget,
            requests_remaining=max(0, self._daily_credit_budget - used),
            reset_at=self._next_reset(now),
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
                acquired = await self._acquire_refresh_lock(distributed_key, token, stale)
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

                reservation_time = self._utc_now()
                reservation_key = self._quota_key_for(reservation_time)
                reservation = await self._reserve(
                    reservation_key,
                    self._quota_window_seconds(reservation_time),
                    stale,
                )
                if reservation is None:
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                    raise QuotaExceededError("daily provider quota is exhausted")

                try:
                    result = await loader()
                except MarketDataError as exc:
                    await self._rollback_quota(reservation_key)
                    return await self._handle_provider_error(key, model, stale, exc)
                except BaseException:
                    await self._rollback_quota(reservation_key)
                    raise

                as_of = self._utc_now()
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
                except CacheUnavailableError:
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                    raise
                except Exception:
                    if stale is not None:
                        return self._decode(stale, model, source="stale")
                else:
                    self._metadata.set(
                        ServiceMetadata(source="provider", as_of=as_of, cached=False, stale=False)
                    )
                    return result
                raise CacheUnavailableError("cache service is unavailable")
        finally:
            if acquired and distributed_key is not None and token is not None:
                with suppress(Exception):
                    await self._cache.release_lock(distributed_key, token)
            if not lock.locked():
                self._singleflight = {
                    name: value for name, value in self._singleflight.items() if name != key
                }

    async def _acquire_refresh_lock(
        self, distributed_key: str, token: str, stale: CacheEntry[Any] | None
    ) -> bool:
        failed = False
        try:
            return await self._cache.acquire_lock(distributed_key, token, ttl_seconds=30)
        except CacheUnavailableError:
            if stale is not None:
                return False
            raise
        except Exception:
            if stale is not None:
                return False
            failed = True
        if failed:
            raise CacheUnavailableError("cache service is unavailable")
        return False

    async def _reserve(
        self, quota_key: str, window_seconds: int, stale: CacheEntry[Any] | None
    ) -> int | None:
        failed = False
        try:
            return await self._cache.reserve_quota(
                quota_key,
                limit=self._daily_credit_budget,
                window_seconds=window_seconds,
            )
        except CacheUnavailableError:
            if stale is not None:
                return None
            raise
        except Exception:
            if stale is not None:
                return None
            failed = True
        if failed:
            raise CacheUnavailableError("cache service is unavailable")
        return None

    async def _rollback_quota(self, quota_key: str) -> None:
        with suppress(Exception):
            await asyncio.shield(self._cache.release_quota(quota_key))

    async def _handle_provider_error(
        self,
        key: str,
        model: type[ResultT],
        stale: CacheEntry[Any] | None,
        error: MarketDataError,
    ) -> ResultT:
        if isinstance(error, (ProviderNotFoundError, ProviderValidationError)):
            cache_failed = False
            try:
                await self._cache.set(
                    key,
                    {"status": "invalid", "code": error.code},
                    ttl_seconds=self._ttls.invalid,
                )
            except CacheUnavailableError:
                if stale is not None:
                    return self._decode(stale, model, source="stale")
                raise
            except Exception:
                if stale is not None:
                    return self._decode(stale, model, source="stale")
                cache_failed = True
            if cache_failed:
                raise CacheUnavailableError("cache service is unavailable")
        if stale is not None:
            return self._decode(stale, model, source="stale")
        raise error

    async def _read(self, key: str) -> CacheEntry[Any] | None:
        failed = False
        try:
            return await self._cache.get(key)
        except CacheUnavailableError:
            raise
        except Exception:
            failed = True
        if failed:
            raise CacheUnavailableError("cache service is unavailable")
        return None

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
        except (KeyError, ValueError, TypeError):
            pass
        else:
            try:
                as_of = datetime.fromisoformat(str(envelope["as_of"]))
            except (KeyError, ValueError, TypeError):
                as_of = self._utc_now()
            self._metadata.set(
                ServiceMetadata(
                    source=source,
                    as_of=as_of,
                    cached=True,
                    stale=source == "stale",
                )
            )
            return result
        raise CacheUnavailableError("cache contained an invalid entry")
