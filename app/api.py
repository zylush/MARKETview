from __future__ import annotations

# mypy: disallow_untyped_decorators=False
import inspect
from collections.abc import Callable
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder

from app.auth import verify_app_key
from app.dependencies import increment_rate, rate_limit_identity, setting

MAX_RANGE_DAYS = 365
SYMBOL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,31}$"


def envelope(
    request: Request,
    data: Any,
    metadata: object | None = None,
) -> dict[str, Any]:
    encoded_data = jsonable_encoder(data)
    response_data = encoded_data
    pagination: dict[str, Any] | None = None
    if isinstance(encoded_data, dict) and {
        "items",
        "next_cursor",
        "total",
    }.issubset(encoded_data):
        response_data = encoded_data["items"]
        pagination = {
            "next_cursor": encoded_data["next_cursor"],
            "total": encoded_data["total"],
        }
        if "limit" in encoded_data:
            pagination = {**pagination, "limit": encoded_data["limit"]}
    meta: dict[str, Any] = {
        "request_id": request.state.request_id,
        "pagination": pagination,
    }
    if metadata is not None:
        encoded_metadata = jsonable_encoder(metadata)
        if isinstance(encoded_metadata, dict):
            meta = {**meta, **encoded_metadata}
    return {
        "success": True,
        "data": response_data,
        "meta": meta,
        "error": None,
    }


def service_metadata(service: object) -> object | None:
    return getattr(service, "last_metadata", None)


def pagination_params(limit: int, cursor: str | None, offset: int) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if cursor is not None:
        return {**params, "cursor": cursor}
    if offset:
        return {**params, "offset": offset}
    return params


def validate_range(date_from: date | None, date_to: date | None) -> None:
    if (date_from is None) != (date_to is None):
        raise HTTPException(422, detail="date_from and date_to must be provided together")
    if date_from is None or date_to is None:
        return
    if date_from > date_to:
        raise HTTPException(422, detail="date_from must not be after date_to")
    if (date_to - date_from).days > MAX_RANGE_DAYS:
        raise HTTPException(422, detail="date range must not exceed one year")


async def call_service(service: object, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    method = next((getattr(service, name, None) for name in names if hasattr(service, name)), None)
    if method is None:
        raise HTTPException(501, detail="service operation is not configured")
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def build_api_router(
    settings: object,
    service: object,
    cache: object | None,
    *,
    session_authorizer: Callable[[Request], bool] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    async def authorize(
        request: Request,
        x_app_key: str | None = Header(default=None, alias="X-App-Key"),
    ) -> None:
        digest = setting(settings, "app_access_key_sha256", "")
        has_session = bool(session_authorizer and session_authorizer(request))
        if not has_session and not verify_app_key(x_app_key, digest):
            raise HTTPException(401, detail="invalid application key")
        window = int(setting(settings, "rate_limit_window_seconds", 60))
        limit = int(setting(settings, "api_rate_limit", 60))
        client = rate_limit_identity(request, settings)
        try:
            count = await increment_rate(cache, f"rate:api:{client}", window)
        except Exception as exc:
            raise HTTPException(503, detail="rate limiter unavailable") from exc
        if count > limit:
            raise HTTPException(429, detail="rate limit exceeded")

    protected = [Depends(authorize)]

    @router.get("/tickers", dependencies=protected)
    async def tickers(
        request: Request,
        search: str | None = Query(default=None, min_length=1, max_length=100),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0, le=100_000),
        cursor: str | None = Query(default=None, max_length=100),
    ) -> dict[str, Any]:
        params = pagination_params(limit, cursor, offset)
        if search is not None:
            params = {**params, "search": search}
        data = await call_service(service, ("tickers", "list_tickers"), **params)
        return envelope(request, data, service_metadata(service))

    @router.get("/exchanges", dependencies=protected)
    async def exchanges(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0, le=100_000),
        cursor: str | None = Query(default=None, min_length=1, max_length=100),
    ) -> dict[str, Any]:
        data = await call_service(
            service,
            ("exchanges", "list_exchanges"),
            **pagination_params(limit, cursor, offset),
        )
        return envelope(request, data, service_metadata(service))

    @router.get("/eod/latest/{symbol}", dependencies=protected)
    async def latest_eod(
        request: Request,
        symbol: str,
    ) -> dict[str, Any]:
        if not __import__("re").fullmatch(SYMBOL_PATTERN, symbol):
            raise HTTPException(422, detail="invalid symbol")
        data = await call_service(service, ("latest_eod", "eod_latest"), symbol.upper())
        return envelope(request, data, service_metadata(service))

    @router.get("/eod/history/{symbol}", dependencies=protected)
    async def eod_history(
        request: Request,
        symbol: str,
        date_from: date,
        date_to: date,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0, le=100_000),
        cursor: str | None = Query(default=None, min_length=1, max_length=100),
    ) -> dict[str, Any]:
        if not __import__("re").fullmatch(SYMBOL_PATTERN, symbol):
            raise HTTPException(422, detail="invalid symbol")
        validate_range(date_from, date_to)
        if hasattr(service, "history"):
            data = await call_service(
                service,
                ("history",),
                symbol.upper(),
                date_from,
                date_to,
                **pagination_params(limit, cursor, offset),
            )
        else:
            data = await call_service(
                service,
                ("eod_history",),
                symbol.upper(),
                date_from=date_from,
                date_to=date_to,
                **pagination_params(limit, cursor, offset),
            )
        return envelope(request, data, service_metadata(service))

    async def corporate_actions(
        request: Request,
        operation: str,
        symbol: str,
        limit: int,
        offset: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        if not __import__("re").fullmatch(SYMBOL_PATTERN, symbol):
            raise HTTPException(422, detail="invalid symbol")
        params = {
            "symbol": symbol.upper(),
            **pagination_params(limit, cursor, offset),
        }
        data = await call_service(service, (operation,), **params)
        return envelope(request, data, service_metadata(service))

    @router.get("/splits/{symbol}", dependencies=protected)
    async def splits(
        request: Request,
        symbol: str,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0, le=100_000),
        cursor: str | None = Query(default=None, min_length=1, max_length=100),
    ) -> dict[str, Any]:
        return await corporate_actions(request, "splits", symbol, limit, offset, cursor)

    @router.get("/dividends/{symbol}", dependencies=protected)
    async def dividends(
        request: Request,
        symbol: str,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0, le=100_000),
        cursor: str | None = Query(default=None, min_length=1, max_length=100),
    ) -> dict[str, Any]:
        return await corporate_actions(request, "dividends", symbol, limit, offset, cursor)

    @router.get("/usage", dependencies=protected)
    async def usage(request: Request) -> dict[str, Any]:
        data = await call_service(service, ("usage", "get_usage"))
        return envelope(request, data, service_metadata(service))

    return router
