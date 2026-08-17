from __future__ import annotations

# mypy: disallow_untyped_decorators=False
import asyncio
import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.auth import verify_app_key
from app.dependencies import increment_rate, rate_limit_identity, setting
from app.market_analysis.domain import MarketAnalysisAnswer
from app.services.market_analysis import MarketAnalysisUnavailableError
from app.validation import (
    validate_research_question,
    validate_symbol,
    validate_symbol_query,
)

MAX_RANGE_DAYS = 365
SYMBOL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,31}$"
DEFAULT_RESEARCH_REQUEST_BYTES = 4096
DEFAULT_RESEARCH_RATE_LIMIT = 10
DEFAULT_RESEARCH_DAILY_BUDGET = 100


@dataclass(frozen=True, slots=True)
class RequestAuthorization:
    using_session: bool
    principal_digest: str


def envelope(
    request: Request,
    data: Any,
    metadata: object | None = None,
) -> dict[str, Any]:
    encoded_data = jsonable_encoder(data, custom_encoder={Decimal: str})
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
        meta = {
            **{
                key: value
                for key, value in encoded_data.items()
                if key not in {"items", "next_cursor", "total", "limit"}
            }
        }
    else:
        meta = {}
    meta = {
        **meta,
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


class ResearchQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    symbol: str = Field(min_length=1, max_length=32)
    question: str = Field(min_length=1, max_length=500)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return validate_symbol(value)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return validate_research_question(value)


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


def optional_service_method(
    service: object,
    name: str,
    *,
    unavailable_message: str,
    enabled: bool = True,
) -> Callable[..., Any]:
    method = getattr(service, name, None)
    if not enabled or method is None or not callable(method):
        raise HTTPException(503, detail=unavailable_message)
    return cast(Callable[..., Any], method)


async def invoke(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def build_api_router(
    settings: object,
    service: object,
    cache: object | None,
    *,
    session_authorizer: Callable[[Request], bool] | None = None,
    csrf_validator: Callable[[Request, str | None], None] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    async def authorize(
        request: Request,
        x_app_key: str | None = Header(default=None, alias="X-App-Key"),
    ) -> RequestAuthorization:
        digest = setting(settings, "app_access_key_sha256", "")
        has_session = bool(session_authorizer and session_authorizer(request))
        has_app_key = verify_app_key(x_app_key, digest)
        if not has_app_key and not has_session:
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
        principal_digest = hashlib.sha256(client.encode("utf-8")).hexdigest()
        return RequestAuthorization(
            using_session=has_session and not has_app_key,
            principal_digest=principal_digest,
        )

    async def enforce_research_rate_limit(request: Request) -> None:
        window = int(setting(settings, "rate_limit_window_seconds", 60))
        client_limit = int(setting(settings, "research_rate_limit", DEFAULT_RESEARCH_RATE_LIMIT))
        client = rate_limit_identity(request, settings)
        try:
            client_count = await increment_rate(
                cache,
                f"rate:research:{client}",
                window,
            )
        except Exception as exc:
            raise HTTPException(503, detail="rate limiter unavailable") from exc
        if client_count > client_limit:
            raise HTTPException(429, detail="research rate limit exceeded")

    def research_daily_window_seconds() -> int:
        now = datetime.now(UTC)
        next_day = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), UTC)
        return max(1, int((next_day - now).total_seconds()))

    async def enforce_symbol_search_limit(request: Request) -> None:
        window = int(setting(settings, "rate_limit_window_seconds", 60))
        limit = int(setting(settings, "symbol_search_rate_limit", 30))
        client = rate_limit_identity(request, settings)
        try:
            count = await increment_rate(
                cache,
                f"rate:symbol-search:{client}",
                window,
            )
        except Exception as exc:
            raise HTTPException(503, detail="rate limiter unavailable") from exc
        if count > limit:
            raise HTTPException(429, detail="symbol search rate limit exceeded")

    protected = [Depends(authorize)]

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

    @router.get("/usage", dependencies=protected)
    async def usage(request: Request) -> dict[str, Any]:
        data = await call_service(service, ("usage", "get_usage"))
        return envelope(request, data, service_metadata(service))

    @router.get("/symbols/search", dependencies=protected)
    async def symbol_search(
        request: Request,
        q: str = Query(min_length=2, max_length=32),
        limit: int = Query(default=8, ge=1, le=8),
    ) -> dict[str, Any]:
        method = optional_service_method(
            service,
            "search_symbols",
            unavailable_message="symbol search service is unavailable",
        )
        checked_query = validate_symbol_query(q)
        await enforce_symbol_search_limit(request)
        data = await invoke(method, checked_query, limit=limit)
        return envelope(request, data)

    @router.post("/research/query")
    async def research_query(
        request: Request,
        payload: ResearchQueryRequest,
        authorization: RequestAuthorization = Depends(authorize),  # noqa: B008
    ) -> dict[str, Any]:
        if authorization.using_session:
            if csrf_validator is None:
                raise HTTPException(503, detail="research service is unavailable")
            csrf_validator(request, request.headers.get("X-CSRF-Token"))
        method = optional_service_method(
            service,
            "query_research",
            unavailable_message="research service is unavailable",
            enabled=setting(settings, "research_enabled", False) is True,
        )
        maximum_bytes = int(
            setting(settings, "research_max_request_bytes", DEFAULT_RESEARCH_REQUEST_BYTES)
        )
        if len(await request.body()) > maximum_bytes:
            raise HTTPException(413, detail="research request body is too large")
        maximum_question_chars = int(setting(settings, "research_max_question_chars", 500))
        if len(payload.question) > maximum_question_chars:
            raise HTTPException(422, detail="research question exceeds configured maximum")
        await enforce_research_rate_limit(request)
        timeout_seconds = float(setting(settings, "research_timeout_seconds", 8.0))
        daily_limit = int(
            setting(
                settings,
                "research_daily_global_limit",
                DEFAULT_RESEARCH_DAILY_BUDGET,
            )
        )

        async def authorize_generation() -> None:
            try:
                budget_count = await increment_rate(
                    cache,
                    f"budget:market-analysis:{datetime.now(UTC).date().isoformat()}",
                    research_daily_window_seconds(),
                )
            except Exception as exc:
                raise HTTPException(503, detail="rate limiter unavailable") from exc
            if budget_count > daily_limit:
                raise HTTPException(429, detail="research daily budget exceeded")

        try:
            async with asyncio.timeout(timeout_seconds):
                data = await invoke(
                    method,
                    payload.symbol,
                    payload.question,
                    before_generation=authorize_generation,
                )
        except TimeoutError:
            raise HTTPException(504, detail="the research request timed out") from None
        if not isinstance(data, MarketAnalysisAnswer) or data.symbol != payload.symbol:
            data = None
            raise MarketAnalysisUnavailableError("market analysis response was invalid") from None
        return envelope(request, data)

    return router
