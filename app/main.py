from __future__ import annotations

# mypy: disallow_untyped_decorators=False
import hmac
import inspect
import logging
import re
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired
from jinja2 import pass_context
from pydantic import SecretStr

from app.api import build_api_router, envelope
from app.auth import SessionSigner, verify_app_key
from app.dependencies import increment_rate, rate_limit_identity, setting
from app.errors import (
    CacheUnavailableError,
    InputValidationError,
    MarketDataError,
    ProviderAccessRestrictedError,
    ProviderAuthenticationError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderValidationError,
    QuotaExceededError,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CSRF_COOKIE = "marketstack_csrf"
logger = logging.getLogger(__name__)
_SAFE_SEMANTIC_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class _UnavailableService:
    """Fail-safe placeholder used only when runtime dependencies are not wired."""


def _default_settings() -> object:
    from app.config import get_settings

    return get_settings()


def _request_id(value: str | None) -> str:
    try:
        return str(uuid.UUID(value)) if value else str(uuid.uuid4())
    except (ValueError, AttributeError):
        return str(uuid.uuid4())


def _error_code(status_code: int) -> str:
    return {
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        422: "validation_error",
        429: "rate_limit_exceeded",
        501: "not_implemented",
        503: "rate_limiter_unavailable",
    }.get(status_code, "request_error")


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "data": None,
            "meta": {"request_id": request_id},
            "error": {"code": code, "message": message},
        },
        headers={"X-Request-ID": request_id},
    )


def _safe_diagnostic(value: object, *, numeric: bool = False) -> int | str | None:
    if numeric:
        return value if isinstance(value, int) and 100 <= value <= 599 else None
    return value if isinstance(value, str) and _SAFE_SEMANTIC_CODE.fullmatch(value) else None


def _log_market_data_failure(request: Request, exc: MarketDataError) -> None:
    logger.warning(
        "market_data_request_failed",
        extra={
            "request_id": getattr(request.state, "request_id", "unavailable"),
            "exception_category": type(exc).__name__,
            "upstream_status": _safe_diagnostic(
                getattr(exc, "upstream_status", None), numeric=True
            ),
            "semantic_code": _safe_diagnostic(getattr(exc, "semantic_code", None)),
        },
    )


def _make_session_signer(secret_value: object, max_age_seconds: int) -> SessionSigner:
    secret = (
        secret_value.get_secret_value()
        if isinstance(secret_value, SecretStr)
        else str(secret_value)
    )
    if len(secret) < 16:
        raise ValueError("session secret must contain at least 16 characters")
    return SessionSigner(secret, max_age_seconds=max_age_seconds)


def _cookie_secure(settings: object) -> bool:
    secure = bool(setting(settings, "cookie_secure", True, "session_cookie_secure"))
    environment = str(setting(settings, "environment", "production", "env")).lower()
    if not secure and environment not in {"local", "development", "dev", "test", "testing"}:
        raise ValueError("non-secure cookies are allowed only in local or test environments")
    return secure


def _origin(settings: object) -> str:
    configured = setting(settings, "allowed_origin", None, "cors_origin")
    if configured:
        return str(configured).rstrip("/")
    origins = setting(settings, "allowed_origins", [])
    if isinstance(origins, str):
        return origins.split(",", maxsplit=1)[0].strip().rstrip("/")
    return str(origins[0]).rstrip("/") if origins else ""


def _valid_origin(request: Request, settings: object) -> bool:
    expected = _origin(settings)
    actual = request.headers.get("origin", "").rstrip("/")
    return bool(expected and actual and hmac.compare_digest(actual, expected))


def _session_subject(request: Request, signer: SessionSigner, cookie_name: str) -> str | None:
    token = request.cookies.get(cookie_name)
    if not token:
        return None
    try:
        return signer.loads(token)
    except (BadSignature, SignatureExpired, ValueError, TypeError):
        return None


async def _form(request: Request) -> dict[str, str]:
    values = await request.form()
    return {key: str(value) for key, value in values.items()}


def _nonce_docs_html(response: HTMLResponse, nonce: str) -> HTMLResponse:
    html = bytes(response.body).decode("utf-8")
    html = html.replace("<script", f'<script nonce="{nonce}"')
    html = html.replace("<style", f'<style nonce="{nonce}"')
    return HTMLResponse(html, status_code=response.status_code)


def create_app(
    *,
    settings: object | None = None,
    service: object | None = None,
    cache: object | None = None,
) -> FastAPI:
    settings = settings or _default_settings()
    service = service or _UnavailableService()
    secure_cookie = _cookie_secure(settings)
    cookie_name = str(setting(settings, "session_cookie_name", "marketstack_session"))
    max_age = int(setting(settings, "session_max_age_seconds", 3600, "session_max_age"))
    signer = _make_session_signer(
        setting(settings, "session_secret", "", "session_secret_key", "secret_key"),
        max_age,
    )
    allowed_origin = _origin(settings)
    configured_hosts = setting(settings, "allowed_hosts", None)
    if isinstance(configured_hosts, str):
        allowed_hosts = [host.strip() for host in configured_hosts.split(",") if host.strip()]
    elif configured_hosts:
        allowed_hosts = [str(host) for host in configured_hosts]
    else:
        origin_host = urlparse(allowed_origin).hostname
        allowed_hosts = [origin_host] if origin_host else ["localhost"]
    templates = Jinja2Templates(directory=BASE_DIR / "templates")

    @pass_context
    def template_url_for(context: dict[str, Any], name: str, **params: Any) -> Any:
        filename = params.pop("filename", None)
        if filename is not None:
            params = {**params, "path": filename}
        return context["request"].url_for(name, **params)

    templates.env.globals["url_for"] = template_url_for

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        closer = getattr(service, "aclose", None)
        if closer is None:
            closer = getattr(cache, "aclose", None)
        if closer is not None:
            result = closer()
            if inspect.isawaitable(result):
                await result

    application = FastAPI(
        title="Marketstack Dashboard API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.service = service
    application.state.cache = cache
    application.mount(
        "/static",
        StaticFiles(directory=BASE_DIR / "static", check_dir=False),
        name="static",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[allowed_origin] if allowed_origin else [],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-App-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @application.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request.state.request_id = _request_id(request.headers.get("X-Request-ID"))
        request.state.csp_nonce = secrets.token_urlsafe(18)
        response = cast(Response, await call_next(request))
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' https://cdn.jsdelivr.net 'nonce-{request.state.csp_nonce}'; "
            f"style-src 'self' https://cdn.jsdelivr.net 'nonce-{request.state.csp_nonce}'; "
            "img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; "
            "frame-ancestors 'none'"
        )
        if secure_cookie:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(HTTPException)
    async def http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        message = str(exc.detail) if isinstance(exc.detail, str) else "request failed"
        return _error_response(request, exc.status_code, _error_code(exc.status_code), message)

    @application.exception_handler(RequestValidationError)
    async def validation_exception(request: Request, exc: RequestValidationError) -> JSONResponse:
        del exc
        return _error_response(request, 422, "validation_error", "request validation failed")

    @application.exception_handler(MarketDataError)
    async def market_data_exception(request: Request, exc: MarketDataError) -> JSONResponse:
        _log_market_data_failure(request, exc)
        if isinstance(exc, (QuotaExceededError, ProviderRateLimitError)):
            code, message = "UPSTREAM_QUOTA_EXHAUSTED", "upstream request quota is exhausted"
        elif isinstance(exc, ProviderAuthenticationError):
            code = "UPSTREAM_AUTHENTICATION_FAILED"
            message = "the market data provider rejected its credentials"
        elif isinstance(exc, ProviderAccessRestrictedError):
            code = "UPSTREAM_ACCESS_RESTRICTED"
            message = "the market data provider does not permit this request"
        elif isinstance(exc, CacheUnavailableError):
            code, message = "CACHE_UNAVAILABLE", "the cache service is unavailable"
        elif isinstance(exc, ProviderNotFoundError):
            code, message = "NOT_FOUND", "market data was not found"
        elif isinstance(exc, (InputValidationError, ProviderValidationError)):
            code, message = "INVALID_REQUEST", "the market data request is invalid"
        elif isinstance(exc, ProviderUnavailableError):
            code, message = "UPSTREAM_UNAVAILABLE", "the market data provider is unavailable"
        else:
            code, message = "UPSTREAM_UNAVAILABLE", "the market data request could not be completed"
        return _error_response(request, exc.status_code, code, message)

    @application.exception_handler(Exception)
    async def unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        del exc
        return _error_response(request, 500, "internal_error", "an unexpected error occurred")

    def render_login(request: Request, csrf_token: str, error: str = "") -> Response:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"csrf_token": csrf_token, "error": error},
        )

    def set_csrf(response: Response, token: str) -> None:
        response.set_cookie(
            CSRF_COOKIE,
            token,
            max_age=max_age,
            secure=secure_cookie,
            httponly=True,
            samesite="strict",
            path="/",
        )

    @application.get("/login", response_class=HTMLResponse, include_in_schema=False)
    @application.get("/", response_class=HTMLResponse)
    async def login_page(request: Request) -> Response:
        if _session_subject(request, signer, cookie_name):
            return RedirectResponse("/dashboard", status_code=303)
        csrf_token = secrets.token_urlsafe(32)
        response = render_login(request, csrf_token)
        set_csrf(response, csrf_token)
        return response

    @application.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request) -> Response:
        subject = _session_subject(request, signer, cookie_name)
        if not subject:
            return RedirectResponse("/", status_code=303)
        csrf_token = secrets.token_urlsafe(32)
        response = templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={"csrf_token": csrf_token, "username": subject, "initial_symbol": "AAPL"},
        )
        set_csrf(response, csrf_token)
        return response

    async def enforce_login_limit(request: Request) -> None:
        window = int(setting(settings, "rate_limit_window_seconds", 60))
        limit = int(setting(settings, "login_rate_limit", 5))
        client = rate_limit_identity(request, settings)
        try:
            count = await increment_rate(cache, f"rate:login:{client}", window)
        except Exception as exc:
            raise HTTPException(503, detail="rate limiter unavailable") from exc
        if count > limit:
            raise HTTPException(429, detail="rate limit exceeded")

    def validate_csrf(request: Request, token: str | None) -> None:
        cookie = request.cookies.get(CSRF_COOKIE)
        if not _valid_origin(request, settings):
            raise HTTPException(403, detail="cross-origin request denied")
        if not token or not cookie or not hmac.compare_digest(token, cookie):
            raise HTTPException(403, detail="invalid CSRF token")

    @application.post("/login", include_in_schema=False)
    @application.post("/auth/login")
    async def login(request: Request) -> Response:
        values = await _form(request)
        validate_csrf(request, values.get("csrf_token"))
        await enforce_login_limit(request)
        digest = str(setting(settings, "app_access_key_sha256", ""))
        if not verify_app_key(values.get("password"), digest):
            token = secrets.token_urlsafe(32)
            response = render_login(request, token, error="Invalid credentials")
            response.status_code = 401
            set_csrf(response, token)
            return response
        response = RedirectResponse("/dashboard", status_code=303)
        response.set_cookie(
            cookie_name,
            signer.dumps("dashboard"),
            max_age=max_age,
            secure=secure_cookie,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return response

    @application.post("/logout", include_in_schema=False)
    @application.post("/auth/logout")
    async def logout(request: Request) -> Response:
        if not _session_subject(request, signer, cookie_name):
            raise HTTPException(401, detail="authentication required")
        values = await _form(request)
        validate_csrf(request, values.get("csrf_token") or request.headers.get("X-CSRF-Token"))
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(
            cookie_name,
            path="/",
            secure=secure_cookie,
            httponly=True,
            samesite="strict",
        )
        response.delete_cookie(
            CSRF_COOKIE,
            path="/",
            secure=secure_cookie,
            httponly=True,
            samesite="strict",
        )
        return response

    @application.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return envelope(request, {"status": "ok"})

    async def require_docs_auth(request: Request) -> None:
        if _session_subject(request, signer, cookie_name):
            return
        app_key = request.headers.get("X-App-Key")
        if not verify_app_key(app_key, setting(settings, "app_access_key_sha256", "")):
            raise HTTPException(401, detail="authentication required")
        window = int(setting(settings, "rate_limit_window_seconds", 60))
        limit = int(setting(settings, "api_rate_limit", 60))
        client = rate_limit_identity(request, settings)
        try:
            count = await increment_rate(cache, f"rate:docs:{client}", window)
        except Exception as exc:
            raise HTTPException(503, detail="rate limiter unavailable") from exc
        if count > limit:
            raise HTTPException(429, detail="rate limit exceeded")

    @application.get("/openapi.json", include_in_schema=False)
    async def openapi_schema(request: Request) -> JSONResponse:
        await require_docs_auth(request)
        return JSONResponse(application.openapi())

    @application.get("/docs", include_in_schema=False)
    async def swagger_docs(request: Request) -> HTMLResponse:
        await require_docs_auth(request)
        response = get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="Marketstack API docs",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            swagger_favicon_url="/static/favicon.ico",
        )
        return _nonce_docs_html(response, request.state.csp_nonce)

    @application.get("/redoc", include_in_schema=False)
    async def redoc_docs(request: Request) -> HTMLResponse:
        await require_docs_auth(request)
        response = get_redoc_html(
            openapi_url="/openapi.json",
            title="Marketstack API reference",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js",
            redoc_favicon_url="/static/favicon.ico",
            with_google_fonts=False,
        )
        return _nonce_docs_html(response, request.state.csp_nonce)

    application.include_router(
        build_api_router(
            settings,
            service,
            cache,
            session_authorizer=lambda request: bool(_session_subject(request, signer, cookie_name)),
        )
    )
    return application
