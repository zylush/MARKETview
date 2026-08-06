# Architecture

## Scope and components

This read-only Marketstack v2 facade is constrained by a small monthly allowance and Vercel's
stateless functions. Redis controls are therefore correctness requirements, not optional speedups.
The v1 surface covers tickers, exchanges, EOD latest/history, splits, dividends, and usage. It
excludes intraday/real-time data, bonds, and ETF holdings.

```text
Browser / API client
        |
Vercel rewrite -> api/index.py -> FastAPI routes/security
                                      |
                              market data service
                                /           \
                         Upstash Redis   Marketstack v2
```

`api/index.py` is the ASGI entry. `app/` owns settings, validation, security, adapters, and
services. `templates/` and `static/` are bundled with the function. The browser session is stored
in a signed Secure, HttpOnly, SameSite=Strict cookie. Upstash holds cache entries, rate limits,
distributed locks, and monthly quota accounting. Nothing depends on writable local storage,
background workers, shared process memory, or instance-local locks.

## Market-data request flow

1. Middleware assigns a request ID and applies host/origin/header policy.
2. The route authenticates a signed session or hashes `X-App-Key` with SHA-256 and compares it to
   `APP_ACCESS_KEY_SHA256` in constant time.
3. Pydantic validates and normalizes bounded inputs.
4. The route applies the Redis rate limit, then the service reads a versioned normalized cache key.
5. A fresh hit returns without spending provider quota.
6. For stale/miss, it acquires a short Redis lock and double-checks the cache.
7. Redis atomically checks and reserves one UTC monthly budget unit.
8. The provider makes exactly one bounded HTTP attempt without automatic retry.
9. Valid normalized data is cached; the ownership-checked lock is released in `finally`.

Every physical attempt counts, including failure. If rate limiting, locking, or quota reservation
cannot be confirmed, Marketstack is not contacted. Already-readable stale data may be returned
with explicit metadata; otherwise the request fails closed.

## Cache and data rules

| Data | Fresh TTL |
| --- | ---: |
| Tickers/exchanges | 7 days |
| Completed EOD history | 30 days |
| Latest EOD and history ending today | 6 hours |
| Splits/dividends | 24 hours |
| Invalid/not-found results | 1 hour |

Valid cached market data remains eligible as a stale fallback for 7 days beyond its fresh TTL.

Keys contain a schema version, endpoint, and SHA-256 digest of canonical parameters. EOD history
is limited to one year and pagination is bounded. External payloads are validated before caching.
The default monthly counter ceiling is 90, and atomic check/increment prevents races exceeding it.

## Security

One high-entropy user value supports login and the `X-App-Key` header. Only its lowercase SHA-256
digest is configured; the raw value is never persisted, logged, returned, or put in JavaScript.
`SESSION_SECRET` is independent. Production cookies are Secure, HttpOnly, SameSite=Strict.
Login/logout require CSRF and same-origin checks. Login/API rate limits use Redis and fail closed.
Protected content is private/no-store. Docs/OpenAPI are authenticated; `/health` is public and does
not contact Marketstack.

Responses use restrictive CSP, nosniff, clickjacking protection, a same-origin referrer policy,
permissions
policy, and production HSTS. Jinja escapes external data; browser code uses `textContent` rather
than raw HTML injection.

## API and operations

JSON uses a consistent `success`, `data`, `meta`, and `error` envelope. Successful market-data
metadata contains `request_id`, `source`, `as_of`, `cached`, `stale`, and `pagination`; page data is
returned as an array while cursor/total information is carried in `meta.pagination`. Validation is
`422`, authentication/authorization `401`/`403`, rate/budget limits `429`, provider failures
without stale data `502`, provider timeouts `504`, and unavailable Redis controls `503`. Stale
success is labelled in metadata.

The v1 API routes are `GET /api/v1/tickers`, `GET /api/v1/exchanges`,
`GET /api/v1/eod/latest/{symbol}`, `GET /api/v1/eod/history/{symbol}`,
`GET /api/v1/splits/{symbol}`, `GET /api/v1/dividends/{symbol}`, and
`GET /api/v1/usage`. Login and logout are `POST /auth/login` and `POST /auth/logout`;
`/docs`, `/redoc`, and `/openapi.json` are authenticated, while `GET /health` is public.

Vercel rewrites to `api/index.py`; `.python-version` selects Python 3.12. Production startup must
reject missing secrets, insecure cookies, or incomplete origin/host policy. Operational checks
start with `/health` and `/api/v1/usage`, which consume no Marketstack calls. Tests replace Redis
and provider transports; browser tests are separately gated. A live smoke is manual and limited to
one bounded request after checking remaining budget.

## Extension policy

Intraday/real-time, bonds, and ETF holdings are not ad-hoc v1 additions. They require verified plan
support, request-cost accounting, endpoint-specific cache/stale policy, schemas, security review,
tests, and an updated budget model; a new API version is preferred when semantics change.
