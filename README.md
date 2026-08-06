# Marketstack Dashboard

A quota-conscious FastAPI dashboard and read-only API for selected Marketstack v2 data. It runs
on Vercel and uses Upstash Redis for shared caching, rate limits, locks, and quota accounting.
Browser authentication uses a signed Secure, HttpOnly, SameSite=Strict cookie.

## Setup

Requires Python 3.12+, a Marketstack v2 API key, and Upstash Redis.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

On macOS/Linux, use `source .venv/bin/activate` and `cp .env.example .env`.

Choose a long random access value. Users enter it at login and API clients send it as
`X-App-Key`; configuration stores only its lowercase SHA-256 digest. Derive it without echoing the
raw value or placing it in shell history:

```powershell
python -c "import getpass,hashlib; print(hashlib.sha256(getpass.getpass('Access key: ').encode()).hexdigest())"
```

Set the digest as `APP_ACCESS_KEY_SHA256`. Generate a separate `SESSION_SECRET` with
`python -c "import secrets; print(secrets.token_urlsafe(48))"`. Complete `.env`, then run
`uvicorn api.index:app --reload --host 127.0.0.1 --port 8000`.

## Environment

| Variable | Purpose |
| --- | --- |
| `ENVIRONMENT` | `development` or `production`. |
| `MARKETSTACK_API_KEY` | Secret provider credential. |
| `MARKETSTACK_BASE_URL` | Use `https://api.marketstack.com/v2`. |
| `MARKETSTACK_MONTHLY_BUDGET` | Physical-call ceiling, normally `90`. |
| `APP_ACCESS_KEY_SHA256` | Digest of the shared login/API value. |
| `SESSION_SECRET` | Independent random session secret. |
| `UPSTASH_REDIS_REST_URL` | Upstash REST endpoint. |
| `UPSTASH_REDIS_REST_TOKEN` | Secret Upstash REST token. |
| `ALLOWED_ORIGIN` | Exact browser origin. |
| `ALLOWED_HOSTS` | Comma-separated host allowlist. |
| `SESSION_COOKIE_SECURE` | `false` locally; always `true` in production. |
| `HTTP_TIMEOUT_SECONDS` | Bounded HTTP timeout. |

Never store the raw access value in `.env`, Vercel, logs, source control, or browser JavaScript.
The provider key, session secret, and Upstash token are also secrets.

## Development

```powershell
ruff check .
ruff format --check .
mypy app api
pytest -m "not e2e"
python -m pip_audit
```

Pytest enforces 80% line and branch coverage. Unit/integration tests use fakes and must not call
external services. Optional browser tests use `playwright install chromium` then `pytest -m e2e --no-cov`;
set repository variable `RUN_PLAYWRIGHT=true` to enable that CI job.

## Free-plan guardrails and smoke tests

The 90-call ceiling leaves headroom below a 100-request allowance. Every physical Marketstack
attempt, including failures, is reserved atomically in Redis. Cache hits cost no calls, automatic
upstream retries are disabled, and accounting failures fail closed. Confirm current plan limits
before changing the ceiling.

Supported v1 routes cover tickers, exchanges, EOD latest/history, splits, dividends, and usage.
There are no v1 routes for intraday/real-time prices, bonds, or ETF holdings; those need separate
plan verification, quota modelling, cache policy, schemas, and tests.

The HTTP surface is:

- `POST /auth/login` and `POST /auth/logout`
- `GET /api/v1/tickers` and `GET /api/v1/exchanges`
- `GET /api/v1/eod/latest/{symbol}` and `GET /api/v1/eod/history/{symbol}`
- `GET /api/v1/splits/{symbol}` and `GET /api/v1/dividends/{symbol}`
- `GET /api/v1/usage` and public `GET /health`
- Authenticated `/docs`, `/redoc`, and `/openapi.json`

Successful market-data responses use the envelope
`{success, data, meta: {request_id, source, as_of, cached, stale, pagination}, error}`.
Paginated endpoints place `next_cursor`, `total`, and any available `limit` in
`meta.pagination`.

By default, smoke only `GET /health`; it does not call Marketstack. Authenticated
`GET /api/v1/usage` also avoids the provider. Only after accepting a possible one-call charge,
inspect usage, request `GET /api/v1/tickers?limit=1`, then inspect usage again. A cache hit may
spend zero; a miss may spend one. Keep `X-App-Key` in protected client storage.

## Upstash and Vercel

1. Create Upstash Redis near the Vercel region and copy its REST URL/token into secrets.
2. Import the repository; `.python-version` selects 3.12 and `vercel.json` routes to
   `api/index.py` while bundling templates/assets.
3. Add all environment variables, the final HTTPS origin/host, and secure production cookies.
4. Deploy, then check `/health` before an opt-in live provider request.

Never rely on serverless instance memory, local files, or background tasks for shared controls.
See [architecture](docs/architecture.md).

## Roadmap

- Improve EOD visualizations and accessible tables without increasing provider use.
- Add metrics for cache age, stale responses, locks, and remaining budget.
- Evaluate intraday/real-time, bonds, and ETF holdings in a later version after plan review.
- Consider cache warming only if provider and Vercel plans support it.
