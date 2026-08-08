# Market Data Dashboard

A quota-conscious FastAPI dashboard and read-only API for daily stock candles from
[Market Data](https://www.marketdata.app/docs/api/). It runs on Vercel and uses Upstash Redis for
shared caching, rate limits, locks, and local daily credit accounting. Browser authentication uses
a signed Secure, HttpOnly, SameSite=Strict cookie.

## Setup

Requires Python 3.12+, a Market Data API token, and Upstash Redis.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

On macOS/Linux, use `source .venv/bin/activate` and `cp .env.example .env`.

Choose a long random application access value. Users enter it at login and API clients send it as
`X-App-Key`; configuration stores only its lowercase SHA-256 digest. Derive it without echoing the
raw value or placing it in shell history:

```powershell
python -c "import getpass,hashlib; print(hashlib.sha256(getpass.getpass('Access key: ').encode()).hexdigest())"
```

Set the digest as `APP_ACCESS_KEY_SHA256`. Generate a separate `SESSION_SECRET` with
`python -c "import secrets; print(secrets.token_urlsafe(48))"`. Complete `.env`, then run
`uvicorn api.index:app --reload --host 127.0.0.1 --port 8000`.

### API and credential boundaries

- `/api/v1/*` is this application's public, client-facing API.
- `https://api.marketdata.app/v1` is the Market Data upstream root used only by the server. The two
  `/v1` labels are independent.
- `MARKETDATA_TOKEN` authenticates outbound requests with one `Authorization: Bearer <token>`
  header. It is never a query parameter.
- `APP_ACCESS_KEY_SHA256` stores the digest of the separate value used for application login and
  `X-App-Key` authentication.
- The normalized cache schema is `v2`; this internal version is independent of both HTTP APIs.

Never put the provider token in `X-App-Key`, browser code, client storage, URLs, logs, or source
control. Never forward the inbound application key to Market Data.

## Environment

| Variable | Purpose |
| --- | --- |
| `ENVIRONMENT` | `development`, `preview`, or `production`. |
| `MARKETDATA_TOKEN` | Secret Bearer token for outbound Market Data calls; required in Preview and Production. |
| `MARKETDATA_BASE_URL` | Exact production upstream root: `https://api.marketdata.app/v1`. |
| `MARKETDATA_DAILY_CREDIT_BUDGET` | Local UTC daily physical-call ceiling; default `90`. |
| `APP_ACCESS_KEY_SHA256` | Digest of the application value sent by clients as `X-App-Key`; unrelated to the provider token. |
| `SESSION_SECRET` | Independent random session-signing secret. |
| `UPSTASH_REDIS_REST_URL` | Upstash REST endpoint. |
| `UPSTASH_REDIS_REST_TOKEN` | Secret Upstash REST token. |
| `ALLOWED_ORIGIN` | Exact browser origin. |
| `ALLOWED_HOSTS` | Comma-separated host allowlist. |
| `SESSION_COOKIE_SECURE` | `false` locally; always `true` in production. |
| `HTTP_TIMEOUT_SECONDS` | Bounded provider HTTP timeout. |

## Supported API

The dashboard accepts a stock symbol directly. It does not provide ticker or exchange discovery,
company metadata, split history, or dividend history because those resources are not supported by
this provider integration.

- `POST /auth/login` and `POST /auth/logout`
- `GET /api/v1/eod/latest/{symbol}`
- `GET /api/v1/eod/history/{symbol}`
- `GET /api/v1/usage`
- Public `GET /health`
- Authenticated `/docs`, `/redoc`, and `/openapi.json`

Removed discovery, exchange, split, and dividend paths return `404`. Successful market-data
responses retain the envelope
`{success, data, meta: {request_id, source, as_of, cached, stale, pagination}, error}`.

Latest and history use Market Data's `/stocks/candles/D/{symbol}/` endpoint. Requests explicitly
set `adjustsplits=false`, so prices are unadjusted for splits. Latest uses `countback=1`
without a `to` value, which asks Market Data for the most recent candle;
history uses bounded `from` and `to` dates. A Saturday or Sunday history end is normalized to the
preceding Friday for the provider request and cache key while the public calendar-date contract is
preserved. HTTP `200` and `203` are accepted as success. A `204`
or a documented no-data result maps to not found. Automatic provider retries are disabled.

## Daily credits and cache

The default ceiling is 90 physical calls per UTC day. The shared Redis counter resets at the next
00:00 UTC. A cache hit costs zero credits. On a cache miss, the service atomically reserves one
credit immediately before the single upstream attempt and rolls it back when Market Data returns
no usable data, including authentication, access, transport, malformed-response, or server
failures. Rejected reservations cannot increment the counter past its limit.

`GET /api/v1/usage` reads this local counter only. It reports used, limit, remaining, reset time,
and `source=local`; it does not call Market Data or verify `MARKETDATA_TOKEN`.

## Development and verification

```powershell
ruff check .
ruff format --check .
mypy app api
pytest -m "not e2e"
python -m pip_audit
```

Pytest enforces at least 80% line and branch coverage. Unit and integration tests use fake
transports and must not call external services. Optional browser tests use
`playwright install chromium` then `pytest -m e2e --no-cov`; set the repository variable
`RUN_PLAYWRIGHT=true` to enable that CI job. Startup, CI, and normal tests never make a live
provider request.

## Upstash and Vercel

1. Create Upstash Redis near the Vercel region and copy its REST URL and token into Vercel secrets.
2. Import the repository; `.python-version` selects Python 3.12 and `vercel.json` routes to
   `api/index.py` while bundling templates and assets.
3. Configure the HTTPS origin, host allowlist, and secure production cookies.
4. Follow the migration checklist below and redeploy.

### Exact Vercel migration sequence

Before step 2, also configure `MARKETDATA_BASE_URL=https://api.marketdata.app/v1` and
`MARKETDATA_DAILY_CREDIT_BUDGET=90` in Production and Preview.

1. Add `MARKETDATA_TOKEN` as a Sensitive variable in both Production and Preview.
2. Deploy the new code while retaining the old `MARKETSTACK_*` variables for rollback safety.
3. Verify public `GET /health`; this makes zero provider calls.
4. Verify authenticated `GET /api/v1/usage`; this also makes zero provider calls.
5. Only with explicit approval, perform one bounded `GET /api/v1/eod/latest/AAPL` live smoke.
6. Confirm the normalized provider response and exactly one local daily-credit increment on a
   cache miss.
7. Only after those checks pass, remove every old `MARKETSTACK_*` variable; the old names are not
   compatibility aliases.
8. Redeploy after removal because existing deployments do not inherit environment changes.

The migration also changes the session and CSRF cookie namespace to `marketdata_*`. Existing
browser sessions will need to sign in once after the new deployment; application credentials and
authentication rules are otherwise unchanged.

Never expose `MARKETDATA_TOKEN` to browser code, `X-App-Key`, URLs, logs, or public artifacts.
Rotate the token immediately if it was ever exposed.

Inspect variable names and URL categories during migration, never secret values. No live request
is necessary to validate startup, health, authentication, local usage, or the test suite.

### Vercel egress risk

Market Data restricts a token to one source IP at a time. A different IP may be rejected until the
previous IP has been inactive for approximately five minutes. Vercel serverless functions do not
guarantee one stable outbound IP, so concurrent or region-shifting invocations can produce `403`
access failures even with a valid token. Keep the application in one region where practical and
use a supported stable-egress design before relying on this provider in production. Do not rotate
or weaken authentication merely to hide an egress-policy failure.

Never rely on serverless instance memory, local files, or background tasks for shared controls.
See [architecture](docs/architecture.md).

## Roadmap

- Improve EOD visualizations and accessible tables without increasing provider use.
- Add metrics for cache age, stale responses, locks, and remaining daily credits.
- Evaluate additional provider resources only after plan, quota, schema, security, and cache-policy
  review.
