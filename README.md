# Market Data Dashboard

A quota-conscious FastAPI dashboard and read-only API for daily stock candles from
[Market Data](https://www.marketdata.app/docs/api/). It runs on Vercel and uses Upstash Redis for
shared caching, rate limits, locks, and local daily credit accounting. Browser authentication uses
a signed Secure, HttpOnly, SameSite=Strict cookie.

## Setup

Requires Python 3.12+, a Market Data API token, and Upstash Redis. The optional SEC-filing research
pilot additionally requires an OpenAI project key and a separate Upstash Vector index; keep it
disabled for the base dashboard setup.

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
| `SEC_USER_AGENT` | Descriptive application/contact identity required by operator SEC refresh and filing-ingestion commands. |
| `SYMBOL_INDEX_SCHEMA_VERSION` | Symbol-index namespace; default `v1`. |
| `SYMBOL_DIRECTORY_MAX_AGE_SECONDS` | Fresh lifetime for a published SEC index generation; default `86400`. |
| `SYMBOL_SEARCH_RATE_LIMIT` | Per-client symbol-search ceiling per configured rate window; default `30`. |
| `RESEARCH_ENABLED` | Opt-in runtime switch; safe default `false`. Enable only after the complete stack and active corpus are verified. |
| `RESEARCH_MAX_QUESTION_CHARS` | Validated research question ceiling; maximum `500`. |
| `RESEARCH_MAX_REQUEST_BYTES` | Research POST body ceiling; default `4096`. |
| `RESEARCH_TIMEOUT_SECONDS` | Application-level research timeout; maximum `10` seconds. |
| `RESEARCH_RATE_LIMIT` | Separate per-client research request ceiling; default `10`. |
| `RESEARCH_DAILY_GLOBAL_LIMIT` | Separate UTC daily research request budget; default `100`, not a Market Data credit counter. |
| `OPENAI_API_KEY` | Secret OpenAI project key used only by server-side research adapters. |
| `UPSTASH_VECTOR_REST_URL` | HTTPS root for the separate 1536-dimension Upstash Vector index; not the Redis URL. |
| `UPSTASH_VECTOR_REST_TOKEN` | Secret token for that Vector index; never reuse the Redis token. |
| `RESEARCH_EMBEDDING_PROVIDER` | Fixed approved value `openai`. |
| `RESEARCH_EMBEDDING_MODEL` | Fixed approved value `text-embedding-3-small`. |
| `RESEARCH_EMBEDDING_DIMENSIONS` | Fixed approved value `1536`; changing it requires a new corpus. |
| `RESEARCH_GENERATION_PROVIDER` | Fixed approved value `openai`. |
| `RESEARCH_GENERATION_MODEL` | Fixed approved value `gpt-5.6-luna`. |
| `RESEARCH_GENERATION_MAX_OUTPUT_TOKENS` | Generation output ceiling; default `700`. |
| `RESEARCH_VECTOR_PROVIDER` | Fixed approved value `upstash`. |
| `RESEARCH_VECTOR_NAMESPACE` | Vector namespace; default `sec-filings-v1`. |
| `RESEARCH_INDEX_SCHEMA_VERSION` | Research metadata/index contract; default `v1`. |
| `RESEARCH_CHUNK_TOKENS` | Deterministic ingestion chunk ceiling; default `800`. |
| `RESEARCH_CHUNK_OVERLAP_TOKENS` | Adjacent-chunk overlap; default `100` and smaller than the chunk ceiling. |
| `RESEARCH_MAX_RESULTS` | Maximum evidence chunks sent to generation; default `5`. |
| `RESEARCH_VECTOR_OVERFETCH` | Retrieval overfetch factor before defensive filtering; default `4`. |
| `RESEARCH_MINIMUM_SCORE` | Retrieval threshold; starts at `0.70` and must be eval-calibrated. |

`OPENAI_API_KEY`, `UPSTASH_VECTOR_REST_URL`, and `UPSTASH_VECTOR_REST_TOKEN` are an all-or-none
configuration group. A partial group fails validation even while research is disabled. Provider,
model, and dimension values are also checked against the approved stack.

## Supported API

The dashboard accepts a stock symbol directly and offers an accessible autocomplete backed by a
cached [SEC company-ticker directory](https://www.sec.gov/files/company_tickers_exchange.json).
Autocomplete begins after two characters, returns at most
eight ranked matches, and never calls Market Data or consumes a Market Data credit. SEC explicitly
does not guarantee that its ticker associations are complete or accurate, so exact-symbol entry
remains available for securities outside that directory.

- `POST /auth/login` and `POST /auth/logout`
- `GET /api/v1/eod/latest/{symbol}`
- `GET /api/v1/eod/history/{symbol}`
- `GET /api/v1/usage`
- `GET /api/v1/symbols/search?q=app&limit=8`
- `POST /api/v1/research/query` (available only when the complete research stack is enabled)
- Public `GET /health`
- Authenticated `/docs`, `/redoc`, and `/openapi.json`

Removed legacy ticker-resource, exchange, split, and dividend paths return `404`. Successful market-data
responses retain the envelope
`{success, data, meta: {request_id, source, as_of, cached, stale, pagination}, error}`.

Latest and history use Market Data's `/stocks/candles/D/{symbol}/` endpoint. Requests explicitly
set `adjustsplits=false&adjustdividends=false`, so prices are unadjusted for splits and
dividends. Latest uses `countback=1`
without a `to` value, which asks Market Data for the most recent candle;
history uses the client's bounded `from` and `to` dates unchanged, including weekends and holidays;
Market Data determines whether candles exist. HTTP `200` and `203` are accepted as success. A `204`
or a documented no-data result maps to not found. Automatic provider retries are disabled.
Provider HTTP `400`, `413`, and `422` responses are treated as uncached upstream request failures,
not client validation errors. Provider-result cache keys carry a candles-contract revision so old
negative entries are bypassed without resetting the independent daily quota counter.

## Daily credits and cache

The default ceiling is 90 physical calls per UTC day. The shared Redis counter resets at the next
00:00 UTC. A cache hit costs zero credits. On a cache miss, the service atomically reserves one
credit immediately before the single upstream attempt and rolls it back when Market Data returns
no usable data, including authentication, access, transport, malformed-response, or server
failures. Rejected reservations cannot increment the counter past its limit.

`GET /api/v1/usage` reads this local counter only. It reports used, limit, remaining, reset time,
and `source=local`; it does not call Market Data or verify `MARKETDATA_TOKEN`.

## Ticker autocomplete and SEC refresh

Interactive searches read only immutable, generation-scoped two-character prefix buckets in
Upstash Redis. A search reads the active manifest and relevant buckets; it never downloads the
SEC directory and never falls back to a hardcoded ticker list. A failed refresh leaves the prior
generation active, and stale results are labelled in response metadata.

Refresh is an explicit control-plane action. Set a descriptive `SEC_USER_AGENT` containing a
monitored contact email, configure the shared Upstash Redis variables, and run:

```powershell
python -m app.refresh_symbols
```

That command performs one bounded, non-retrying fetch under the
[SEC fair-access policy](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
from the
`company_tickers_exchange.json` source, validates the response, writes every new bucket, and
switches the manifest last. Do not run it from a browser request, application startup, normal CI,
or a Vercel background task. The initial fetch and any scheduler activation require explicit
operator approval. The feature adds Redis storage/commands and controlled SEC traffic, but zero
Market Data credits.

## Research/RAG boundary

The selected stack is SEC EDGAR for source filings, OpenAI
[`text-embedding-3-small`](https://developers.openai.com/api/docs/models/text-embedding-3-small)
at 1536 dimensions, OpenAI
[`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna) for strict structured
answers, a separate Upstash Vector index, and the existing Upstash Redis database as the control
plane. Redis stores locks, checkpoints, immutable manifests, active-generation pointers, and
budgets; it is not the vector database. Raw filing documents are not retained. The vector index
stores normalized public filing chunks and validated SEC metadata needed for retrieval.

The adapters and provider-neutral core are fail-closed. `RESEARCH_ENABLED=false` remains the safe
default. If any of the three provider settings is supplied, all three must be present; enabling an
unsupported provider, model, dimension, URL, or namespace fails settings validation. No SEC
download or ingestion runs in a browser request, application startup, deployment, or CI. The
interactive path embeds the bounded question, searches only active symbol/corpus generations,
defensively filters again, and calls generation with `store=false`, no tools, and strict output.

Every returned claim identifies retrieved evidence. Citation URLs are reconstructed from validated
chunk metadata and must be canonical SEC Archives HTTPS URLs; model-supplied URLs are never trusted.
Weak, absent, conflicting, cross-symbol, malformed, or unsupported evidence returns
`insufficient_evidence`. Personalized buy/sell/hold requests are refused. Research cannot generate
or modify deterministic market-price data, its cache, or its Market Data quota.

The API retains session CSRF, a 4 KB request ceiling, a 500-character question ceiling, a separate
per-client rate limit, and a global UTC daily research reservation. The dashboard currently has one
shared subject, so this is explicitly a one-owner pilot, not a real per-user spending boundary. The
Vercel function has a 10-second maximum; the application deadline defaults to eight seconds, with
bounded provider stages and no interactive retries.

As of 2026-08-09, the verified list prices used for the pilot estimate were $0.02 per million input
tokens for `text-embedding-3-small`, and $1.00 per million input tokens plus $6.00 per million output
tokens for `gpt-5.6-luna`. Recheck official pricing before creating resources or increasing the
corpus. OpenAI states that API data is not used for training unless opted in; default abuse-monitoring
logs may be retained for up to 30 days. Calls set `store=false`, but true zero-data retention requires
separate provider approval. See the [research operations runbook](docs/research-runbook.md) for
bounded ingestion, privacy, smoke-test, rollout, rollback, and credential-rotation procedures.

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
4. Add the non-secret symbol/research controls from `.env.example`, keeping
   `RESEARCH_ENABLED=false`, then follow the migration checklist and the separately gated
   [research runbook](docs/research-runbook.md).

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

After the application deployment is healthy, an operator may separately approve and run one SEC
symbol-directory refresh against the Production Redis instance. Confirm that authenticated
`/api/v1/symbols/search?q=ap&limit=8` returns source/as-of/stale metadata and that the Market Data
usage count does not change. RAG resources, secrets, ingestion, live-query smoke tests, commits,
pushes, and deployment remain separate approval gates in the research runbook.

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
- Add privacy-safe metrics for cache age, stale responses, generation state, locks, and budgets.
- Replace the one-owner research pilot boundary with real identities before any multi-user rollout.
