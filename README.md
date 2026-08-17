# MarketView

MarketView is an AI-powered market intelligence dashboard and read-only API. It collects daily
stock candles from [Market Data](https://www.marketdata.app/docs/api/) and can use the OpenAI API,
with bounded prompts, to generate summaries and analysis from that collected market data.

MarketView does **not** retrieve or analyze SEC filings. It has no filing RAG pipeline, embedding
workflow, vector database, document corpus, ingestion job, or reranker. The SEC company-ticker
directory is used only to power ticker autocomplete.

The FastAPI application runs on Vercel and uses Upstash Redis for shared caching, rate limits,
locks, symbol-directory generations, and local daily Market Data credit accounting. Browser
authentication uses a signed Secure, HttpOnly, SameSite=Strict cookie.

## Setup

Requires Python 3.12+, a Market Data API token, and Upstash Redis. Configure an OpenAI project key
only when enabling AI-assisted market responses.

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

## Credential boundaries

- `/api/v1/*` is MarketView's client-facing API.
- `https://api.marketdata.app/v1` is the Market Data upstream root used only by the server.
- `MARKETDATA_TOKEN` authenticates outbound Market Data requests and is never a query parameter.
- `OPENAI_API_KEY` is used only by the server for bounded generation calls.
- `APP_ACCESS_KEY_SHA256` is the digest of the separate application login/API value.

Never put provider tokens in `X-App-Key`, browser code, client storage, URLs, logs, prompts, or
source control. Never forward the inbound application key to Market Data or OpenAI.

## Environment

| Variable | Purpose |
| --- | --- |
| `ENVIRONMENT` | `development`, `preview`, or `production`. |
| `MARKETDATA_TOKEN` | Secret Bearer token for outbound Market Data calls; required in Preview and Production. |
| `MARKETDATA_BASE_URL` | Exact upstream root: `https://api.marketdata.app/v1`. |
| `MARKETDATA_DAILY_CREDIT_BUDGET` | Local UTC daily physical-call ceiling; default `90`. |
| `APP_ACCESS_KEY_SHA256` | Digest of the application access value; unrelated to provider tokens. |
| `SESSION_SECRET` | Independent random session-signing secret. |
| `UPSTASH_REDIS_REST_URL` | Upstash Redis REST endpoint. |
| `UPSTASH_REDIS_REST_TOKEN` | Secret Upstash Redis REST token. |
| `ALLOWED_ORIGIN` | Exact browser origin. |
| `ALLOWED_HOSTS` | Comma-separated host allowlist. |
| `SESSION_COOKIE_SECURE` | `false` locally; always `true` in production. |
| `HTTP_TIMEOUT_SECONDS` | Bounded upstream HTTP timeout. |
| `SEC_USER_AGENT` | Contact identity used only by the operator ticker-directory refresh. |
| `SYMBOL_INDEX_SCHEMA_VERSION` | Symbol-index namespace; default `v1`. |
| `SYMBOL_DIRECTORY_MAX_AGE_SECONDS` | Fresh lifetime for a published ticker-directory generation. |
| `SYMBOL_SEARCH_RATE_LIMIT` | Per-client symbol-search ceiling per configured rate window. |
| `RESEARCH_ENABLED` | Opt-in switch for AI-assisted market analysis; safe default `false`. |
| `RESEARCH_MAX_QUESTION_CHARS` | Validated question ceiling; maximum `500`. |
| `RESEARCH_MAX_REQUEST_BYTES` | Research POST body ceiling; default `4096`. |
| `RESEARCH_TIMEOUT_SECONDS` | Application-level research deadline; maximum `10` seconds. |
| `RESEARCH_RATE_LIMIT` | Separate per-client research request ceiling. |
| `RESEARCH_DAILY_GLOBAL_LIMIT` | UTC daily AI request budget; independent of Market Data credits. |
| `OPENAI_API_KEY` | Secret OpenAI project key for server-side AI-assisted responses. |
| `RESEARCH_GENERATION_PROVIDER` | Approved generation provider (`openai`). |
| `RESEARCH_GENERATION_MODEL` | Approved OpenAI generation model. |
| `RESEARCH_GENERATION_MAX_OUTPUT_TOKENS` | Bounded response-token ceiling. |

Keep research disabled unless the OpenAI configuration is complete. No vector or embedding
configuration is required.

## Supported API

The dashboard accepts a stock symbol directly and offers accessible autocomplete backed by a
cached [SEC company-ticker directory](https://www.sec.gov/files/company_tickers_exchange.json).
Autocomplete begins after two characters, returns at most eight ranked matches, and never calls
Market Data or OpenAI. SEC does not guarantee complete or accurate ticker associations, so exact
symbol entry remains available.

- `POST /auth/login` and `POST /auth/logout`
- `GET /api/v1/eod/latest/{symbol}`
- `GET /api/v1/eod/history/{symbol}`
- `GET /api/v1/usage`
- `GET /api/v1/symbols/search?q=app&limit=8`
- `POST /api/v1/research/query` with `{symbol, question}`
- Public `GET /health`
- Authenticated `/docs`, `/redoc`, and `/openapi.json`

Latest and history use Market Data's `/stocks/candles/D/{symbol}/` endpoint. Requests explicitly
set `adjustsplits=false&adjustdividends=false`, so returned prices are unadjusted. Market Data
determines whether candles exist on requested dates. Automatic provider retries are disabled.

Successful market-data responses retain the envelope
`{success, data, meta: {request_id, source, as_of, cached, stale, pagination}, error}`.

## AI-assisted market analysis

MarketView sends only validated, bounded market evidence collected for the requested symbol to the
OpenAI API. Prompt engineering constrains the model to that evidence and asks it to generate a
market summary or analysis. The application validates inputs, limits request and output size,
applies a deadline and budgets, and does not expose provider credentials.

The model is not a market-data source. Market facts and calculations remain traceable to the
collected Market Data records; absent records are never invented. If required evidence is missing,
malformed, stale, or unavailable, the endpoint returns a typed insufficient or unavailable outcome
instead of asking the model to fill the gap. Generated output is AI-assisted information, not
personalized investment advice.

MarketView makes no claim that OpenAI access, paid calls, Preview verification, or deployment has
been completed. Those actions require separate operator authorization.

## Daily credits and cache

The default ceiling is 90 physical Market Data calls per UTC day. A cache hit costs zero credits.
On a cache miss, the service atomically reserves one credit immediately before the single upstream
attempt and rolls it back when Market Data returns no usable data. Rejected reservations cannot
increment the counter past its limit.

`GET /api/v1/usage` reads this local counter only. It does not call Market Data or validate the
provider token.

## Ticker autocomplete and SEC refresh

Interactive searches read immutable, generation-scoped prefix buckets in Upstash Redis. They never
download SEC data during a browser request and never fall back to a hardcoded ticker list. A failed
refresh leaves the previous generation active.

Refresh is an explicit control-plane action:

```powershell
python -m app.refresh_symbols
```

The command performs one bounded fetch under the
[SEC fair-access policy](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data),
validates the company-ticker directory, writes new buckets, and switches the manifest last. It does
not download filings or feed SEC content to OpenAI.

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
`playwright install chromium` then `pytest -m e2e --no-cov`.

## Deployment notes

Create Upstash Redis near the Vercel region, configure secrets in the platform, keep
`RESEARCH_ENABLED=false` initially, and verify health/authentication without live provider calls.
Any live Market Data or OpenAI smoke test, configuration mutation, deployment, or publication is a
separate approval gate.

Market Data restricts a token to one source IP at a time. Vercel does not guarantee stable outbound
IP addresses, so use a supported stable-egress design before relying on this provider in
production. Never rely on serverless instance memory, local files, or background tasks for shared
controls. See [architecture](docs/architecture.md).

## Roadmap

- Improve EOD visualizations and accessible tables without increasing provider use.
- Expand deterministic evidence validation for market summaries and comparisons.
- Add privacy-safe metrics for cache age, provider stages, and budgets.
- Replace the one-owner AI pilot boundary with real identities before multi-user rollout.
