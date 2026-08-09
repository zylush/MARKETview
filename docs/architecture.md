# Architecture

## Scope and components

This read-only Market Data facade is constrained by a small daily allowance and Vercel's
stateless functions. Redis controls are therefore correctness requirements, not optional speedups.
The public v1 surface covers latest daily EOD, historical daily EOD, local usage, and cached SEC
company-ticker search. Exchange resources, company-detail resources, splits, and dividends are
intentionally absent. An opt-in, citation-grounded SEC-filings research path is implemented for a
one-owner pilot and remains disabled unless its complete approved provider stack is configured.

```text
Browser / API client
        |
Vercel rewrite -> api/index.py -> FastAPI routes/security
                                      |
                    dashboard service composition
                       /              |             \
             market data       symbol search       SEC research
               /    \               |              /    |    \
      Upstash Redis  Market Data /v1 |       OpenAI  Vector  Redis control
                                     +-- Upstash Redis prefix index

Operator-only command -> SEC ticker directory -> validate/bucket -> Upstash manifest switch
Operator-only ingestion -> SEC filings -> parse/chunk/embed -> stage/verify/publish/cleanup
```

`api/index.py` is the ASGI entry. `app/` owns settings, validation, security, adapters, and
services. `templates/` and `static/` are bundled with the function. The browser session is stored
in a signed Secure, HttpOnly, SameSite=Strict cookie. Upstash holds cache entries, rate limits,
distributed locks, and daily UTC credit accounting. Nothing depends on writable local storage,
background workers, shared process memory, or instance-local locks.

## Symbol-directory boundary

`GET /api/v1/symbols/search` validates two-to-32-character queries, caps results at eight, applies
a separate Redis rate limit, and reads only the active manifest plus relevant two-character
prefix buckets. Ranking puts ticker-prefix matches before company-name token-prefix matches and is
deterministic. Results include SEC source, as-of time, and stale status. External text is validated
server-side and inserted into the browser with `textContent` only.

The request path has no directory source and cannot refresh. `python -m app.refresh_symbols` is the
separate control plane: it requires a descriptive contact `SEC_USER_AGENT`, makes one bounded
non-retrying request to the fixed
[SEC directory URL](https://www.sec.gov/files/company_tickers_exchange.json), validates at most
100,000 rows, writes an
immutable generation, and publishes its manifest last. Failed source or bucket writes preserve
the previous generation. SEC does not guarantee ticker-association accuracy or coverage; the UI
therefore preserves exact-symbol submission. This flow costs no Market Data credits.

## Research/RAG boundary

The approved implementation uses [SEC EDGAR submissions and Archives](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
OpenAI `text-embedding-3-small` at exactly 1536 dimensions, OpenAI `gpt-5.6-luna`, a separate
Upstash Vector index, and the existing Upstash Redis instance as the research control plane.
`RESEARCH_ENABLED=false` is the safe default. The OpenAI key, Vector URL, and Vector token form an
all-or-none group; partial credentials or a non-approved provider/model/dimension fail settings
validation. Redis and Vector credentials are distinct and are never interchangeable.

```text
Operator CLI (never web/startup/CI)
  -> validate symbol + CIK + forms + inclusive dates + count
  -> fixed SEC submissions/Archives origins (descriptive User-Agent, <=5 requests/s)
  -> safe HTML/Inline XBRL visible-text parser (no DTD/entity/network loading)
  -> deterministic bounded overlapping chunks
  -> OpenAI embeddings
  -> stage immutable Vector generation
  -> verify expected IDs/count
  -> compare-and-swap the Redis active-generation manifest
  -> checkpoint completion, then clean the superseded Vector generation

Authenticated query
  -> validate/authenticate/CSRF/rate-limit/reserve global budget
  -> embed bounded question
  -> read active manifests from Redis
  -> Vector provider filter by symbol + corpus/model
  -> defensive active-generation/metadata/symbol/score filtering
  -> OpenAI strict structured generation (store=false, no tools)
  -> verify evidence IDs/quotes and construct canonical SEC citations
```

Immutable metadata binds each evidence chunk to symbol, CIK, accession, form, filing date,
canonical SEC Archives URL, parsed-content hash, corpus/chunker/embedding versions, active
generation, ordinal, and deterministic chunk ID. Embedding vectors are separated from evidence
objects and are never sent to answer generation. Raw downloaded filings are held only in bounded
ingestion memory and are not persisted. Normalized public filing chunks live in Vector; Redis
stores only control data such as manifests, checkpoints, locks, and budget counters.

Generation replacement is fail-safe. New points are staged under immutable IDs, verified, and
published through a compare-and-swap Redis control manifest before old points are eligible for
cleanup. An interruption before publication leaves the old generation active; an interruption after
publication leaves the verified new generation active and cleanup resumable. Content hashes and
accessions make reruns idempotent. Neither ingestion nor SEC discovery can execute from an
interactive request, application lifespan hook, Vercel deployment hook, browser, or normal CI.

Retrieved filing text is always untrusted data. The vector query restricts symbol and active
versions, and the core repeats those checks after retrieval. Generated claims must reference
retrieved chunk IDs and verified evidence. URLs are never accepted from the model; citations are
derived from validated SEC metadata and must use canonical Archives HTTPS paths. Weak, absent,
conflicting, cross-symbol, malformed, or unsupported evidence yields `insufficient_evidence`.
Personalized buy/sell/hold requests are refused before paid retrieval. Research is isolated from
deterministic market-price generation, Market Data cache keys, and Market Data credit accounting.

Interactive research uses the existing authenticated POST, same-origin double-submit CSRF for
sessions, a 4 KB body ceiling, a 500-character question ceiling, separate per-client rate limiting,
and an atomic UTC global daily reservation. A reservation is released only when failure is known to
occur before a paid call may have started; timeouts or downstream failures after that marker remain
charged conservatively. The current shared dashboard subject is not a real user identity, so this
is a one-owner pilot with a global budget, not a per-user spending system.

Vercel allows 10 seconds for `api/index.py`; the application deadline defaults to eight seconds and
is shared across embedding, Vector retrieval, and generation. Interactive adapters make no retry.
Long-running SEC discovery and ingestion are operator processes outside the Vercel function.
Operational rollout and every live action remain separately approval-gated; see
[the research runbook](research-runbook.md).

Failed one-filing ingestion recovery uses append-only Redis audit records. The original checkpoint
and first retry claim/result are immutable. A separately authorized second attempt is eligible only
after an exact terminal `vector_verification` failure and creates new attempt-two claim/result keys
with atomic compare-and-set checks against all legacy bytes. Claim authorization requires both the
opaque job digest and the non-public first retry attempt digest, preventing public plan inputs from
consuming the one allowed attempt. The claim is won before SEC discovery;
concurrent, stale, ambiguous, active-generation, cleanup-pending, or inconsistent Vector/control
state fails closed. A claim without a result is never reclaimed automatically, terminal replays make
zero provider calls, and no third attempt exists.

## API, cache, and credential boundaries

| Boundary | Value | Purpose |
| --- | --- | --- |
| Application API | `/api/v1/*` | Stable client-facing routes owned by this service. |
| Provider API | `https://api.marketdata.app/v1` | Exact production root for server-only Market Data requests. |
| SEC discovery | `https://data.sec.gov/submissions/*` | Fixed operator-only filing metadata source. |
| SEC documents | `https://www.sec.gov/Archives/edgar/data/*` | Fixed canonical filing-document source. |
| OpenAI API | `https://api.openai.com/v1` | Fixed server-only embeddings and Responses API root. |
| Vector data | Separate Upstash Vector HTTPS root | Public normalized chunks and immutable generation metadata. |
| Research control | Existing Upstash Redis HTTPS root | Locks, checkpoints, manifests, active pointers, and budgets. |
| Cache schema | `cache_schema_version = "v2"` | Internal Redis compatibility marker, independent of both HTTP APIs. |

`MARKETDATA_TOKEN` is the outbound provider credential. It is unwrapped only at the provider
boundary and sent in exactly one `Authorization: Bearer <token>` header. It is never sent in a URL,
query parameter, `X-App-Key`, browser response, or client asset.

`APP_ACCESS_KEY_SHA256` instead stores the digest of the raw application access value presented by
clients in `X-App-Key` or at login. The application credential is never forwarded upstream, and
the provider token cannot authenticate a client to `/api/v1`.

## Market-data request flow

1. Middleware assigns a request ID and applies host, origin, and header policy.
2. The route authenticates a signed session or hashes `X-App-Key` and compares it to
   `APP_ACCESS_KEY_SHA256` in constant time.
3. Pydantic validates and normalizes the symbol, dates, cursor, and limit.
4. The route applies the Redis rate limit, then the service reads a versioned normalized cache key.
5. A fresh hit returns without spending a provider credit.
6. For a stale entry or miss, the service acquires a short Redis lock and double-checks the cache.
7. Redis atomically checks and reserves one unit from the current UTC day's local budget.
8. The provider makes exactly one bounded HTTP attempt without automatic retry.
9. Usable normalized data is cached. A failed or non-usable attempt atomically rolls back its
   reservation. The ownership-checked lock is released in `finally`.

If rate limiting, locking, or quota reservation cannot be confirmed, Market Data is not contacted.
Already-readable stale data may be returned with explicit metadata; otherwise the request fails
closed. A rejected reservation never increments the counter beyond the configured limit.

## Provider contract and data rules

Both supported operations call `/stocks/candles/D/{symbol}/` beneath the upstream `/v1` root:

| Operation | Query |
| --- | --- |
| Latest | `countback=1&adjustsplits=false&adjustdividends=false` (no `to`, so the provider returns the most recent candle) |
| History | `from=YYYY-MM-DD&to=YYYY-MM-DD&adjustsplits=false&adjustdividends=false` |

The explicit `adjustsplits=false&adjustdividends=false` settings mean returned prices are
unadjusted for stock splits and dividends.
History dates pass through unchanged, including weekends and holidays; documented provider
`no_data` responses become application 404 responses. Provider HTTP `400`, `413`, and `422`
responses are uncached upstream request failures (502), never application validation errors.
Latest/history result keys include a provider-contract revision, which bypasses obsolete negative
entries without changing the global cache schema, rate limits, or daily provider quota key.
HTTP `200` and `203` are successful. HTTP `204` and documented no-data payloads map to not found.
The adapter validates the provider's parallel open, high, low, close, volume, and timestamp arrays
before creating immutable normalized bars. History is ordered chronologically and paginated
locally, so moving through an already-fetched result does not add provider calls.

| Data | Fresh TTL |
| --- | ---: |
| Completed EOD history | 30 days |
| Latest EOD and history ending today | 6 hours |
| Invalid/not-found result | 1 hour |

Valid cached market data remains eligible as a stale fallback for seven days beyond its fresh TTL.
Keys contain schema `v2`, an endpoint name, and a SHA-256 digest of canonical parameters. External
payloads are validated before caching. The history date range is bounded to one year.

The local daily counter uses the UTC calendar date and expires at the following UTC midnight. Its
default ceiling is 90. Cache hits and `/api/v1/usage` consume zero. One credit is reserved only
immediately before a physical request and is rolled back for authentication, access, no-data,
transport, malformed-response, and server failures that yield no usable data.

## Security and failure mapping

One high-entropy application value supports login and the `X-App-Key` header. Only its lowercase
SHA-256 digest is configured; the raw value is never persisted, logged, returned, or put in
JavaScript. `SESSION_SECRET` is independent. Production cookies are Secure, HttpOnly, and
SameSite=Strict. Login and logout require CSRF and same-origin checks. Login and API rate limits
use Redis and fail closed. Protected content is private/no-store. Docs and OpenAPI are
authenticated; `/health` is public and never contacts Market Data.

Responses use restrictive CSP, nosniff, clickjacking protection, a same-origin referrer policy,
permissions policy, and production HSTS. Jinja escapes external data; browser code writes external
values with `textContent`.

Provider logging is restricted to fixed safe fields such as request ID, failure category,
upstream status, and a whitelisted semantic code. It excludes credentials, authorization headers,
provider URLs, parameters, payloads, symbols, dates, and raw exceptions.

Research observability follows the stricter rule: log only request ID, sanitized stage/category,
duration/count buckets, outcome, and whether a cost reservation remains charged. Never log complete
questions, prompts, generated output, filing bodies, chunk text, embeddings, vector payloads,
authorization headers, credentials, raw provider responses, or exceptions retaining requests.

JSON uses a consistent `success`, `data`, `meta`, and `error` envelope. Successful market-data
metadata contains `request_id`, `source`, `as_of`, `cached`, `stale`, and `pagination`. Validation
is `422`; application authentication/authorization is `401`/`403`; local budget and provider
credit exhaustion are `429`; sanitized provider authentication, payment-plan, IP-access, and other
upstream failures are `502`; provider timeouts are `504`; and unavailable Redis controls are
`503`. Stale success is labelled in metadata.

## API and operations

The supported routes are:

- `GET /api/v1/eod/latest/{symbol}`
- `GET /api/v1/eod/history/{symbol}`
- `GET /api/v1/usage`
- `GET /api/v1/symbols/search`
- `POST /api/v1/research/query` (fail-closed unless the complete approved stack is enabled)
- `POST /auth/login` and `POST /auth/logout`
- Authenticated `/docs`, `/redoc`, and `/openapi.json`
- Public `GET /health`

Removed legacy ticker-resource, exchange, split, and dividend routes return `404` and do not call
the service or provider. The dashboard offers SEC-backed autocomplete while preserving direct
symbol submission; selecting a suggestion loads only latest, history, and local usage.

Production requires `MARKETDATA_TOKEN` and an upstream root that normalizes to exactly
`https://api.marketdata.app/v1`. HTTP, alternate hosts, ports, userinfo, queries, fragments,
missing `/v1`, and additional paths fail validation. Startup and CI never make a provider call.

Operational checks begin with public `/health` and authenticated `/api/v1/usage`; both consume
zero provider credits. A live EOD smoke request is optional, can consume one credit on a cache
miss, and must be explicitly approved.

## Vercel migration and egress constraint

Use this sequence without inspecting or printing secret values:

Before step 2, configure `MARKETDATA_BASE_URL=https://api.marketdata.app/v1` and
`MARKETDATA_DAILY_CREDIT_BUDGET=90` in Production and Preview as well.

1. Add `MARKETDATA_TOKEN` as a Sensitive variable in both Production and Preview.
2. Deploy the new code while retaining the old `MARKETSTACK_*` variables for rollback safety.
3. Verify public `GET /health`; this makes zero provider calls.
4. Verify authenticated `GET /api/v1/usage`; this also makes zero provider calls.
5. Only with explicit approval, perform one bounded `GET /api/v1/eod/latest/AAPL` live smoke.
6. Confirm the normalized provider response and exactly one local daily-credit increment on a
   cache miss.
7. Only after those checks pass, remove every old `MARKETSTACK_*` variable; no legacy alias is
   supported.
8. Redeploy after removal so the environment change is applied.

The provider migration intentionally moves session and CSRF cookies to a `marketdata_*` namespace.
Previously issued browser sessions require one new login, while the application credential and
authentication policy remain unchanged.

Never expose `MARKETDATA_TOKEN` to browser code, `X-App-Key`, URLs, logs, or artifacts. Rotate the
token immediately if it was ever exposed.

Market Data permits a token from only one source IP at a time. A new IP may be rejected until the
previous IP has been inactive for approximately five minutes. Vercel serverless egress is not
guaranteed to be stable, so concurrent or region-shifting calls can receive `403` even when the
token is valid. Prefer a single region where practical and introduce supported stable egress
before treating the integration as production-reliable. Token rotation does not solve changing
egress IPs.

## Extension policy

Additional provider resources are not ad-hoc v1 additions. Each requires verified plan support,
credit accounting, endpoint-specific cache and stale policy, schemas, security review, tests, and
an updated budget model. A new public API version is preferred when semantics change.
