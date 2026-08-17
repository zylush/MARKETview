# Architecture

## Scope

MarketView is a read-only FastAPI facade for Market Data with optional OpenAI-assisted analysis of
collected market evidence. It supports latest daily EOD, historical daily EOD, local usage, cached
ticker search, and bounded market questions. It does not retrieve SEC filings and contains no RAG,
embedding, vector-search, corpus, ingestion, or reranking architecture.

```text
Browser / API client
        |
Vercel rewrite -> api/index.py -> FastAPI routes and security
                                      |
                              service composition
                          /            |             \
                 market evidence   ticker search   AI analysis
                    /    \              |              |
       Upstash Redis   Market Data      |          OpenAI API

Operator command -> SEC company-ticker directory -> validate/bucket -> Redis manifest switch
```

The SEC directory branch exists only for ticker autocomplete. SEC filings are not requested,
stored, searched, or supplied to the model.

## Runtime components

`api/index.py` is the ASGI entry point. `app/` owns settings, validation, security, provider
adapters, and services. `templates/` and `static/` are bundled with the Vercel function.

Upstash Redis provides cache entries, rate limits, distributed locks, immutable ticker-directory
generations, and daily UTC accounting. Shared correctness never depends on writable local storage,
background workers, process memory, or instance-local locks.

Market Data is the sole source of price candles. The OpenAI model receives a bounded selection of
validated market evidence and produces AI-assisted language under a server-controlled prompt. It
does not replace, retrieve, or manufacture market records.

## Request flow

```text
validated {symbol, question}
  -> classify supported market intent
  -> fetch/cache the required latest or historical candles
  -> normalize and validate evidence
  -> compute trusted numeric facts in application code
  -> if evidence is sufficient, make at most one bounded generation call when required
  -> validate and return the response envelope
```

Evidence failures remain explicit. A malformed or sparse successful provider response is
insufficient evidence; a timeout or provider error is unavailable. The model is never asked to
invent missing observations. Deterministic market facts and calculations are kept separate from
generated prose so the application remains the authority for numbers.

The generation request uses a bounded timeout and output-token limit, disables tools, and is sent
only from the server. Complete questions, prompts, answers, provider payloads, credentials, and
authorization headers must not be logged.

## Symbol-directory boundary

`GET /api/v1/symbols/search` validates two-to-32-character queries, caps results at eight, applies
a separate Redis rate limit, and reads only the active manifest and prefix buckets. Ranking is
deterministic. Results include source, as-of time, and stale status.

The interactive request path cannot refresh the directory. `python -m app.refresh_symbols` is an
operator-only control plane that makes one bounded request to the fixed
[SEC company-ticker directory](https://www.sec.gov/files/company_tickers_exchange.json), validates
the response, writes an immutable generation, and publishes its manifest last. Failed writes
preserve the prior generation. This operation neither accesses filings nor invokes OpenAI.

## Security and privacy boundaries

- Browser/API authentication and Market Data/OpenAI credentials are independent.
- Provider credentials remain in server-side environment configuration.
- Symbols and questions are schema-validated and size-bounded at the API boundary.
- Same-origin browser mutations use the existing session and CSRF controls.
- Rate limits, absolute deadlines, and daily budgets bound provider use.
- External market payloads are validated before calculation or generation.
- Generated text is treated as untrusted output before rendering.
- No credentials, raw prompts, full questions, model responses, or provider response bodies are
  written to application logs.

OpenAI data handling and retention depend on the account's current provider terms and controls.
Operators must verify those terms before enabling a live environment. This architecture document
does not claim that live provider verification or deployment has occurred.

## Public surface

- `POST /auth/login` and `POST /auth/logout`
- `GET /api/v1/eod/latest/{symbol}`
- `GET /api/v1/eod/history/{symbol}`
- `GET /api/v1/usage`
- `GET /api/v1/symbols/search`
- `POST /api/v1/research/query`
- `GET /health`
- authenticated API documentation routes

The research endpoint preserves `{symbol, question}`. Its scope is analysis of collected market
information only; disclosure and filing questions are unsupported.

## Deployment boundary

The Vercel application deadline, Redis controls, Market Data daily credit ceiling, and OpenAI
research budget are independent limits. Health and local usage checks make no provider calls.
Live smoke tests, secret changes, resource creation, deployment, and publication require explicit
authorization and are not implied by local test success.

Market Data may enforce source-IP restrictions, while serverless egress addresses may vary. A
stable-egress design is required before production reliance where the provider account requires a
single source IP.
