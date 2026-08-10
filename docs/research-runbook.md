# SEC Filing Research Runbook

## Status and approval boundary

The production adapters are implemented for the approved pilot stack, but
`RESEARCH_ENABLED=false` remains the safe default. This runbook is a procedure, not authorization.
Do not create resources, add or rotate credentials, make a live SEC/OpenAI/Upstash request, ingest
a filing, enable research, commit, push, or deploy until the user explicitly approves that action.

This feature covers research over SEC 10-K, 10-Q, and relevant 8-K filings. It does not add dividend
data, dividend calculations, dividend UI, split resources, exchange resources, or any other removed
market-data endpoint.

## Approved stack

| Layer | Selection | Data handled |
| --- | --- | --- |
| Filing source | [SEC EDGAR submissions and Archives](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Public filing metadata and documents during operator ingestion only. |
| Embeddings | OpenAI [`text-embedding-3-small`](https://developers.openai.com/api/docs/models/text-embedding-3-small), 1536 dimensions | Normalized public filing chunks and bounded questions. |
| Answer generation | OpenAI [`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna) | Bounded question plus at most the configured number of retrieved evidence chunks. |
| Vector data | Separate [Upstash Vector](https://upstash.com/docs/vector/overall/getstarted) index | Normalized public chunks, vectors, and validated filing/generation metadata. |
| Control state | Existing Upstash Redis | Locks, resumable checkpoints, immutable manifests, active-generation pointers, and budgets. |
| Raw-document storage | None | Downloaded SEC bodies exist only in bounded ingestion memory. |

SEC requests use fixed origins, a descriptive User-Agent, at most five requests per second, bounded
concurrency and sizes, no redirects, and no arbitrary client-provided URL. Review the
[SEC developer guidance](https://www.sec.gov/about/developer-resources) before every automation or
schedule change.

## Price and capacity checkpoint

The following prices were verified on 2026-08-09 and must be rechecked on the official pages before
resource creation, ingestion expansion, or a model change:

- `text-embedding-3-small`: $0.02 per million input tokens.
- `gpt-5.6-luna`: $1.00 per million input tokens and $6.00 per million output tokens.
- Upstash Vector pricing and free-tier limits: check the current
  [Vector pricing page](https://upstash.com/pricing/vector) and the selected region in the console.

The planning estimate assumed 25 symbols, three years, about 600 filings, 24 million parsed tokens,
roughly 34,300 vectors, and 1,000 monthly questions averaging 4,500 input plus 500 output tokens.
Under those assumptions, initial embeddings are about $0.48 and Luna generation about $7.50 per
1,000 questions; query embeddings and vector requests are small. Vector storage was estimated below
1 GB. These are estimates, not caps or invoices. Set provider spend alerts and retain the application
global daily budget.

## Exact environment contract

Start with research disabled. Keep secrets in the platform secret store, never `.env.example`, Git,
terminal history, screenshots, logs, URLs, or browser assets.

```env
RESEARCH_ENABLED=false
RESEARCH_MAX_QUESTION_CHARS=500
RESEARCH_MAX_REQUEST_BYTES=4096
RESEARCH_TIMEOUT_SECONDS=8
RESEARCH_RATE_LIMIT=10
RESEARCH_DAILY_GLOBAL_LIMIT=100

OPENAI_API_KEY=<OpenAI-project-secret>
UPSTASH_VECTOR_REST_URL=<https://separate-vector-index.upstash.io>
UPSTASH_VECTOR_REST_TOKEN=<separate-vector-token>

RESEARCH_EMBEDDING_PROVIDER=openai
RESEARCH_EMBEDDING_MODEL=text-embedding-3-small
RESEARCH_EMBEDDING_DIMENSIONS=1536
RESEARCH_GENERATION_PROVIDER=openai
RESEARCH_GENERATION_MODEL=gpt-5.6-luna
RESEARCH_GENERATION_MAX_OUTPUT_TOKENS=700
RESEARCH_VECTOR_PROVIDER=upstash
RESEARCH_VECTOR_NAMESPACE=sec-filings-v1
RESEARCH_INDEX_SCHEMA_VERSION=v1
RESEARCH_CHUNK_TOKENS=800
RESEARCH_CHUNK_OVERLAP_TOKENS=100
RESEARCH_MAX_RESULTS=5
RESEARCH_VECTOR_OVERFETCH=4
RESEARCH_MINIMUM_SCORE=0.70

SEC_USER_AGENT=MarketView/1.0 monitored-contact@example.com
```

`OPENAI_API_KEY`, `UPSTASH_VECTOR_REST_URL`, and `UPSTASH_VECTOR_REST_TOKEN` are an all-or-none
group. Supplying only one or two fails configuration validation, including while disabled. The
Vector URL/token must belong to a separate 1536-dimensional index; the Redis URL/token do not work
for Vector. The starting score threshold is not permanent—calibrate it against the fixed evaluation
set before widening the corpus.

## Privacy, retention, and observability

OpenAI's [API data controls](https://developers.openai.com/api/docs/guides/your-data) state that API
data is not used to train models unless the customer opts in. Default abuse-monitoring logs may be
retained for up to 30 days. Generation requests set `store=false`; this is not the same as approved
zero-data retention. If zero-data retention or a specific residency is required, obtain provider
approval before enabling the pilot.

SEC filings are public, but user questions are not. Upstash Vector holds normalized public filing
chunks, embeddings, and filing metadata until generation cleanup or index deletion. Confirm the
chosen Upstash region, account controls, retention expectations, and current legal terms before
creating the index. Redis contains only control metadata; raw filing bodies are not retained, so a
rebuild requires another approved SEC download.

Allowed operational fields are request ID, sanitized stage/failure category, duration/count bucket,
outcome, and whether a reservation stayed charged. Never log or serialize complete questions,
prompts, answers, filing bodies, chunk text, evidence quotes, embeddings, vectors, credentials,
Authorization headers, raw provider responses, request URLs containing secrets, or exceptions that
retain request/response objects.

## Ingestion operation

Ingestion is an operator command only. It must not run from `api/index.py`, FastAPI lifespan hooks,
browser code, Vercel build/deployment hooks, normal CI, or a scheduler without separate approval.
The command requires explicit symbol, CIK, form, inclusive dates, and count. Dry-run is the default.

One-symbol, one-filing dry-run:

```powershell
python -m app.ingest_research `
  --symbol AAPL `
  --cik 0000320193 `
  --forms 10-K `
  --from 2025-01-01 `
  --to 2025-12-31 `
  --limit 1 `
  --timeout-seconds 300
```

Even dry-run performs bounded live SEC discovery when real adapters are configured, so it requires
live-SEC approval. It performs no filing download, embedding, vector mutation, manifest switch, or
cleanup. Inspect only sanitized JSON counts and opaque job/accession digests.

After the dry-run plan is accepted, the separately approved mutation adds `--apply`:

```powershell
python -m app.ingest_research `
  --symbol AAPL `
  --cik 0000320193 `
  --forms 10-K `
  --from 2025-01-01 `
  --to 2025-12-31 `
  --limit 1 `
  --timeout-seconds 300 `
  --apply
```

Apply downloads the one planned filing, parses and chunks it, purchases embeddings, stages immutable
Vector points, verifies their IDs/count, compare-and-swaps the Redis active manifest, checkpoints
completion, and only then removes a superseded generation. Document embeddings are prevalidated and
sent as immutable batches of at most 96 inputs under one absolute operation deadline; batches are
never retried automatically. Repeating a successful bounded command is idempotent. If interrupted
before publication, the prior generation stays active. If interrupted after publication, the new
generation stays active and cleanup resumes on the next approved run.

A completed failed checkpoint is different from a successful replay: the ordinary command does not
retry it. After a separate read-only diagnosis and explicit recovery approval, the exact one-filing
pilot may be retried once by adding `--retry-failed` together with `--apply`:

```powershell
python -m app.ingest_research `
  --symbol AAPL `
  --cik 0000320193 `
  --forms 10-K `
  --from 2025-01-01 `
  --to 2025-12-31 `
  --limit 1 `
  --timeout-seconds 300 `
  --apply `
  --retry-failed
```

Recovery preserves the original failed checkpoint byte-for-byte. Redis atomically creates a separate
one-attempt claim and immutable terminal result. Concurrent claims, stale claims, ambiguous or
multi-filing legacy checkpoints, an active generation, a non-cleaned generation, pending cleanup, or
any existing/partial/inconsistent Vector state fail closed before paid embedding. A failed recovery
is not eligible for another automatic attempt. A crash after the claim remains permanently
fail-closed for this single-owner pilot; do not delete its keys or reclaim it automatically. CLI
diagnostics expose only fixed failure-stage codes.

After the first recovery has a terminal `vector_verification` failure and a separately approved
isolated Vector smoke has passed, an operator may authorize exactly one append-only second attempt.
The command requires the exact opaque job digest as the value of
`--retry-failed-attempt-two`:

```powershell
python -m app.ingest_research `
  --symbol AAPL `
  --cik 0000320193 `
  --forms 10-K `
  --from 2025-01-01 `
  --to 2025-12-31 `
  --limit 1 `
  --timeout-seconds 300 `
  --apply `
  --retry-failed-attempt-two <EXACT_64_HEX_JOB_DIGEST>
```

This command is documentation, not permission to execute it. Attempt two must receive a separate
one-time live approval. The authorization digest is the random, non-public attempt digest from the
immutable first retry claim. Obtain it only through an approved read-only control-plane operation,
inject it into that single operator process as
`RESEARCH_RETRY_ATTEMPT_TWO_AUTHORIZATION`, and clear it afterward. The CLI reads this variable only
when `--retry-failed-attempt-two` is present; it does not accept the authorization value as an
argument. Never paste, print, log, report, or persist its literal value in shell history.
The deterministic job digest alone cannot claim attempt two. Before SEC discovery, Redis atomically
compares the exact original checkpoint, first claim, and first terminal result, then creates only a
new immutable attempt-two claim. Its terminal result is written to a separate append-only key.
Neither operation rewrites,
expires, or deletes the three original records. A concurrent claim loses before any provider call;
a claim without a terminal result remains permanently fail-closed; and a repeated terminal command
returns a fixed sanitized replay with zero SEC, OpenAI, or Vector calls. A prior stage other than
`vector_verification`, an active generation, pending cleanup, or inconsistent control/Vector state
is ineligible. There is no automatic retry and no third attempt.

Expected exit behavior is deterministic: zero for a successful dry-run/apply or no-op; nonzero for
argument/configuration, partial filing failure, provider/vector, or control-state failure. Do not
work around a nonzero result by deleting Redis keys or vector points manually.

### Isolated Vector smoke probe

The operator-only Vector smoke is a separate command from ingestion. Its default invocation is a
zero-network dry run:

```powershell
python -m app.vector_smoke
```

It prints fixed-length, domain-separated fingerprints for the Vector endpoint, Vector token, and
OpenAI key. Compare those opaque values with an independently recorded operator baseline; never
record or display the underlying credentials. The OpenAI-key fingerprint can confirm that two
environments use the same key, but it does not independently prove which OpenAI project owns that
key. Confirm project ownership in the OpenAI console as a separate read-only check.

A live smoke requires its own explicit approval and both CLI acknowledgements:

```powershell
python -m app.vector_smoke --apply --acknowledge-live-vector-smoke
```

The live command writes exactly one synthetic 1536-dimensional nonzero point in the fixed
`marketview-nonprod-smoke-v1` namespace. It performs one upsert, bounded read-after-write fetch
polling under one absolute deadline, one exact-ID delete in `finally`, and bounded cleanup
verification. It never constructs or calls SEC, OpenAI, or Redis providers and never retries either
write. A result is successful only when the point was verified and its deletion was also verified.
Do not run it against a Production index without a separate Production-specific approval.
Even a recorded successful smoke does not authorize another ingestion attempt: an append-only second
recovery attempt requires a separate one-time approval after the smoke passes.

### Retrieval-only diagnostic

The operator-only retrieval diagnostic explains why the fixed indexed Apple 10-K risk case has no
safe evidence without calling the answer generator. Its default invocation validates local
configuration and prints a zero-network plan:

```powershell
python -m app.retrieval_diagnostic
```

The case, symbol, filing type, and question are fixed in code and have no CLI overrides. A live run
requires separate approval and both acknowledgements:

```powershell
python -m app.retrieval_diagnostic `
  --apply `
  --acknowledge-live-retrieval-diagnostic
```

The optional `--timeout-seconds` value must be between 1 and 120. A live run requires exactly one
active AAPL generation, no pending cleanup, and an exact Vector inspection. It authorizes one unit
against the existing UTC-daily global research budget, revalidates the safety state, commits the
unit immediately before one query embedding, and performs one Vector search using the configured
result limit multiplied by the configured overfetch factor. It never invokes answer generation.
Confirmed failures before commit release the reservation; an indeterminate commit or any failure
after commit retains the unit.

Output is aggregate-only: counts, rounded score ranges, the configured threshold, fixed rejection
counters, paid-call counts, and the committed budget units. It never contains credentials,
endpoints, provider fingerprints, the internal question, generation or chunk identifiers, filing
text, citations, embeddings, vectors, exception text, or raw provider responses. Exit `0` means a
dry-run or at least one accepted safe hit, exit `1` means retrieval/provider failure, and exit `3`
means usage, configuration, preflight, or budget rejection.

Do not run the live form without separate approval. Keep Preview `RESEARCH_ENABLED=false`; this
diagnostic does not authorize a Preview query, a configuration change, or Production promotion.

## Preview-first rollout and bounded smoke test

Request and record explicit approval for each gate independently:

1. **Resources:** Create a dedicated OpenAI project/key with a low spend cap and alerts. Create one
   separate Upstash Vector index with cosine similarity, 1536 dimensions, and a region appropriate
   for the Vercel function. Do not modify Redis or Market Data credentials.
2. **Environment:** Add all three research provider settings as Sensitive variables to Preview in
   one change, plus the non-secret fixed contract. Keep `RESEARCH_ENABLED=false` and redeploy only
   after deployment approval.
3. **Ingestion:** From a trusted operator environment, run exactly the dry-run above. After reviewing
   its bounded plan, request another approval and run exactly the one-filing `--apply` command.
4. **Enable/query:** Set `RESEARCH_ENABLED=true` in Preview and redeploy only with approval. Sign in
   through the dashboard, select AAPL, and ask one bounded filing question such as “What supply-chain
   risks were disclosed?” Do not issue automatic retries.
5. **Repository/deployment:** Commit, push, and Production deployment each require explicit approval.
   Do not infer them from approval of resources, secrets, ingestion, or the query smoke.

For the one-query smoke, verify all of the following:

- the response is answered with dated canonical `https://www.sec.gov/Archives/edgar/data/...`
  citations, or safely reports insufficient evidence;
- no model-provided or non-SEC URL is clickable;
- refusal works for a personalized buy/sell/hold question without calling retrieval;
- the global research budget changes according to the documented reservation policy;
- `/api/v1/usage` and Market Data quota/cache keys do not change;
- logs contain only safe operational fields;
- total execution remains below the eight-second application deadline and ten-second Vercel limit.

Only after Preview passes may the same resource/environment/ingestion/query sequence be proposed for
Production. Never bulk-ingest, enable a schedule, or expand symbols/date ranges as part of smoke
testing.

## Rollback

1. Set `RESEARCH_ENABLED=false` in the affected Vercel environment and redeploy with approval.
2. Verify authenticated research returns sanitized unavailable behavior and makes zero OpenAI/Vector
   calls; `/health`, `/api/v1/usage`, autocomplete, latest, and history remain healthy.
3. Keep the previously active Redis manifest and both old/new Vector generations through the rollback
   window. Do not delete vectors merely because the flag is off.
4. If a corpus publication caused the incident, atomically restore the prior known-good active
   manifest, then verify retrieval while still disabled before proposing re-enable.
5. Resume or clean an abandoned staged generation only through the operator workflow after approval.

A model, dimension, chunker, schema, or corpus-version change is a migration, not an in-place edit:
use a new namespace/version, re-embed into a new generation, evaluate it, atomically switch, and keep
the previous generation for rollback.

## Credential rotation

Rotate immediately if a credential appears in source control, browser assets, logs, screenshots,
responses, prompts, exception graphs, or an untrusted system.

1. Disable research and redeploy if exposure is suspected.
2. Create a replacement OpenAI project key or rotate the Upstash Vector token in its own provider.
3. Update the corresponding Sensitive variable in Preview and Production without printing either
   old or new values; redeploy each approved environment.
4. Run configuration/health checks while disabled, then one separately approved bounded query.
5. Revoke the old credential after the replacement is verified.
6. Review logs and repository history for the exposure and apply the provider's incident procedure.

Rotate Redis separately only if the Redis control credential was exposed. `MARKETDATA_TOKEN`,
`APP_ACCESS_KEY_SHA256`, and `SESSION_SECRET` are independent credentials and must never be replaced
with an OpenAI, Vector, or Redis secret.
