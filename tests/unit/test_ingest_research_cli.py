from __future__ import annotations

import importlib
import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from app.research.control import IngestionFailureStage, IngestionStageError
from app.research.ingestion import ResearchIngestionRetryError


def test_cli_import_does_not_construct_live_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object, **__: object) -> object:
        raise AssertionError("live adapter constructed at import")

    monkeypatch.setattr("httpx.AsyncClient", fail)

    import app.ingest_research as cli

    assert callable(cli.main)


@pytest.mark.asyncio
async def test_cli_defaults_to_dry_run_and_outputs_sanitized_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import app.ingest_research as cli

    async def fake_run(request: Any, *, deadline: Any) -> SimpleNamespace:
        deadline.raise_if_expired()
        assert request.apply is False
        return SimpleNamespace(
            dry_run=True,
            planned_count=1,
            processed_count=0,
            skipped_count=0,
            failed_count=0,
            inserted_count=0,
            removed_count=0,
            opaque_job_id="j" * 64,
            opaque_accession_ids=("a" * 64,),
            errors=(),
        )

    exit_code = await cli.async_main(
        [
            "--symbol",
            "aapl",
            "--cik",
            "0000320193",
            "--forms",
            "10-K,10-Q",
            "--from",
            "2025-01-01",
            "--to",
            "2025-12-31",
            "--limit",
            "1",
        ],
        runner_factory=lambda _: SimpleNamespace(run=fake_run),
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output == {
        "applied": False,
        "dry_run": True,
        "errors": [],
        "failure_stages": [],
        "failed_count": 0,
        "inserted_count": 0,
        "job_id": "j" * 64,
        "planned_count": 1,
        "processed_count": 0,
        "removed_count": 0,
        "skipped_count": 0,
    }
    assert "AAPL" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_cli_requires_apply_for_mutation_and_maps_partial_failure_to_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import app.ingest_research as cli

    async def fake_run(request: Any, *, deadline: Any) -> SimpleNamespace:
        deadline.raise_if_expired()
        assert request.apply is True
        return SimpleNamespace(
            dry_run=False,
            planned_count=2,
            processed_count=1,
            skipped_count=0,
            failed_count=1,
            inserted_count=2,
            removed_count=0,
            opaque_job_id="b" * 64,
            opaque_accession_ids=("c" * 64,),
            errors=("ingestion failed for one filing",),
        )

    exit_code = await cli.async_main(
        [
            "--symbol",
            "AAPL",
            "--cik",
            "0000320193",
            "--forms",
            "10-K",
            "--from",
            "2025-01-01",
            "--to",
            "2025-12-31",
            "--limit",
            "2",
            "--apply",
        ],
        runner_factory=lambda _: SimpleNamespace(run=fake_run),
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["applied"] is True
    assert output["errors"] == ["ingestion failed for one filing"]
    assert "sec.gov" not in json.dumps(output).lower()


@pytest.mark.asyncio
async def test_cli_retry_failed_is_explicit_and_requires_apply() -> None:
    import app.ingest_research as cli

    seen: dict[str, bool] = {}

    async def fake_run(request: Any, *, deadline: Any) -> SimpleNamespace:
        deadline.raise_if_expired()
        seen["retry_failed"] = request.retry_failed
        return SimpleNamespace(
            dry_run=False,
            planned_count=1,
            processed_count=1,
            skipped_count=0,
            failed_count=0,
            inserted_count=128,
            removed_count=0,
            opaque_job_id="b" * 64,
            errors=(),
            failure_stages=(),
        )

    base = [
        "--symbol",
        "AAPL",
        "--cik",
        "0000320193",
        "--forms",
        "10-K",
        "--from",
        "2025-01-01",
        "--to",
        "2025-12-31",
        "--limit",
        "1",
        "--retry-failed",
    ]
    with pytest.raises(SystemExit) as caught:
        await cli.async_main(base, runner_factory=lambda _: SimpleNamespace(run=fake_run))
    assert caught.value.code == 2

    assert (
        await cli.async_main(
            [*base, "--apply"],
            runner_factory=lambda _: SimpleNamespace(run=fake_run),
        )
        == 0
    )
    assert seen == {"retry_failed": True}


def test_cli_rejects_invalid_allowlist_arguments_before_factory_call() -> None:
    import app.ingest_research as cli

    called = False

    def factory(_: object) -> object:
        nonlocal called
        called = True
        return object()

    async def run() -> None:
        with pytest.raises(SystemExit) as caught:
            await cli.async_main(
                [
                    "--symbol",
                    "bad symbol!",
                    "--cik",
                    "0000320193",
                    "--forms",
                    "10-K",
                    "--from",
                    "2025-01-01",
                    "--to",
                    "2025-12-31",
                ],
                runner_factory=factory,
            )
        assert caught.value.code == 2

    import asyncio

    asyncio.run(run())
    assert called is False


def test_cli_date_parser_accepts_iso_dates_only() -> None:
    import app.ingest_research as cli

    assert cli.parse_iso_date("2025-01-31") == date(2025, 1, 31)
    with pytest.raises(ValueError, match="date must use ISO"):
        cli.parse_iso_date("01/31/2025")


def test_cli_timeout_is_operator_bounded() -> None:
    import app.ingest_research as cli

    assert cli.RunnerConfig(timeout_seconds=900).timeout_seconds == 900.0
    with pytest.raises(ValueError, match="timeout"):
        cli.RunnerConfig(timeout_seconds=901)


@pytest.mark.asyncio
async def test_cli_maps_sanitized_configuration_failure_to_distinct_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import app.ingest_research as cli

    def factory(_: object) -> object:
        raise ValueError("secret OPENAI_API_KEY raw-value")

    exit_code = await cli.async_main(
        [
            "--symbol",
            "AAPL",
            "--cik",
            "0000320193",
            "--forms",
            "10-K",
            "--from",
            "2025-01-01",
            "--to",
            "2025-12-31",
        ],
        runner_factory=factory,
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_CONFIG
    assert output["error_category"] == "configuration"
    assert "raw-value" not in json.dumps(output)


@pytest.mark.asyncio
async def test_cli_propagates_only_fixed_ingestion_failure_stage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import app.ingest_research as cli

    sensitive_sentinel = "private filing text and credential"

    async def fake_run(request: Any, *, deadline: Any) -> object:
        del request
        deadline.raise_if_expired()
        try:
            raise RuntimeError(sensitive_sentinel)
        except RuntimeError:
            raise IngestionStageError(IngestionFailureStage.EMBEDDING) from None

    exit_code = await cli.async_main(
        [
            "--symbol",
            "AAPL",
            "--cik",
            "0000320193",
            "--forms",
            "10-K",
            "--from",
            "2025-01-01",
            "--to",
            "2025-12-31",
            "--apply",
        ],
        runner_factory=lambda _: SimpleNamespace(run=fake_run),
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_PROVIDER
    assert output["failure_stages"] == ["embedding"]
    assert sensitive_sentinel not in json.dumps(output)


@pytest.mark.asyncio
async def test_cli_uses_typed_retry_error_for_lock_recovery_without_message_sniffing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import app.ingest_research as cli

    async def fake_run(request: Any, *, deadline: Any) -> object:
        del request
        deadline.raise_if_expired()
        raise ResearchIngestionRetryError()

    exit_code = await cli.async_main(
        [
            "--symbol",
            "AAPL",
            "--cik",
            "0000320193",
            "--forms",
            "10-K",
            "--from",
            "2025-01-01",
            "--to",
            "2025-12-31",
            "--apply",
            "--retry-failed",
        ],
        runner_factory=lambda _: SimpleNamespace(run=fake_run),
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_LOCK_RECOVERY
    assert output["error_category"] == "lock_recovery"
    assert output["failure_stages"] == []


@pytest.mark.asyncio
async def test_cli_maps_checkpoint_recovery_failure_to_distinct_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import app.ingest_research as cli

    async def fake_run(request: Any, *, deadline: Any) -> object:
        del request
        deadline.raise_if_expired()
        raise ResearchIngestionRetryError()

    exit_code = await cli.async_main(
        [
            "--symbol",
            "AAPL",
            "--cik",
            "0000320193",
            "--forms",
            "10-K",
            "--from",
            "2025-01-01",
            "--to",
            "2025-12-31",
        ],
        runner_factory=lambda _: SimpleNamespace(run=fake_run),
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_LOCK_RECOVERY
    assert output["error_category"] == "lock_recovery"


@pytest.mark.asyncio
@pytest.mark.parametrize("run_fails", [False, True])
async def test_cli_closes_all_default_owned_providers_in_reverse_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_fails: bool,
) -> None:
    import app.ingest_research as cli

    closed: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        async def aclose(self) -> None:
            closed.append(self.name)

    class ProviderUnavailableError(RuntimeError):
        pass

    async def fake_run(request: Any, *, deadline: Any) -> SimpleNamespace:
        del request
        deadline.raise_if_expired()
        if run_fails:
            raise ProviderUnavailableError("provider unavailable")
        return SimpleNamespace(
            dry_run=True,
            planned_count=0,
            processed_count=0,
            skipped_count=0,
            failed_count=0,
            inserted_count=0,
            removed_count=0,
            opaque_job_id="j" * 64,
            errors=(),
        )

    resources = tuple(
        Resource(name) for name in ("embedder", "generator", "control", "vector", "source")
    )

    async def default_factory(_: object) -> object:
        return cli.OwnedResearchIngestionRunner(
            runner=SimpleNamespace(run=fake_run),
            owned_resources=resources,
        )

    monkeypatch.setattr(cli, "_default_runner_factory", default_factory)

    exit_code = await cli.async_main(
        [
            "--symbol",
            "AAPL",
            "--cik",
            "0000320193",
            "--forms",
            "10-K",
            "--from",
            "2025-01-01",
            "--to",
            "2025-12-31",
        ]
    )

    assert exit_code == (cli.EXIT_PROVIDER if run_fails else cli.EXIT_SUCCESS)
    assert closed == ["source", "vector", "control", "generator", "embedder"]
    capsys.readouterr()


@pytest.mark.asyncio
async def test_cli_does_not_close_injected_runner(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import app.ingest_research as cli

    closed = False

    class InjectedRunner:
        async def run(self, request: Any, *, deadline: Any) -> SimpleNamespace:
            del request
            deadline.raise_if_expired()
            return SimpleNamespace(
                dry_run=True,
                planned_count=0,
                processed_count=0,
                skipped_count=0,
                failed_count=0,
                inserted_count=0,
                removed_count=0,
                opaque_job_id="j" * 64,
                errors=(),
            )

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    exit_code = await cli.async_main(
        [
            "--symbol",
            "AAPL",
            "--cik",
            "0000320193",
            "--forms",
            "10-K",
            "--from",
            "2025-01-01",
            "--to",
            "2025-12-31",
        ],
        runner_factory=lambda _: InjectedRunner(),
    )

    assert exit_code == cli.EXIT_SUCCESS
    assert closed is False
    capsys.readouterr()


@pytest.mark.asyncio
async def test_default_factory_closes_constructed_providers_when_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ingest_research as cli
    import app.providers.openai_research as openai_research
    import app.providers.redis_research as redis_research
    import app.providers.upstash_vector as upstash_vector
    from app.research.domain import EmbeddingDescriptor

    closed: list[str] = []

    class FakeEmbedder:
        descriptor = EmbeddingDescriptor(
            provider="openai",
            model="text-embedding-3-small",
            version="openai:text-embedding-3-small:1536:v1",
            dimensions=1536,
        )

        def __init__(self, **_: object) -> None:
            pass

        async def aclose(self) -> None:
            closed.append("embedder")

    class FakeGenerator:
        def __init__(self, **_: object) -> None:
            pass

        async def aclose(self) -> None:
            closed.append("generator")

    class FakeControl:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def aclose(self) -> None:
            closed.append("control")

    def fail_vector(*_: object, **__: object) -> object:
        raise RuntimeError("vector construction failed")

    monkeypatch.setattr(
        cli,
        "_settings",
        lambda: SimpleNamespace(
            openai_api_key=SecretStr("openai-secret"),
            upstash_vector_rest_url="https://vector.example.upstash.io",
            upstash_vector_rest_token=SecretStr("vector-secret"),
            upstash_redis_rest_url="https://redis.example.upstash.io",
            upstash_redis_rest_token=SecretStr("redis-secret"),
            sec_user_agent="Marketstack test ops@example.com",
            research_generation_max_output_tokens=321,
            research_index_schema_version="v7",
            research_chunk_tokens=1200,
            research_chunk_overlap_tokens=150,
            research_vector_namespace="sec-filings-v2",
        ),
    )
    monkeypatch.setattr(openai_research, "OpenAIEmbedder", FakeEmbedder)
    monkeypatch.setattr(openai_research, "OpenAIAnswerGenerator", FakeGenerator)
    monkeypatch.setattr(redis_research, "RedisResearchControl", FakeControl)
    monkeypatch.setattr(upstash_vector, "UpstashVectorStore", fail_vector)

    with pytest.raises(RuntimeError, match="vector construction failed"):
        await cli._default_runner_factory(cli.RunnerConfig(timeout_seconds=6.0))

    assert closed == ["control", "generator", "embedder"]


@pytest.mark.asyncio
async def test_default_factory_uses_canonical_settings_and_checkpoint_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.config as config
    import app.ingest_research as cli
    import app.providers.openai_research as openai_research
    import app.providers.redis_research as redis_research
    import app.providers.sec_filing_parser as sec_parser
    import app.providers.sec_filings as sec_filings
    import app.providers.upstash_vector as upstash_vector
    import app.research.chunker as chunker
    import app.research.ingestion as ingestion
    from app.research.domain import EmbeddingDescriptor

    values = {
        "OPENAI_API_KEY": "openai-secret",
        "UPSTASH_REDIS_REST_TOKEN": "redis-secret",
        "UPSTASH_REDIS_REST_URL": "https://redis.example.upstash.io",
        "UPSTASH_VECTOR_REST_TOKEN": "vector-secret",
        "UPSTASH_VECTOR_REST_URL": "https://vector.example.upstash.io",
        "SEC_USER_AGENT": "Marketstack test ops@example.com",
        "RESEARCH_ENABLED": "true",
        "RESEARCH_VECTOR_NAMESPACE": "sec-filings-v2",
        "RESEARCH_INDEX_SCHEMA_VERSION": "v7",
        "RESEARCH_CHUNK_TOKENS": "1200",
        "RESEARCH_CHUNK_OVERLAP_TOKENS": "150",
        "RESEARCH_GENERATION_MAX_OUTPUT_TOKENS": "321",
        "RESEARCH_TIMEOUT_SECONDS": "6",
        "UPSTASH_VECTOR_NAMESPACE": "sec-filings-v9",
        "RESEARCH_CORPUS_VERSION": "wrong-corpus",
        "RESEARCH_CHUNKER_VERSION": "wrong-chunker",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()

    seen: dict[str, Any] = {}

    class FakeEmbedder:
        descriptor = EmbeddingDescriptor(
            provider="openai",
            model="text-embedding-3-small",
            version="openai:text-embedding-3-small:1536:v1",
            dimensions=1536,
        )

        def __init__(self, *, api_key: SecretStr, **_: object) -> None:
            seen["openai_key"] = api_key

    class FakeGenerator:
        def __init__(self, *, api_key: SecretStr, max_output_tokens: int, **_: object) -> None:
            seen["generator_key"] = api_key
            seen["max_output_tokens"] = max_output_tokens

    class FakeVectorStore:
        def __init__(self, url: str, token: SecretStr, *, namespace: str, **_: object) -> None:
            seen["vector"] = (url, token, namespace)

    class FakeControl:
        def __init__(
            self,
            endpoint: str,
            token: SecretStr,
            *,
            timeout_seconds: float,
            **_: object,
        ) -> None:
            seen["control"] = (endpoint, token, timeout_seconds)

    class FakeChunker:
        def __init__(self, *, max_tokens: int, overlap_tokens: int) -> None:
            seen["chunker"] = (max_tokens, overlap_tokens)

    class FakeCore:
        def __init__(self, **kwargs: object) -> None:
            seen["core"] = kwargs

    class FakeSource:
        def __init__(self, *, user_agent: str) -> None:
            seen["user_agent"] = user_agent

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            seen["runner"] = kwargs

    monkeypatch.setattr(openai_research, "OpenAIEmbedder", FakeEmbedder)
    monkeypatch.setattr(openai_research, "OpenAIAnswerGenerator", FakeGenerator)
    monkeypatch.setattr(upstash_vector, "UpstashVectorStore", FakeVectorStore)
    monkeypatch.setattr(redis_research, "RedisResearchControl", FakeControl)
    monkeypatch.setattr(chunker, "DeterministicSectionChunker", FakeChunker)
    monkeypatch.setattr(sec_filings, "SecFilingSource", FakeSource)
    monkeypatch.setattr(sec_parser, "SecFilingParser", lambda: "parser")
    monkeypatch.setattr(ingestion, "ResearchIngestionRunner", FakeRunner)

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "app.research.service":
            return SimpleNamespace(ResearchCoreService=FakeCore)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    owned_runner = await cli._default_runner_factory(cli.RunnerConfig(timeout_seconds=6.0))
    runner = owned_runner.runner

    assert isinstance(runner, FakeRunner)
    assert seen["max_output_tokens"] == 321
    assert seen["vector"][2] == "sec-filings-v2"
    assert seen["core"]["corpus"].corpus_version == "v7"
    assert seen["core"]["corpus"].chunker_version == "tokens-1200-150-v1"
    assert seen["chunker"] == (1200, 150)
    assert seen["control"][2] == 3.0
    assert seen["runner"]["checkpoints"]._control is seen["core"]["control_plane"]
    assert seen["user_agent"] == "Marketstack test ops@example.com"
    await owned_runner.aclose()
