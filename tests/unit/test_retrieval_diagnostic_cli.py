from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import SecretStr

from app.retrieval_diagnostic import EXIT_CONFIG, EXIT_SUCCESS, async_main


@dataclass(frozen=True)
class _Settings:
    openai_api_key: SecretStr | None = field(
        default_factory=lambda: SecretStr("private-openai-key")
    )
    upstash_vector_rest_url: str | None = "https://research-vector.upstash.io"
    upstash_vector_rest_token: SecretStr | None = field(
        default_factory=lambda: SecretStr("private-vector-token")
    )
    upstash_redis_rest_url: str | None = "https://research-redis.upstash.io"
    upstash_redis_rest_token: SecretStr | None = field(
        default_factory=lambda: SecretStr("private-redis-token")
    )
    research_embedding_provider: str = "openai"
    research_embedding_model: str = "text-embedding-3-small"
    research_embedding_dimensions: int = 1536
    research_vector_provider: str = "upstash"
    research_vector_namespace: str = "sec-filings-v1"
    research_index_schema_version: str = "v1"
    research_chunk_tokens: int = 800
    research_chunk_overlap_tokens: int = 100
    research_max_results: int = 5
    research_vector_overfetch: int = 4
    research_minimum_score: float = 0.70
    research_daily_global_limit: int = 100
    research_enabled: bool = False


@pytest.mark.asyncio
async def test_diagnostic_cli_dry_run_constructs_no_runtime_and_emits_no_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_runtime(_: object) -> object:
        raise AssertionError("dry run must not construct providers")

    exit_code = await async_main(
        [],
        settings_factory=_Settings,
        runtime_factory=forbidden_runtime,
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_SUCCESS
    assert output["diagnostic_case"] == "aapl_latest_10k_risks_v1"
    assert output["dry_run"] is True
    assert output["applied"] is False
    assert output["passed"] is False
    assert output["requested_candidate_limit"] == 20
    assert output["call_counts"] == {
        "answer_generation": 0,
        "embedding": 0,
        "vector_search": 0,
    }
    assert "private" not in json.dumps(output)


@pytest.mark.asyncio
@pytest.mark.parametrize("flags", [["--apply"], ["--acknowledge-live-retrieval-diagnostic"]])
async def test_diagnostic_cli_requires_both_live_flags(
    flags: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = await async_main(
        flags,
        settings_factory=lambda: pytest.fail("settings must not be read"),
        runtime_factory=lambda _: pytest.fail("runtime must not be built"),
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == EXIT_CONFIG
    assert output["error_category"] == "authorization"
    assert output["budget_units_committed"] == 0
    assert output["call_counts"]["embedding"] == 0
