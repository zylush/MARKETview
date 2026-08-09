from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

import api.index as runtime
import app.services.research as research_runtime
from api.index import _runtime_dependencies
from app.cache.upstash import UpstashCache
from app.config import Settings
from app.research.domain import EmbeddingDescriptor
from app.services.dashboard import DashboardService
from app.services.market_data import MarketDataService
from app.services.research import DisabledResearchService, ResearchRuntime
from app.services.symbols import SymbolSearchService
from tests.unit.test_config import complete_research_values, secure_production_values


def test_vercel_uses_fastapi_zero_config_routing() -> None:
    config = json.loads(Path("vercel.json").read_text(encoding="utf-8"))

    assert config["functions"]["api/index.py"]["includeFiles"] == "{templates,static}/**/*"
    assert "rewrites" not in config


def test_vercel_runtime_never_selects_process_local_cache() -> None:
    settings = Settings(
        environment="development",
        vercel="1",
        **secure_production_values(),
        _env_file=None,
    )

    _, service, cache = _runtime_dependencies(settings=settings)
    try:
        assert settings.environment == "production"
        assert isinstance(cache, UpstashCache)
        assert isinstance(service, DashboardService)
        assert isinstance(service._market_data, MarketDataService)
        assert isinstance(service._symbol_search, SymbolSearchService)
        assert service._symbol_search._source is None
        assert isinstance(service._research, DisabledResearchService)
    finally:
        asyncio.run(service.aclose())


def _secure_environment(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    values = {
        "ENVIRONMENT": "production",
        "MARKETDATA_TOKEN": "  runtime-provider-token  ",
        "MARKETDATA_BASE_URL": "https://api.marketdata.app/v1/",
        "HTTP_TIMEOUT_SECONDS": "7.5",
        "MARKETDATA_DAILY_CREDIT_BUDGET": "81",
        "SESSION_SECRET": "a-session-secret-longer-than-32-bytes",
        "APP_ACCESS_KEY_SHA256": "a" * 64,
        "UPSTASH_REDIS_REST_URL": "https://cache-name.upstash.io",
        "UPSTASH_REDIS_REST_TOKEN": "cache-token",
        "ALLOWED_ORIGIN": "https://market.example",
        "ALLOWED_HOSTS": "market.example",
        "SESSION_COOKIE_SECURE": "true",
        **overrides,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_environment_settings_are_unwrapped_only_at_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class RecordingProvider:
        def __init__(
            self,
            token: str,
            *,
            base_url: str,
            timeout_seconds: float,
        ) -> None:
            nonlocal captured
            captured = {
                "token": token,
                "base_url": base_url,
                "timeout_seconds": timeout_seconds,
            }

        async def aclose(self) -> None:
            return None

    _secure_environment(monkeypatch)
    monkeypatch.setattr(runtime, "MarketDataAppProvider", RecordingProvider)

    settings = Settings(_env_file=None)
    _, service, _ = _runtime_dependencies(settings=settings)
    try:
        assert captured == {
            "token": "runtime-provider-token",
            "base_url": "https://api.marketdata.app/v1",
            "timeout_seconds": 7.5,
        }
        assert isinstance(captured["token"], str)
        assert service._market_data._daily_credit_budget == 81
        assert service._symbol_search._source is None
        assert isinstance(service._research, DisabledResearchService)
    finally:
        asyncio.run(service.aclose())


@pytest.mark.parametrize(
    ("override_name", "override_value", "expected_name"),
    [
        ("MARKETDATA_BASE_URL", "https://api.marketdata.app/v2", "MARKETDATA_BASE_URL"),
        ("MARKETDATA_TOKEN", "<your-marketdata-token>", "MARKETDATA_TOKEN"),
        ("MARKETDATA_TOKEN", "example", "MARKETDATA_TOKEN"),
        ("MARKETDATA_TOKEN", "change-me", "MARKETDATA_TOKEN"),
        ("MARKETDATA_TOKEN", "your-api-key", "MARKETDATA_TOKEN"),
    ],
)
def test_invalid_production_provider_configuration_fails_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
    override_name: str,
    override_value: str,
    expected_name: str,
) -> None:
    provider_constructions = 0

    class ForbiddenProvider:
        def __init__(self, *_: object, **__: object) -> None:
            nonlocal provider_constructions
            provider_constructions += 1

    _secure_environment(monkeypatch, **{override_name: override_value})
    monkeypatch.setattr(runtime, "MarketDataAppProvider", ForbiddenProvider)

    with pytest.raises((ValidationError, RuntimeError), match=expected_name):
        _runtime_dependencies(settings=Settings(_env_file=None))

    assert provider_constructions == 0


def test_missing_production_provider_token_fails_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_constructions = 0

    class ForbiddenProvider:
        def __init__(self, *_: object, **__: object) -> None:
            nonlocal provider_constructions
            provider_constructions += 1

    _secure_environment(monkeypatch)
    monkeypatch.delenv("MARKETDATA_TOKEN")
    monkeypatch.setattr(runtime, "MarketDataAppProvider", ForbiddenProvider)

    with pytest.raises(ValidationError, match="MARKETDATA_TOKEN"):
        _runtime_dependencies(settings=Settings(_env_file=None))

    assert provider_constructions == 0


def test_old_marketstack_only_variables_fail_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_constructions = 0

    class ForbiddenProvider:
        def __init__(self, *_: object, **__: object) -> None:
            nonlocal provider_constructions
            provider_constructions += 1

    _secure_environment(monkeypatch)
    monkeypatch.delenv("MARKETDATA_TOKEN")
    monkeypatch.setenv("MARKETSTACK_API_KEY", "old-provider-secret")
    monkeypatch.setattr(runtime, "MarketDataAppProvider", ForbiddenProvider)

    with pytest.raises(ValidationError, match="MARKETDATA_TOKEN") as captured:
        _runtime_dependencies(settings=Settings(_env_file=None))

    assert "old-provider-secret" not in str(captured.value)
    assert provider_constructions == 0


def test_disabled_runtime_never_constructs_research_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions = 0

    def forbidden(_: Settings) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("disabled research must not construct providers")

    settings = Settings(
        environment="production",
        **secure_production_values(),
        _env_file=None,
    )
    monkeypatch.setattr(runtime, "build_research_runtime", forbidden)

    _, service, _ = _runtime_dependencies(settings=settings)
    try:
        assert isinstance(service._research, DisabledResearchService)
        assert constructions == 0
    finally:
        asyncio.run(service.aclose())


def test_enabled_runtime_is_selected_without_startup_network_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = Settings(
        environment="production",
        research_enabled=True,
        **secure_production_values(),
        **complete_research_values(),
        _env_file=None,
    )
    fake_research = object()
    calls: list[Settings] = []

    def build(settings: Settings) -> object:
        calls.append(settings)
        return fake_research

    monkeypatch.setattr(runtime, "build_research_runtime", build)

    _, service, _ = _runtime_dependencies(settings=configured)
    try:
        assert service._research is fake_research
        assert calls == [configured]
    finally:
        asyncio.run(service.aclose())


@pytest.mark.asyncio
async def test_research_builder_passes_exact_approved_values_and_owns_one_shared_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, object]] = {}

    class RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = dict(kwargs)
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    class RecordingEmbedder:
        def __init__(self, **kwargs: object) -> None:
            captured["embedder"] = dict(kwargs)
            self.descriptor = EmbeddingDescriptor(
                provider="openai",
                model="text-embedding-3-small",
                version="openai:text-embedding-3-small:1536:v1",
                dimensions=1536,
            )

    class RecordingGenerator:
        def __init__(self, **kwargs: object) -> None:
            captured["generator"] = dict(kwargs)

    class RecordingVector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["vector"] = {"args": args, **kwargs}

    class RecordingControl:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["control"] = {"args": args, **kwargs}

    monkeypatch.setattr(research_runtime.httpx, "AsyncClient", RecordingClient)
    monkeypatch.setattr(research_runtime, "OpenAIEmbedder", RecordingEmbedder)
    monkeypatch.setattr(research_runtime, "OpenAIAnswerGenerator", RecordingGenerator)
    monkeypatch.setattr(research_runtime, "UpstashVectorStore", RecordingVector)
    monkeypatch.setattr(research_runtime, "RedisResearchControl", RecordingControl)
    settings = Settings(
        environment="test",
        research_enabled=True,
        upstash_redis_rest_url="https://redis-control.upstash.io",
        upstash_redis_rest_token=SecretStr("redis-control-secret"),
        **complete_research_values(),
        _env_file=None,
    )

    built = research_runtime.build_research_runtime(settings)
    assert isinstance(built, ResearchRuntime)
    shared = captured["embedder"]["client"]
    assert shared is captured["generator"]["client"]
    assert shared is captured["vector"]["client"]
    assert shared is captured["control"]["client"]
    assert isinstance(captured["embedder"]["api_key"], SecretStr)
    assert isinstance(captured["generator"]["api_key"], SecretStr)
    assert captured["generator"]["max_output_tokens"] == 700
    assert captured["vector"]["args"] == (
        "https://example-index-us1-vector.upstash.io",
        settings.upstash_vector_rest_token,
    )
    assert captured["vector"]["namespace"] == "sec-filings-v1"
    assert captured["control"]["args"] == (
        "https://redis-control.upstash.io",
        settings.upstash_redis_rest_token,
    )
    assert built._core._chunker is None
    assert built._core._corpus.corpus_version == "v1"
    assert built._core._corpus.chunker_version == "tokens-800-100-v1"
    assert built._core._policy.minimum_score == 0.70
    assert built._core._policy.max_results == 5
    assert built._core._policy.overfetch_factor == 4
    assert built._core._policy.request_timeout_seconds < settings.research_timeout_seconds

    await built.aclose()
    await built.aclose()

    assert isinstance(shared, RecordingClient)
    assert shared.close_calls == 1


@pytest.mark.asyncio
async def test_injected_research_client_is_not_closed_by_runtime() -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(
        environment="test",
        research_enabled=True,
        upstash_redis_rest_url="https://redis-control.upstash.io",
        upstash_redis_rest_token=SecretStr("redis-control-secret"),
        **complete_research_values(),
        _env_file=None,
    )
    built = research_runtime.build_research_runtime(settings, client=client)

    await built.aclose()

    assert not client.is_closed
    assert requests == 0
    await client.aclose()
