from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import api.index as runtime
from api.index import _runtime_dependencies
from app.cache.upstash import UpstashCache
from app.config import Settings
from app.services.dashboard import DashboardService
from app.services.market_analysis import DisabledMarketAnalysisService, MarketAnalysisService
from app.services.market_data import MarketDataService
from app.services.symbols import SymbolSearchService
from tests.unit.test_config import complete_research_values, secure_production_values


def test_vercel_uses_fastapi_zero_config_routing() -> None:
    config = json.loads(Path("vercel.json").read_text(encoding="utf-8"))

    assert config["functions"]["api/index.py"]["includeFiles"] == "{templates,static}/**/*"
    assert "rewrites" not in config


def test_vercel_runtime_uses_shared_cache_and_disabled_analysis() -> None:
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
        assert isinstance(service._market_analysis, DisabledMarketAnalysisService)
    finally:
        asyncio.run(service.aclose())


def test_enabled_runtime_builds_openai_market_analysis_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class RecordingGenerator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(runtime, "OpenAIMarketAnalysis", RecordingGenerator)
    settings = Settings(
        environment="production",
        research_enabled=True,
        **secure_production_values(),
        **complete_research_values(),
        _env_file=None,
    )

    _, service, _ = _runtime_dependencies(settings=settings)
    try:
        assert isinstance(service._market_analysis, MarketAnalysisService)
        assert captured["api_key"] is settings.openai_api_key
        assert captured["model"] == settings.research_generation_model
        assert captured["max_output_tokens"] == settings.research_generation_max_output_tokens
        assert "embedding" not in repr(captured).lower()
        assert "vector" not in repr(captured).lower()
    finally:
        asyncio.run(service.aclose())


def test_local_runtime_without_marketdata_token_remains_fail_closed() -> None:
    settings = Settings(environment="test", _env_file=None)

    _, service, cache = _runtime_dependencies(settings=settings)

    assert type(service) is object
    assert cache is not None
