import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import api.index as runtime
from api.index import _runtime_dependencies
from app.cache.upstash import UpstashCache
from app.config import Settings
from app.services.market_data import MarketDataService
from tests.unit.test_config import secure_production_values


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
        assert isinstance(service, MarketDataService)
    finally:
        asyncio.run(service.aclose())


def _secure_environment(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    values = {
        "ENVIRONMENT": "production",
        "MARKETSTACK_API_KEY": "  runtime-provider-key  ",
        "MARKETSTACK_BASE_URL": "https://api.marketstack.com/v2/",
        "MARKETSTACK_TIMEOUT_SECONDS": "7.5",
        "SESSION_SECRET": "a-session-secret-longer-than-32-bytes",
        "APP_ACCESS_KEY_SHA256": "a" * 64,
        "UPSTASH_REDIS_REST_URL": "https://cache-name.upstash.io",
        "UPSTASH_REDIS_REST_TOKEN": "cache-token",
        "ALLOWED_ORIGIN": "https://market.example",
        "ALLOWED_HOSTS": "market.example",
        "SESSION_COOKIE_SECURE": "true",
        **overrides,
    }
    monkeypatch.delenv("MARKETSTACK_ACCESS_KEY", raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_environment_settings_are_unwrapped_only_at_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class RecordingProvider:
        def __init__(
            self,
            access_key: str,
            *,
            base_url: str,
            timeout_seconds: float,
        ) -> None:
            nonlocal captured
            captured = {
                "access_key": access_key,
                "base_url": base_url,
                "timeout_seconds": timeout_seconds,
            }

        async def aclose(self) -> None:
            return None

    _secure_environment(monkeypatch)
    monkeypatch.setattr(runtime, "MarketstackProvider", RecordingProvider)

    settings = Settings(_env_file=None)
    _, service, _ = _runtime_dependencies(settings=settings)
    try:
        assert captured == {
            "access_key": "runtime-provider-key",
            "base_url": "https://api.marketstack.com/v2",
            "timeout_seconds": 7.5,
        }
        assert isinstance(captured["access_key"], str)
    finally:
        asyncio.run(service.aclose())


@pytest.mark.parametrize(
    ("override_name", "override_value", "expected_name"),
    [
        ("MARKETSTACK_BASE_URL", "https://api.marketstack.com/v1", "MARKETSTACK_BASE_URL"),
        ("MARKETSTACK_API_KEY", "<your-marketstack-api-key>", "MARKETSTACK_API_KEY"),
        ("MARKETSTACK_API_KEY", "example", "MARKETSTACK_API_KEY"),
        ("MARKETSTACK_API_KEY", "change-me", "MARKETSTACK_API_KEY"),
        ("MARKETSTACK_API_KEY", "your-api-key", "MARKETSTACK_API_KEY"),
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
    monkeypatch.setattr(runtime, "MarketstackProvider", ForbiddenProvider)

    with pytest.raises((ValidationError, RuntimeError), match=expected_name):
        _runtime_dependencies(settings=Settings(_env_file=None))

    assert provider_constructions == 0


def test_missing_production_provider_key_fails_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_constructions = 0

    class ForbiddenProvider:
        def __init__(self, *_: object, **__: object) -> None:
            nonlocal provider_constructions
            provider_constructions += 1

    _secure_environment(monkeypatch)
    monkeypatch.delenv("MARKETSTACK_API_KEY")
    monkeypatch.setattr(runtime, "MarketstackProvider", ForbiddenProvider)

    with pytest.raises(ValidationError, match="MARKETSTACK_API_KEY"):
        _runtime_dependencies(settings=Settings(_env_file=None))

    assert provider_constructions == 0
