import asyncio
import json
from pathlib import Path

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
