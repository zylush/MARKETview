import pytest
from pydantic import ValidationError

from app.config import Settings


def secure_production_values() -> dict[str, object]:
    return {
        "marketstack_access_key": "provider-key",
        "session_secret": "a-session-secret-longer-than-32-bytes",
        "app_access_key_sha256": "a" * 64,
        "upstash_redis_rest_url": "https://cache-name.upstash.io",
        "upstash_redis_rest_token": "cache-token",
        "allowed_origin": "https://market.example",
        "allowed_hosts": "market.example,www.market.example",
        "cookie_secure": True,
    }


def test_production_settings_fail_closed_when_security_values_are_missing() -> None:
    with pytest.raises(ValidationError, match="production settings are incomplete"):
        Settings(environment="production", _env_file=None)


def test_production_settings_accept_complete_secure_environment_contract() -> None:
    settings = Settings(
        environment="production",
        **secure_production_values(),
        _env_file=None,
    )

    assert settings.marketstack_api_key == "provider-key"
    assert settings.allowed_hosts == ("market.example", "www.market.example")


@pytest.mark.parametrize(("marker", "value"), [("VERCEL", "1"), ("VERCEL_ENV", "preview")])
def test_vercel_markers_force_production_posture_without_environment(
    monkeypatch: pytest.MonkeyPatch, marker: str, value: str
) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv(marker, value)

    with pytest.raises(ValidationError, match="production settings are incomplete"):
        Settings(_env_file=None)


def test_vercel_marker_overrides_misspelled_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "prodcution")
    monkeypatch.setenv("VERCEL", "1")

    settings = Settings(**secure_production_values(), _env_file=None)

    assert settings.environment == "production"
    assert settings.is_deployed


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"marketstack_base_url": "http://api.marketstack.com/v2"}, "MARKETSTACK_BASE_URL"),
        ({"marketstack_base_url": "https://evil.example/v2"}, "MARKETSTACK_BASE_URL"),
        ({"upstash_redis_rest_url": "http://cache-name.upstash.io"}, "UPSTASH_REDIS_REST_URL"),
        ({"upstash_redis_rest_url": "https://cache.example"}, "UPSTASH_REDIS_REST_URL"),
        ({"allowed_origin": "http://market.example"}, "ALLOWED_ORIGIN"),
        ({"allowed_origin": "https://*.market.example"}, "ALLOWED_ORIGIN"),
        ({"allowed_hosts": "https://market.example"}, "ALLOWED_HOSTS"),
        ({"allowed_hosts": "*.market.example"}, "ALLOWED_HOSTS"),
        ({"allowed_hosts": "market.example/path"}, "ALLOWED_HOSTS"),
    ],
)
def test_production_rejects_untrusted_url_and_host_shapes(
    override: dict[str, object], message: str
) -> None:
    values = {**secure_production_values(), **override}
    with pytest.raises(ValidationError, match=message):
        Settings(environment="production", **values, _env_file=None)


def test_development_allows_mock_http_endpoints() -> None:
    settings = Settings(
        environment="test",
        marketstack_base_url="http://marketstack.test/v2",
        upstash_redis_rest_url="http://redis.test",
        allowed_origin="http://localhost:8000",
        allowed_hosts="localhost,testserver",
        _env_file=None,
    )

    assert settings.environment == "test"


def test_unknown_environment_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="ENVIRONMENT"):
        Settings(environment="prodcution", _env_file=None)
