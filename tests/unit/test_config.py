from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings

_MARKETDATA_SECRET = "marketdata-traceback-secret-sentinel"
_UPSTASH_SECRET = "upstash-traceback-secret-sentinel"
_SESSION_SECRET = "session-traceback-secret-sentinel-longer-than-32-bytes"
_SECRET_SENTINELS = (_MARKETDATA_SECRET, _UPSTASH_SECRET, _SESSION_SECRET)


def secure_production_values() -> dict[str, object]:
    return {
        "marketdata_token": "marketdata-production-token",
        "session_secret": "a-session-secret-longer-than-32-bytes",
        "app_access_key_sha256": "a" * 64,
        "upstash_redis_rest_url": "https://cache-name.upstash.io",
        "upstash_redis_rest_token": "cache-token",
        "allowed_origin": "https://market.example",
        "allowed_hosts": "market.example,www.market.example",
        "cookie_secure": True,
    }


def _assert_secret_free_exception(error: BaseException) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        surfaces = (str(current), repr(current))
        assert not any(secret in surface for secret in _SECRET_SENTINELS for surface in surfaces)
        traceback = current.__traceback__
        while traceback is not None:
            frame_locals = repr(traceback.tb_frame.f_locals)
            assert not any(secret in frame_locals for secret in _SECRET_SENTINELS)
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def test_production_settings_fail_closed_when_security_values_are_missing() -> None:
    with pytest.raises(ValidationError, match="MARKETDATA_TOKEN"):
        Settings(environment="production", _env_file=None)


def test_production_settings_accept_complete_secure_environment_contract() -> None:
    settings = Settings(environment="production", **secure_production_values(), _env_file=None)

    assert settings.marketdata_token.get_secret_value() == "marketdata-production-token"
    assert settings.marketdata_base_url == "https://api.marketdata.app/v1"
    assert settings.marketdata_daily_credit_budget == 90
    assert settings.cache_schema_version == "v2"
    assert settings.session_cookie_name == "marketdata_session"
    assert settings.allowed_hosts == ("market.example", "www.market.example")


@pytest.mark.parametrize(("marker", "value"), [("VERCEL", "1"), ("VERCEL_ENV", "preview")])
def test_vercel_markers_force_production_posture(
    monkeypatch: pytest.MonkeyPatch, marker: str, value: str
) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv(marker, value)

    with pytest.raises(ValidationError, match="MARKETDATA_TOKEN"):
        Settings(_env_file=None)


def test_vercel_marker_overrides_misspelled_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "prodcution")
    monkeypatch.setenv("VERCEL", "1")

    settings = Settings(**secure_production_values(), _env_file=None)

    assert settings.environment == "production"
    assert settings.is_deployed


def test_canonical_token_loads_from_environment_and_is_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKETDATA_TOKEN", "  md_live_canonical_123  ")

    settings = Settings(_env_file=None)

    assert isinstance(settings.marketdata_token, SecretStr)
    assert settings.marketdata_token.get_secret_value() == "md_live_canonical_123"


def test_old_marketstack_only_configuration_is_not_an_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("MARKETSTACK_API_KEY", "old-secret-must-not-appear")
    monkeypatch.setenv("MARKETSTACK_ACCESS_KEY", "old-legacy-secret-must-not-appear")
    for name, value in secure_production_values().items():
        if name != "marketdata_token":
            monkeypatch.setenv(name.upper(), str(value))

    with pytest.raises(ValidationError, match="MARKETDATA_TOKEN") as captured:
        Settings(_env_file=None)

    assert "old-secret-must-not-appear" not in str(captured.value)
    assert "old-legacy-secret-must-not-appear" not in str(captured.value)


@pytest.mark.parametrize(
    "invalid_token",
    [
        "",
        "   ",
        "token-with-\n-control",
        "token-with-\x7f-control",
        '"real-looking-token"',
        "'<your-marketdata-token>'",
        "<your-marketdata-token>",
        "your-marketdata-token",
        "your-api-key",
        "example",
        "change-me",
        "replace-with-your-key",
    ],
)
def test_explicit_invalid_tokens_are_rejected_without_disclosure(invalid_token: str) -> None:
    with pytest.raises(ValidationError, match="MARKETDATA_TOKEN is invalid") as captured:
        Settings(marketdata_token=invalid_token, _env_file=None)

    if invalid_token.strip():
        assert invalid_token not in str(captured.value)


@pytest.mark.parametrize("invalid_token", [None, 123, object()])
def test_non_text_tokens_are_rejected(invalid_token: object) -> None:
    with pytest.raises(ValidationError, match="MARKETDATA_TOKEN is invalid"):
        Settings(marketdata_token=invalid_token, _env_file=None)


def test_missing_token_is_allowed_only_in_explicit_local_environment() -> None:
    settings = Settings(environment="test", _env_file=None)

    assert settings.marketdata_token.get_secret_value() == ""
    assert settings.is_local_environment


def test_development_allows_mock_http_v1_endpoint() -> None:
    settings = Settings(
        environment="test",
        marketdata_base_url="http://marketdata.test:8080/v1/",
        _env_file=None,
    )

    assert settings.marketdata_base_url == "http://marketdata.test:8080/v1"


def test_production_normalizes_exact_marketdata_v1_url() -> None:
    settings = Settings(
        environment="production",
        marketdata_base_url="https://api.marketdata.app/v1/",
        **secure_production_values(),
        _env_file=None,
    )

    assert settings.marketdata_base_url == "https://api.marketdata.app/v1"


@pytest.mark.parametrize(
    "marketdata_base_url",
    [
        "https://api.marketdata.app",
        "https://api.marketdata.app/v2",
        "https://api.marketdata.app/v1/stocks",
        "https://api.marketdata.app/v1?format=json",
        "https://api.marketdata.app/v1#docs",
        "http://api.marketdata.app/v1",
        "https://marketdata.app/v1",
        "https://evil.example/v1",
        "https://user:password@api.marketdata.app/v1",
        "https://api.marketdata.app:443/v1",
        "https://api.marketdata.app:8443/v1",
    ],
)
def test_production_requires_exact_marketdata_v1_url(marketdata_base_url: str) -> None:
    with pytest.raises(ValidationError, match="MARKETDATA_BASE_URL"):
        Settings(
            environment="production",
            marketdata_base_url=marketdata_base_url,
            **secure_production_values(),
            _env_file=None,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
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


def test_timeout_and_daily_budget_load_from_canonical_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setenv("MARKETDATA_DAILY_CREDIT_BUDGET", "123")

    settings = Settings(_env_file=None)

    assert settings.http_timeout_seconds == 7.5
    assert settings.marketdata_daily_credit_budget == 123


def test_all_settings_secrets_are_redacted_from_settings_surfaces(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(
        marketdata_token=_MARKETDATA_SECRET,
        upstash_redis_rest_token=_UPSTASH_SECRET,
        session_secret=_SESSION_SECRET,
        _env_file=None,
    )

    exposed_surfaces = (
        repr(settings),
        str(settings),
        repr(settings.model_dump()),
        settings.model_dump_json(),
        caplog.text,
    )
    assert isinstance(settings.marketdata_token, SecretStr)
    assert isinstance(settings.upstash_redis_rest_token, SecretStr)
    assert isinstance(settings.session_secret, SecretStr)
    for sentinel in _SECRET_SENTINELS:
        assert all(sentinel not in surface for surface in exposed_surfaces)


def test_validation_error_discards_secret_exception_graph_and_traceback_locals() -> None:
    with pytest.raises(ValidationError, match="ALLOWED_ORIGIN") as captured:
        Settings(
            environment="production",
            **{
                **secure_production_values(),
                "marketdata_token": _MARKETDATA_SECRET,
                "upstash_redis_rest_token": _UPSTASH_SECRET,
                "session_secret": _SESSION_SECRET,
                "allowed_origin": "http://invalid.example",
            },
            _env_file=None,
        )

    _assert_secret_free_exception(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_invalid_token_error_discards_raw_token_from_exception_graph() -> None:
    with pytest.raises(ValidationError, match="MARKETDATA_TOKEN is invalid") as captured:
        Settings(marketdata_token=f"{_MARKETDATA_SECRET}\n", _env_file=None)

    _assert_secret_free_exception(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_environment_validation_error_discards_token_from_traceback_locals() -> None:
    with (
        patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "MARKETDATA_TOKEN": _MARKETDATA_SECRET,
                "ALLOWED_ORIGIN": "http://invalid.example",
            },
            clear=True,
        ),
        pytest.raises(ValidationError) as captured,
    ):
        Settings(_env_file=None)

    _assert_secret_free_exception(captured.value)


def test_unknown_environment_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="ENVIRONMENT"):
        Settings(environment="prodcution", _env_file=None)
