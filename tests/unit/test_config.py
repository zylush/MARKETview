import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings

_TRACEBACK_MARKETSTACK_SECRET = "marketstack-traceback-secret-sentinel"
_TRACEBACK_UPSTASH_SECRET = "upstash-traceback-secret-sentinel"
_TRACEBACK_SESSION_SECRET = "session-traceback-secret-sentinel-longer-than-32-bytes"
_TRACEBACK_SECRET_SENTINELS = (
    _TRACEBACK_MARKETSTACK_SECRET,
    _TRACEBACK_UPSTASH_SECRET,
    _TRACEBACK_SESSION_SECRET,
)
_CONFLICT_CANONICAL_SECRET = "canonical-conflict-traceback-secret"
_CONFLICT_LEGACY_SECRET = "legacy-conflict-traceback-secret"
_CONFLICT_SECRET_SENTINELS = (
    _CONFLICT_CANONICAL_SECRET,
    _CONFLICT_LEGACY_SECRET,
)


def conflicting_settings_from_environment() -> Settings:
    with patch.dict(
        os.environ,
        {
            "MARKETSTACK_API_KEY": _CONFLICT_CANONICAL_SECRET,
            "MARKETSTACK_ACCESS_KEY": _CONFLICT_LEGACY_SECRET,
        },
    ):
        return Settings(_env_file=None)


def conflicting_settings(source: str) -> Settings:
    if source == "environment":
        return conflicting_settings_from_environment()
    return Settings(
        marketstack_api_key=_CONFLICT_CANONICAL_SECRET,
        marketstack_access_key=_CONFLICT_LEGACY_SECRET,
        _env_file=None,
    )


def secure_production_values() -> dict[str, object]:
    return {
        "marketstack_api_key": "ms_live_config_test_1234567890",
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

    assert settings.marketstack_api_key.get_secret_value() == "ms_live_config_test_1234567890"
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


def test_production_normalizes_the_exact_marketstack_v2_url() -> None:
    settings = Settings(
        environment="production",
        marketstack_base_url="https://api.marketstack.com/v2/",
        **secure_production_values(),
        _env_file=None,
    )

    assert settings.marketstack_base_url == "https://api.marketstack.com/v2"


@pytest.mark.parametrize(
    "marketstack_base_url",
    [
        "https://api.marketstack.com/v1",
        "https://api.marketstack.com",
        "https://api.marketstack.com/v2/eod",
        "https://api.marketstack.com/v2?format=json",
        "https://api.marketstack.com/v2#docs",
        "http://api.marketstack.com/v2",
        "https://marketstack.com/v2",
        "https://user:password@api.marketstack.com/v2",
        "https://api.marketstack.com:8443/v2",
    ],
)
def test_production_requires_the_exact_marketstack_v2_url(
    marketstack_base_url: str,
) -> None:
    with pytest.raises(ValidationError, match="MARKETSTACK_BASE_URL"):
        Settings(
            environment="production",
            marketstack_base_url=marketstack_base_url,
            **secure_production_values(),
            _env_file=None,
        )


def test_canonical_marketstack_key_loads_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKETSTACK_API_KEY", "  ms_live_canonical_123  ")
    monkeypatch.delenv("MARKETSTACK_ACCESS_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert isinstance(settings.marketstack_api_key, SecretStr)
    assert settings.marketstack_api_key.get_secret_value() == "ms_live_canonical_123"


def test_legacy_marketstack_key_loads_when_canonical_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MARKETSTACK_API_KEY", raising=False)
    monkeypatch.setenv("MARKETSTACK_ACCESS_KEY", "ms_live_legacy_123")

    settings = Settings(_env_file=None)

    assert settings.marketstack_api_key.get_secret_value() == "ms_live_legacy_123"


def test_equal_dual_marketstack_key_definitions_use_the_canonical_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKETSTACK_API_KEY", " ms_live_same_123 ")
    monkeypatch.setenv("MARKETSTACK_ACCESS_KEY", "ms_live_same_123")

    settings = Settings(_env_file=None)

    assert settings.marketstack_api_key.get_secret_value() == "ms_live_same_123"


def test_conflicting_marketstack_key_definitions_fail_without_disclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_sentinel = "canonical-secret-sentinel"
    legacy_sentinel = "legacy-secret-sentinel"
    monkeypatch.setenv("MARKETSTACK_API_KEY", canonical_sentinel)
    monkeypatch.setenv("MARKETSTACK_ACCESS_KEY", legacy_sentinel)

    with pytest.raises(RuntimeError) as captured:
        Settings(_env_file=None)

    message = str(captured.value)
    assert "conflicting Marketstack credential variables" in message
    assert canonical_sentinel not in message
    assert legacy_sentinel not in message


@pytest.mark.parametrize("source", ["constructor", "environment"])
def test_conflicting_marketstack_keys_discard_exception_graph_and_traceback_locals(
    source: str,
) -> None:
    with pytest.raises(
        RuntimeError, match="conflicting Marketstack credential variables"
    ) as captured:
        conflicting_settings(source)

    pending: list[BaseException] = [captured.value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert not any(
            secret in surface
            for secret in _CONFLICT_SECRET_SENTINELS
            for surface in (str(current), repr(current))
        )
        traceback = current.__traceback__
        while traceback is not None:
            frame_locals = repr(traceback.tb_frame.f_locals)
            assert not any(secret in frame_locals for secret in _CONFLICT_SECRET_SENTINELS)
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    "invalid_key",
    [
        "   ",
        "key-with-\n-control",
        '"<your-marketstack-api-key>"',
        "<your-marketstack-api-key>",
        "example",
        "change-me",
        "your-api-key",
    ],
)
def test_explicit_invalid_marketstack_keys_are_rejected_without_disclosure(
    invalid_key: str,
) -> None:
    with pytest.raises(ValidationError) as captured:
        Settings(marketstack_api_key=invalid_key, _env_file=None)

    assert "MARKETSTACK_API_KEY is invalid" in str(captured.value)
    if invalid_key.strip():
        assert invalid_key not in str(captured.value)


def test_all_settings_secrets_are_redacted_from_settings_surfaces(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marketstack_sentinel = "ms_live_redaction_sentinel_123"
    upstash_sentinel = "upstash-redaction-sentinel-123"
    session_sentinel = "session-redaction-sentinel-longer-than-32-bytes"

    settings = Settings(
        marketstack_api_key=marketstack_sentinel,
        upstash_redis_rest_token=upstash_sentinel,
        session_secret=session_sentinel,
        _env_file=None,
    )

    exposed_surfaces = (
        repr(settings),
        str(settings),
        repr(settings.model_dump()),
        settings.model_dump_json(),
        caplog.text,
    )
    assert isinstance(settings.marketstack_api_key, SecretStr)
    assert isinstance(settings.upstash_redis_rest_token, SecretStr)
    assert isinstance(settings.session_secret, SecretStr)
    for sentinel in (marketstack_sentinel, upstash_sentinel, session_sentinel):
        assert all(sentinel not in surface for surface in exposed_surfaces)


def test_production_validation_errors_do_not_serialize_secret_inputs() -> None:
    marketstack_sentinel = "marketstack-validation-sentinel"
    upstash_sentinel = "upstash-validation-sentinel"
    session_sentinel = "session-validation-sentinel-longer-than-32-bytes"
    values = {
        **secure_production_values(),
        "marketstack_api_key": marketstack_sentinel,
        "upstash_redis_rest_token": upstash_sentinel,
        "session_secret": session_sentinel,
        "allowed_origin": "http://invalid.example",
    }

    with pytest.raises(ValidationError, match="ALLOWED_ORIGIN") as captured:
        Settings(environment="production", **values, _env_file=None)

    exposed_surfaces = (
        str(captured.value),
        repr(captured.value.errors()),
        captured.value.json(),
    )
    for sentinel in (marketstack_sentinel, upstash_sentinel, session_sentinel):
        assert all(sentinel not in surface for surface in exposed_surfaces)


def test_sanitized_validation_error_discards_raw_exception_graph_and_traceback_locals() -> None:
    with pytest.raises(ValidationError, match="ALLOWED_ORIGIN") as captured:
        Settings(
            environment="production",
            **{
                **secure_production_values(),
                "marketstack_api_key": _TRACEBACK_MARKETSTACK_SECRET,
                "upstash_redis_rest_token": _TRACEBACK_UPSTASH_SECRET,
                "session_secret": _TRACEBACK_SESSION_SECRET,
                "allowed_origin": "http://invalid.example",
            },
            _env_file=None,
        )

    pending: list[BaseException] = [captured.value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        exception_surfaces = (str(current), repr(current))
        assert not any(
            secret in surface
            for secret in _TRACEBACK_SECRET_SENTINELS
            for surface in exception_surfaces
        )
        traceback = current.__traceback__
        while traceback is not None:
            frame_locals = repr(traceback.tb_frame.f_locals)
            assert not any(secret in frame_locals for secret in _TRACEBACK_SECRET_SENTINELS)
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_unknown_environment_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="ENVIRONMENT"):
        Settings(environment="prodcution", _env_file=None)
