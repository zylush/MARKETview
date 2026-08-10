from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings

_MARKETDATA_SECRET = "marketdata-traceback-secret-sentinel"
_UPSTASH_SECRET = "upstash-traceback-secret-sentinel"
_SESSION_SECRET = "session-traceback-secret-sentinel-longer-than-32-bytes"
_OPENAI_SECRET = "openai-traceback-secret-sentinel"
_VECTOR_SECRET = "vector-traceback-secret-sentinel"
_SECRET_SENTINELS = (
    _MARKETDATA_SECRET,
    _UPSTASH_SECRET,
    _SESSION_SECRET,
    _OPENAI_SECRET,
    _VECTOR_SECRET,
)


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


def test_production_settings_fail_closed_when_security_values_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MARKETDATA_TOKEN", raising=False)

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
    monkeypatch.delenv("MARKETDATA_TOKEN", raising=False)
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
    monkeypatch.delenv("MARKETDATA_TOKEN", raising=False)
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


def test_missing_token_is_allowed_only_in_explicit_local_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MARKETDATA_TOKEN", raising=False)

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


def test_symbol_and_research_controls_load_without_external_provider_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "MARKETview admin@example.com")
    monkeypatch.setenv("SYMBOL_INDEX_SCHEMA_VERSION", "v7")
    monkeypatch.setenv("SYMBOL_DIRECTORY_MAX_AGE_SECONDS", "43200")
    monkeypatch.setenv("SYMBOL_SEARCH_RATE_LIMIT", "24")
    monkeypatch.setenv("RESEARCH_ENABLED", "false")
    monkeypatch.setenv("RESEARCH_MAX_QUESTION_CHARS", "480")
    monkeypatch.setenv("RESEARCH_MAX_REQUEST_BYTES", "3072")
    monkeypatch.setenv("RESEARCH_TIMEOUT_SECONDS", "6.5")
    monkeypatch.setenv("RESEARCH_RATE_LIMIT", "7")
    monkeypatch.setenv("RESEARCH_DAILY_GLOBAL_LIMIT", "40")

    settings = Settings(_env_file=None)

    assert settings.sec_user_agent == "MARKETview admin@example.com"
    assert settings.symbol_index_schema_version == "v7"
    assert settings.symbol_directory_max_age_seconds == 43_200
    assert settings.symbol_search_rate_limit == 24
    assert settings.research_enabled is False
    assert settings.research_max_question_chars == 480
    assert settings.research_max_request_bytes == 3072
    assert settings.research_timeout_seconds == 6.5
    assert settings.research_rate_limit == 7
    assert settings.research_daily_global_limit == 40


def test_research_is_disabled_by_default_without_model_or_vector_credentials() -> None:
    settings = Settings(environment="test", _env_file=None)

    assert settings.research_enabled is False
    assert settings.openai_api_key is None
    assert settings.upstash_vector_rest_url is None
    assert settings.upstash_vector_rest_token is None


def complete_research_values() -> dict[str, object]:
    return {
        "openai_api_key": _OPENAI_SECRET,
        "upstash_vector_rest_url": "https://example-index-us1-vector.upstash.io",
        "upstash_vector_rest_token": _VECTOR_SECRET,
        "research_embedding_provider": "openai",
        "research_embedding_model": "text-embedding-3-small",
        "research_embedding_dimensions": 1536,
        "research_generation_provider": "openai",
        "research_generation_model": "gpt-5.6-luna",
        "research_generation_max_output_tokens": 700,
        "research_vector_provider": "upstash",
        "research_vector_namespace": "sec-filings-v1",
        "research_index_schema_version": "v1",
        "research_chunk_tokens": 800,
        "research_chunk_overlap_tokens": 100,
        "research_max_results": 5,
        "research_vector_overfetch": 4,
        "research_minimum_score": 0.70,
    }


def test_complete_enabled_research_configuration_is_typed_and_secret_safe() -> None:
    settings = Settings(
        environment="test",
        research_enabled=True,
        **complete_research_values(),
        _env_file=None,
    )

    assert isinstance(settings.openai_api_key, SecretStr)
    assert isinstance(settings.upstash_vector_rest_token, SecretStr)
    assert settings.openai_api_key.get_secret_value() == _OPENAI_SECRET
    assert settings.upstash_vector_rest_token.get_secret_value() == _VECTOR_SECRET
    assert settings.research_embedding_model == "text-embedding-3-small"
    assert settings.research_embedding_dimensions == 1536
    assert settings.research_generation_model == "gpt-5.6-luna"
    assert settings.research_vector_namespace == "sec-filings-v1"
    surfaces = (
        repr(settings),
        str(settings),
        repr(settings.model_dump()),
        settings.model_dump_json(),
    )
    assert all(secret not in surface for secret in _SECRET_SENTINELS for surface in surfaces)


@pytest.mark.parametrize(
    "missing_name",
    ["openai_api_key", "upstash_vector_rest_url", "upstash_vector_rest_token"],
)
def test_enabled_research_requires_complete_provider_configuration(missing_name: str) -> None:
    with pytest.raises(ValidationError, match="research configuration is incomplete") as captured:
        Settings(
            environment="test",
            research_enabled=True,
            **{
                name: value
                for name, value in complete_research_values().items()
                if name != missing_name
            },
            _env_file=None,
        )

    _assert_secret_free_exception(captured.value)


def test_partial_research_credentials_fail_closed_even_while_disabled() -> None:
    with pytest.raises(ValidationError, match="research configuration is incomplete") as captured:
        Settings(
            environment="test",
            openai_api_key=_OPENAI_SECRET,
            _env_file=None,
        )

    _assert_secret_free_exception(captured.value)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"research_embedding_provider": "other"}, "RESEARCH_EMBEDDING_PROVIDER"),
        ({"research_embedding_model": "other"}, "RESEARCH_EMBEDDING_MODEL"),
        ({"research_embedding_dimensions": 3}, "RESEARCH_EMBEDDING_DIMENSIONS"),
        ({"research_generation_provider": "other"}, "RESEARCH_GENERATION_PROVIDER"),
        ({"research_generation_model": "other"}, "RESEARCH_GENERATION_MODEL"),
        ({"research_vector_provider": "other"}, "RESEARCH_VECTOR_PROVIDER"),
        ({"research_vector_namespace": "../unsafe"}, "RESEARCH_VECTOR_NAMESPACE"),
        (
            {"upstash_vector_rest_url": "https://vector.example/api"},
            "UPSTASH_VECTOR_REST_URL",
        ),
    ],
)
def test_enabled_research_rejects_unsupported_or_unsafe_contract_values(
    override: dict[str, object], message: str
) -> None:
    values = {**complete_research_values(), **override}

    with pytest.raises(ValidationError, match=message):
        Settings(
            environment="test",
            research_enabled=True,
            **values,
            _env_file=None,
        )


def test_research_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="RESEARCH_CHUNK_OVERLAP_TOKENS"):
        Settings(
            environment="test",
            research_enabled=True,
            **{
                **complete_research_values(),
                "research_chunk_tokens": 100,
                "research_chunk_overlap_tokens": 100,
            },
            _env_file=None,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"research_generation_max_output_tokens": 2001}, "RESEARCH_GENERATION_MAX_OUTPUT_TOKENS"),
        (
            {"research_max_results": 5, "research_vector_overfetch": 5},
            "RESEARCH_MAX_RESULTS",
        ),
        ({"research_vector_namespace": "research-v1"}, "RESEARCH_VECTOR_NAMESPACE"),
    ],
)
def test_enabled_research_rejects_values_the_runtime_adapters_cannot_honor(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings(
            environment="test",
            research_enabled=True,
            **{**complete_research_values(), **override},
            _env_file=None,
        )


@pytest.mark.parametrize("research_enabled", [False, True])
def test_configured_research_caps_results_at_generator_evidence_limit(
    research_enabled: bool,
) -> None:
    with pytest.raises(ValidationError, match="RESEARCH_MAX_RESULTS must not exceed 8"):
        Settings(
            environment="test",
            research_enabled=research_enabled,
            **{
                **complete_research_values(),
                "research_max_results": 9,
                "research_vector_overfetch": 2,
            },
            _env_file=None,
        )


def test_configured_research_allows_eight_results_with_safe_overfetch_product() -> None:
    settings = Settings(
        environment="test",
        research_enabled=True,
        **{
            **complete_research_values(),
            "research_max_results": 8,
            "research_vector_overfetch": 2,
        },
        _env_file=None,
    )

    assert settings.research_max_results == 8
    assert settings.research_max_results * settings.research_vector_overfetch == 16


def test_disabled_unconfigured_research_retains_dormant_max_results_range() -> None:
    settings = Settings(
        environment="test",
        research_max_results=20,
        research_vector_overfetch=1,
        _env_file=None,
    )

    assert settings.research_enabled is False
    assert settings.research_max_results == 20


def test_research_secret_validation_discards_exception_graph_and_traceback_locals() -> None:
    with pytest.raises(ValidationError) as captured:
        Settings(
            environment="test",
            research_enabled=True,
            **{
                **complete_research_values(),
                "openai_api_key": f"{_OPENAI_SECRET}\n",
                "upstash_vector_rest_token": f"{_VECTOR_SECRET}\n",
            },
            _env_file=None,
        )

    _assert_secret_free_exception(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    "user_agent",
    ["MarketView", "MarketView admin@localhost", "MarketView admin@example.com\r\nInjected: 1"],
)
def test_invalid_sec_user_agent_is_rejected_before_refresh(user_agent: str) -> None:
    with pytest.raises(ValidationError, match="SEC_USER_AGENT"):
        Settings(sec_user_agent=user_agent, _env_file=None)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RESEARCH_MAX_QUESTION_CHARS", "501"),
        ("RESEARCH_MAX_REQUEST_BYTES", "1023"),
        ("RESEARCH_MAX_REQUEST_BYTES", "65537"),
    ],
)
def test_research_request_limits_reject_misleading_or_unsafe_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


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
