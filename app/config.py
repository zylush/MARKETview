from __future__ import annotations

import re
import secrets
from functools import lru_cache
from typing import Annotated, Any, cast
from urllib.parse import urlsplit

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_INVALID_MARKETDATA_TOKEN = "__invalid_marketdata_token__"  # noqa: S105
_INVALID_RESEARCH_SECRET = "__invalid_research_secret__"  # noqa: S105
_PRODUCTION_MARKETDATA_BASE_URL = "https://api.marketdata.app/v1"
_SEC_CONTACT_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_TOKEN_PLACEHOLDERS = frozenset(
    {
        "<your-marketdata-token>",
        "your-marketdata-token",
        "your-api-key",
        "example",
        "change-me",
        "changeme",
        "replace-me",
        "replace-with-your-key",
    }
)


def _normalized_token_secret(value: object) -> SecretStr:
    if not isinstance(value, (str, SecretStr)):
        return SecretStr(_INVALID_MARKETDATA_TOKEN)
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    contains_control = any(ord(character) < 32 or ord(character) == 127 for character in raw)
    normalized = raw.strip()
    is_quoted = (
        len(normalized) >= 2 and normalized[0] in {"'", '"'} and normalized[-1] == normalized[0]
    )
    placeholder = normalized.lower()
    if (
        not normalized
        or contains_control
        or is_quoted
        or placeholder in _TOKEN_PLACEHOLDERS
        or (placeholder.startswith("<") and placeholder.endswith(">"))
    ):
        return SecretStr(_INVALID_MARKETDATA_TOKEN)
    return SecretStr(normalized)


def _normalized_research_secret(value: object) -> SecretStr:
    if not isinstance(value, (str, SecretStr)):
        return SecretStr(_INVALID_RESEARCH_SECRET)
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    normalized = raw.strip()
    placeholder = normalized.lower()
    if (
        not normalized
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
        or (
            len(normalized) >= 2 and normalized[0] in {"'", '"'} and normalized[-1] == normalized[0]
        )
        or placeholder in _TOKEN_PLACEHOLDERS
        or (placeholder.startswith("<") and placeholder.endswith(">"))
    ):
        return SecretStr(_INVALID_RESEARCH_SECRET)
    return SecretStr(normalized)


class Settings(BaseSettings):
    """Environment-backed application settings with secret-safe validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    environment: str = "development"
    vercel: str | None = Field(default=None, validation_alias="VERCEL")
    vercel_env: str | None = Field(default=None, validation_alias="VERCEL_ENV")
    marketdata_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="MARKETDATA_TOKEN",
    )
    marketdata_base_url: str = Field(
        default=_PRODUCTION_MARKETDATA_BASE_URL,
        validation_alias="MARKETDATA_BASE_URL",
    )
    http_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        le=60,
        validation_alias="HTTP_TIMEOUT_SECONDS",
    )
    marketdata_daily_credit_budget: int = Field(
        default=90,
        gt=0,
        validation_alias="MARKETDATA_DAILY_CREDIT_BUDGET",
    )
    cache_schema_version: str = "v2"

    sec_user_agent: str = Field(default="", validation_alias="SEC_USER_AGENT", max_length=200)
    symbol_index_schema_version: str = Field(
        default="v1",
        validation_alias="SYMBOL_INDEX_SCHEMA_VERSION",
        pattern=r"^v[1-9][0-9]{0,5}$",
    )
    symbol_directory_max_age_seconds: int = Field(
        default=24 * 3600,
        gt=0,
        le=30 * 24 * 3600,
        validation_alias="SYMBOL_DIRECTORY_MAX_AGE_SECONDS",
    )
    symbol_search_rate_limit: int = Field(
        default=30,
        gt=0,
        validation_alias="SYMBOL_SEARCH_RATE_LIMIT",
    )
    research_enabled: bool = Field(default=False, validation_alias="RESEARCH_ENABLED")
    research_max_question_chars: int = Field(
        default=500,
        ge=50,
        le=500,
        validation_alias="RESEARCH_MAX_QUESTION_CHARS",
    )
    research_max_request_bytes: int = Field(
        default=4096,
        ge=1024,
        le=65536,
        validation_alias="RESEARCH_MAX_REQUEST_BYTES",
    )
    research_timeout_seconds: float = Field(
        default=8.0,
        gt=0,
        le=10,
        validation_alias="RESEARCH_TIMEOUT_SECONDS",
    )
    research_rate_limit: int = Field(
        default=10,
        gt=0,
        validation_alias="RESEARCH_RATE_LIMIT",
    )
    research_daily_global_limit: int = Field(
        default=100,
        gt=0,
        validation_alias="RESEARCH_DAILY_GLOBAL_LIMIT",
    )
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    upstash_vector_rest_url: str | None = Field(
        default=None,
        validation_alias="UPSTASH_VECTOR_REST_URL",
    )
    upstash_vector_rest_token: SecretStr | None = Field(
        default=None,
        validation_alias="UPSTASH_VECTOR_REST_TOKEN",
    )
    research_embedding_provider: str = Field(
        default="openai",
        validation_alias="RESEARCH_EMBEDDING_PROVIDER",
    )
    research_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="RESEARCH_EMBEDDING_MODEL",
    )
    research_embedding_dimensions: int = Field(
        default=1536,
        ge=1,
        le=65_536,
        validation_alias="RESEARCH_EMBEDDING_DIMENSIONS",
    )
    research_generation_provider: str = Field(
        default="openai",
        validation_alias="RESEARCH_GENERATION_PROVIDER",
    )
    research_generation_model: str = Field(
        default="gpt-5.6-luna",
        validation_alias="RESEARCH_GENERATION_MODEL",
    )
    research_generation_max_output_tokens: int = Field(
        default=700,
        ge=128,
        le=4096,
        validation_alias="RESEARCH_GENERATION_MAX_OUTPUT_TOKENS",
    )
    research_vector_provider: str = Field(
        default="upstash",
        validation_alias="RESEARCH_VECTOR_PROVIDER",
    )
    research_vector_namespace: str = Field(
        default="sec-filings-v1",
        validation_alias="RESEARCH_VECTOR_NAMESPACE",
    )
    research_index_schema_version: str = Field(
        default="v1",
        pattern=r"^v[1-9][0-9]{0,5}$",
        validation_alias="RESEARCH_INDEX_SCHEMA_VERSION",
    )
    research_chunk_tokens: int = Field(
        default=800,
        ge=100,
        le=4000,
        validation_alias="RESEARCH_CHUNK_TOKENS",
    )
    research_chunk_overlap_tokens: int = Field(
        default=100,
        ge=0,
        le=1000,
        validation_alias="RESEARCH_CHUNK_OVERLAP_TOKENS",
    )
    research_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        validation_alias="RESEARCH_MAX_RESULTS",
    )
    research_vector_overfetch: int = Field(
        default=4,
        ge=1,
        le=10,
        validation_alias="RESEARCH_VECTOR_OVERFETCH",
    )
    research_minimum_score: float = Field(
        default=0.70,
        ge=0,
        le=1,
        validation_alias="RESEARCH_MINIMUM_SCORE",
    )

    upstash_redis_rest_url: str | None = None
    upstash_redis_rest_token: SecretStr | None = None
    session_secret: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    session_cookie_name: str = "marketdata_session"
    session_max_age_seconds: int = Field(default=3600, gt=0)
    cookie_secure: bool = Field(
        default=False,
        validation_alias=AliasChoices("SESSION_COOKIE_SECURE", "COOKIE_SECURE"),
    )
    app_access_key_sha256: str = ""
    allowed_origin: str = "http://localhost:8000"
    allowed_hosts: Annotated[tuple[str, ...], NoDecode] = (
        "localhost",
        "127.0.0.1",
        "testserver",
    )
    login_rate_limit: int = Field(default=5, gt=0)
    api_rate_limit: int = Field(default=60, gt=0)
    rate_limit_window_seconds: int = Field(default=60, gt=0)

    def __init__(self, **values: Any) -> None:
        sanitized_error: ValidationError | None = None
        try:
            super().__init__(**values)
        except ValidationError as error:
            sanitized_error = ValidationError.from_exception_data(
                self.__class__.__name__,
                cast(
                    list[InitErrorDetails],
                    error.errors(include_url=False, include_input=False),
                ),
                hide_input=True,
            )
            values = {}
        if sanitized_error is not None:
            raise sanitized_error from None

    @model_validator(mode="before")
    @classmethod
    def protect_secret_inputs_and_derive_posture(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        token_keys = {"marketdata_token", "MARKETDATA_TOKEN"}
        other_secret_keys = {
            "session_secret",
            "SESSION_SECRET",
            "upstash_redis_rest_token",
            "UPSTASH_REDIS_REST_TOKEN",
        }
        research_secret_keys = {
            "openai_api_key",
            "OPENAI_API_KEY",
            "upstash_vector_rest_token",
            "UPSTASH_VECTOR_REST_TOKEN",
        }
        protected = {
            key: _normalized_token_secret(value)
            if key in token_keys
            else _normalized_research_secret(value)
            if key in research_secret_keys
            else SecretStr(value)
            if key in other_secret_keys and isinstance(value, str)
            else value
            for key, value in values.items()
        }
        vercel = protected.get("vercel", protected.get("VERCEL"))
        vercel_env = protected.get("vercel_env", protected.get("VERCEL_ENV"))
        marker = str(vercel or "").strip().lower()
        deployed = marker not in {"", "0", "false", "no", "off"} or bool(
            str(vercel_env or "").strip()
        )
        return {**protected, "environment": "production"} if deployed else protected

    @field_validator("marketdata_token")
    @classmethod
    def reject_invalid_marketdata_token(cls, value: SecretStr) -> SecretStr:
        if value.get_secret_value() == _INVALID_MARKETDATA_TOKEN:
            raise ValueError("MARKETDATA_TOKEN is invalid")
        return value

    @field_validator("openai_api_key", "upstash_vector_rest_token")
    @classmethod
    def reject_invalid_research_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and value.get_secret_value() == _INVALID_RESEARCH_SECRET:
            raise ValueError("research provider credential is invalid")
        return value

    @field_validator("marketdata_base_url", mode="before")
    @classmethod
    def normalize_marketdata_base_url(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value[:-1] if value.endswith("/") else value
        try:
            parsed = urlsplit(normalized)
            port = parsed.port
        except ValueError as error:
            raise ValueError("MARKETDATA_BASE_URL must target a v1 API root") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != "/v1"
            or parsed.query
            or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError("MARKETDATA_BASE_URL must target a v1 API root")
        return normalized

    @field_validator("upstash_redis_rest_url", mode="before")
    @classmethod
    def strip_trailing_slash(cls, value: object) -> object:
        return value.rstrip("/") if isinstance(value, str) else value

    @field_validator("upstash_vector_rest_url", mode="before")
    @classmethod
    def normalize_upstash_vector_url(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise ValueError("UPSTASH_VECTOR_REST_URL must be an Upstash HTTPS root")
        normalized = value.strip().rstrip("/")
        try:
            parsed = urlsplit(normalized)
            port = parsed.port
        except ValueError:
            raise ValueError("UPSTASH_VECTOR_REST_URL must be an Upstash HTTPS root") from None
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.lower().endswith(".upstash.io")
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("UPSTASH_VECTOR_REST_URL must be an Upstash HTTPS root")
        return normalized

    @field_validator("research_vector_namespace")
    @classmethod
    def validate_research_vector_namespace(cls, value: str) -> str:
        if re.fullmatch(r"sec-filings-v[1-9][0-9]{0,5}", value) is None:
            raise ValueError("RESEARCH_VECTOR_NAMESPACE is invalid")
        return value

    @field_validator("sec_user_agent", mode="before")
    @classmethod
    def validate_sec_user_agent(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if normalized and (
            len(normalized) < 8
            or _SEC_CONTACT_EMAIL.search(normalized) is None
            or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
        ):
            raise ValueError("SEC_USER_AGENT must identify the application and contact email")
        return normalized

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(host.strip() for host in value.split(",") if host.strip())
        return value

    @model_validator(mode="after")
    def validate_security_posture(self) -> Settings:
        allowed_environments = {
            "local",
            "development",
            "dev",
            "test",
            "testing",
            "production",
            "prod",
        }
        if self.environment.lower() not in allowed_environments:
            raise ValueError("ENVIRONMENT must be an explicit supported value")
        self._validate_research_configuration()
        if not self.is_deployed:
            return self

        missing: list[str] = []
        if not self.marketdata_token.get_secret_value():
            missing.append("MARKETDATA_TOKEN")
        if self.marketdata_base_url != _PRODUCTION_MARKETDATA_BASE_URL:
            missing.append("MARKETDATA_BASE_URL")
        if (
            "session_secret" not in self.model_fields_set
            or len(self.session_secret.get_secret_value()) < 32
        ):
            missing.append("SESSION_SECRET")
        if len(self.app_access_key_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.app_access_key_sha256
        ):
            missing.append("APP_ACCESS_KEY_SHA256")
        if not self.upstash_redis_rest_url:
            missing.append("UPSTASH_REDIS_REST_URL")
        if (
            not self.upstash_redis_rest_token
            or not self.upstash_redis_rest_token.get_secret_value()
        ):
            missing.append("UPSTASH_REDIS_REST_TOKEN")
        if not self.allowed_origin:
            missing.append("ALLOWED_ORIGIN")
        if not self.allowed_hosts:
            missing.append("ALLOWED_HOSTS")
        if not self.cookie_secure:
            missing.append("SESSION_COOKIE_SECURE=true")
        if not self._valid_https_url(
            self.upstash_redis_rest_url, host_suffix=".upstash.io", root_only=True
        ):
            missing.append("UPSTASH_REDIS_REST_URL")
        if not self._valid_https_url(self.allowed_origin, root_only=True):
            missing.append("ALLOWED_ORIGIN")
        if any(not self._valid_host(host) for host in self.allowed_hosts):
            missing.append("ALLOWED_HOSTS")
        if missing:
            raise ValueError(
                "production settings are incomplete or insecure: " + ", ".join(missing)
            )
        return self

    def _validate_research_configuration(self) -> None:
        credentials = (
            self.openai_api_key,
            self.upstash_vector_rest_url,
            self.upstash_vector_rest_token,
        )
        configured_count = sum(value is not None for value in credentials)
        if self.research_enabled or configured_count:
            missing = (
                "OPENAI_API_KEY"
                if self.openai_api_key is None
                else "UPSTASH_VECTOR_REST_URL"
                if self.upstash_vector_rest_url is None
                else "UPSTASH_VECTOR_REST_TOKEN"
                if self.upstash_vector_rest_token is None
                else None
            )
            if missing is not None:
                raise ValueError(f"research configuration is incomplete: {missing}")
        expected = (
            (self.research_embedding_provider, "openai", "RESEARCH_EMBEDDING_PROVIDER"),
            (
                self.research_embedding_model,
                "text-embedding-3-small",
                "RESEARCH_EMBEDDING_MODEL",
            ),
            (self.research_embedding_dimensions, 1536, "RESEARCH_EMBEDDING_DIMENSIONS"),
            (self.research_generation_provider, "openai", "RESEARCH_GENERATION_PROVIDER"),
            (self.research_generation_model, "gpt-5.6-luna", "RESEARCH_GENERATION_MODEL"),
            (self.research_vector_provider, "upstash", "RESEARCH_VECTOR_PROVIDER"),
        )
        if (self.research_enabled or configured_count) and any(
            actual != required for actual, required, _ in expected
        ):
            invalid = next(name for actual, required, name in expected if actual != required)
            raise ValueError(f"{invalid} is not supported by the configured research stack")
        if self.research_chunk_overlap_tokens >= self.research_chunk_tokens:
            raise ValueError(
                "RESEARCH_CHUNK_OVERLAP_TOKENS must be smaller than RESEARCH_CHUNK_TOKENS"
            )
        if self.research_generation_max_output_tokens > 2000:
            raise ValueError("RESEARCH_GENERATION_MAX_OUTPUT_TOKENS must not exceed 2000")
        if (self.research_enabled or configured_count) and self.research_max_results > 8:
            raise ValueError("RESEARCH_MAX_RESULTS must not exceed 8")
        if self.research_max_results * self.research_vector_overfetch > 20:
            raise ValueError(
                "RESEARCH_MAX_RESULTS multiplied by RESEARCH_VECTOR_OVERFETCH must not exceed 20"
            )

    @staticmethod
    def _valid_https_url(
        value: str | None,
        *,
        host_suffix: str | None = None,
        root_only: bool,
    ) -> bool:
        if not value or "*" in value:
            return False
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
            port = parsed.port
        except ValueError:
            return False
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or port not in (None, 443)
        ):
            return False
        if host_suffix is not None and not host.lower().endswith(host_suffix):
            return False
        return not root_only or parsed.path in {"", "/"}

    @staticmethod
    def _valid_host(host: str) -> bool:
        if not host or host != host.strip() or len(host) > 253:
            return False
        if any(character in host for character in ("*", "/", ":", "?", "#", "@")):
            return False
        labels = host.rstrip(".").split(".")
        return all(
            label
            and len(label) <= 63
            and label[0].isalnum()
            and label[-1].isalnum()
            and all(character.isalnum() or character == "-" for character in label)
            for label in labels
        )

    @property
    def session_cookie_secure(self) -> bool:
        return self.cookie_secure

    @property
    def is_deployed(self) -> bool:
        return self.environment.lower() in {"production", "prod"}

    @property
    def is_local_environment(self) -> bool:
        return not self.is_deployed and self.environment.lower() in {
            "local",
            "development",
            "dev",
            "test",
            "testing",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
