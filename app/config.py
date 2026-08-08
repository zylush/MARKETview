from __future__ import annotations

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

_INVALID_MARKETSTACK_KEY = "__invalid_marketstack_key__"


class _MarketstackCredentialConflictError(RuntimeError):
    pass


class Settings(BaseSettings):
    """Environment-backed application settings; secrets are never represented as plain text."""

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
    marketstack_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="MARKETSTACK_API_KEY",
    )
    legacy_marketstack_access_key: SecretStr | None = Field(
        default=None,
        validation_alias="MARKETSTACK_ACCESS_KEY",
        exclude=True,
        repr=False,
    )
    marketstack_base_url: str = "https://api.marketstack.com/v2"
    marketstack_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        le=60,
        validation_alias=AliasChoices("HTTP_TIMEOUT_SECONDS", "MARKETSTACK_TIMEOUT_SECONDS"),
    )
    upstash_redis_rest_url: str | None = None
    upstash_redis_rest_token: SecretStr | None = None
    provider_monthly_budget: int = Field(
        default=90,
        gt=0,
        validation_alias=AliasChoices("MARKETSTACK_MONTHLY_BUDGET", "PROVIDER_MONTHLY_BUDGET"),
    )
    cache_schema_version: str = "v1"

    session_secret: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    session_cookie_name: str = "marketstack_session"
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
        sanitized_conflict: RuntimeError | None = None
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
        except _MarketstackCredentialConflictError:
            sanitized_conflict = RuntimeError("conflicting Marketstack credential variables")
            values = {}
        if sanitized_error is not None:
            raise sanitized_error from None
        if sanitized_conflict is not None:
            raise sanitized_conflict from None

    @model_validator(mode="before")
    @classmethod
    def derive_deployment_posture(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        vercel = values.get("vercel", values.get("VERCEL"))
        vercel_env = values.get("vercel_env", values.get("VERCEL_ENV"))
        marker = str(vercel or "").strip().lower()
        deployed = marker not in {"", "0", "false", "no", "off"} or bool(
            str(vercel_env or "").strip()
        )
        return {**values, "environment": "production"} if deployed else values

    @model_validator(mode="before")
    @classmethod
    def normalize_marketstack_credentials(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        canonical_keys = ("marketstack_api_key", "MARKETSTACK_API_KEY")
        legacy_keys = (
            "marketstack_access_key",
            "legacy_marketstack_access_key",
            "MARKETSTACK_ACCESS_KEY",
        )
        canonical = cls._first_present(values, canonical_keys)
        legacy = cls._first_present(values, legacy_keys)
        normalized_canonical = cls._normalize_marketstack_key(canonical)
        normalized_legacy = cls._normalize_marketstack_key(legacy)
        if (
            normalized_canonical is not None
            and normalized_legacy is not None
            and normalized_canonical != normalized_legacy
        ):
            values = {}
            canonical = None
            legacy = None
            normalized_canonical = None
            normalized_legacy = None
            raise _MarketstackCredentialConflictError
        selected = normalized_canonical if normalized_canonical is not None else normalized_legacy
        normalized_values = {
            key: value
            for key, value in values.items()
            if key not in {*canonical_keys, *legacy_keys}
        }
        secret_input_keys = {
            "session_secret",
            "SESSION_SECRET",
            "upstash_redis_rest_token",
            "UPSTASH_REDIS_REST_TOKEN",
        }
        protected_values = {
            key: SecretStr(value) if key in secret_input_keys and isinstance(value, str) else value
            for key, value in normalized_values.items()
        }
        if selected is not None:
            return {**protected_values, "MARKETSTACK_API_KEY": selected}
        return protected_values

    @staticmethod
    def _first_present(values: dict[str, Any], keys: tuple[str, ...]) -> Any:
        return next((values[key] for key in keys if key in values), None)

    @staticmethod
    def _normalize_marketstack_key(value: Any) -> SecretStr | None:
        if value is None:
            return None
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        if any(ord(character) < 32 or ord(character) == 127 for character in raw_value):
            return SecretStr(_INVALID_MARKETSTACK_KEY)
        normalized = raw_value.strip()
        placeholder_values = {
            "<your-marketstack-api-key>",
            "your-marketstack-api-key",
            "your-api-key",
            "example",
            "change-me",
            "changeme",
            "replace-me",
            "replace-with-your-key",
        }
        is_quoted = (
            len(normalized) >= 2 and normalized[0] in {"'", '"'} and normalized[-1] == normalized[0]
        )
        if not normalized or is_quoted or normalized.lower() in placeholder_values:
            return SecretStr(_INVALID_MARKETSTACK_KEY)
        return SecretStr(normalized)

    @field_validator("marketstack_api_key")
    @classmethod
    def reject_invalid_marketstack_key(cls, value: SecretStr) -> SecretStr:
        if value.get_secret_value() == _INVALID_MARKETSTACK_KEY:
            raise ValueError("MARKETSTACK_API_KEY is invalid")
        return value

    @field_validator("marketstack_base_url", "upstash_redis_rest_url", mode="before")
    @classmethod
    def strip_trailing_slash(cls, value: object) -> object:
        return value.rstrip("/") if isinstance(value, str) else value

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(host.strip() for host in value.split(",") if host.strip())
        return value

    @model_validator(mode="after")
    def validate_production_security(self) -> Settings:
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
        if self.environment.lower() not in {"production", "prod"}:
            return self
        missing: list[str] = []
        if not self.marketstack_api_key.get_secret_value():
            missing.append("MARKETSTACK_API_KEY")
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
        if self.marketstack_base_url != "https://api.marketstack.com/v2":
            missing.append("MARKETSTACK_BASE_URL")
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

    @staticmethod
    def _valid_https_url(
        value: str | None,
        *,
        exact_host: str | None = None,
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
        normalized_host = host.lower()
        if exact_host is not None and normalized_host != exact_host:
            return False
        if host_suffix is not None and not normalized_host.endswith(host_suffix):
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
    def marketstack_monthly_budget(self) -> int:
        return self.provider_monthly_budget

    @property
    def http_timeout_seconds(self) -> float:
        return self.marketstack_timeout_seconds

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
