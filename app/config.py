from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings; secrets are never represented as plain text."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    environment: str = "development"
    vercel: str | None = Field(default=None, validation_alias="VERCEL")
    vercel_env: str | None = Field(default=None, validation_alias="VERCEL_ENV")
    marketstack_access_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("MARKETSTACK_API_KEY", "MARKETSTACK_ACCESS_KEY"),
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

    session_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
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
        if not self.marketstack_api_key:
            missing.append("MARKETSTACK_API_KEY")
        if "session_secret" not in self.model_fields_set or len(self.session_secret) < 32:
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
            self.marketstack_base_url, exact_host="api.marketstack.com", root_only=False
        ):
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
    def marketstack_api_key(self) -> str:
        return self.marketstack_access_key.get_secret_value()

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
