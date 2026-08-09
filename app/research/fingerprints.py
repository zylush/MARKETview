from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import SecretStr

_FINGERPRINT_PREFIX = "mvfp1-"
_FINGERPRINT_HEX_LENGTH = 32
_VECTOR_ENDPOINT_DOMAIN = b"marketview/vector-endpoint/v1\x00"
_VECTOR_TOKEN_DOMAIN = b"marketview/vector-token/v1\x00"
_OPENAI_KEY_DOMAIN = b"marketview/openai-key/v1\x00"


@dataclass(frozen=True, slots=True)
class ResearchProviderFingerprints:
    """Opaque operator comparisons, not provider-account attestations."""

    vector_endpoint: str
    vector_token: str
    openai_api_key: str
    openai_project_confirmation: str = "not_independently_verified"


def _fingerprint(domain: bytes, value: str) -> str:
    material = value.encode("utf-8")
    digest = hashlib.sha256(domain + material).hexdigest()[:_FINGERPRINT_HEX_LENGTH]
    material = b""
    return f"{_FINGERPRINT_PREFIX}{digest}"


def _normalized_vector_endpoint(value: object) -> str:
    if not isinstance(value, str):
        value = None
        raise ValueError("the vector endpoint is invalid")
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
    except ValueError:
        value = None
        normalized = ""
        raise ValueError("the vector endpoint is invalid") from None
    hostname = parsed.hostname
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or not hostname.lower().endswith(".upstash.io")
        or hostname.lower() == "upstash.io"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        value = None
        normalized = ""
        parsed = urlsplit("https://redacted.upstash.io")
        hostname = None
        raise ValueError("the vector endpoint is invalid")
    return f"https://{hostname.lower()}"


def _secret_value(value: object) -> str:
    if not isinstance(value, SecretStr):
        raise ValueError("a provider credential is invalid")
    secret = value.get_secret_value()
    if not secret or secret != secret.strip():
        secret = ""
        raise ValueError("a provider credential is invalid")
    return secret


def research_provider_fingerprints(
    *,
    vector_endpoint: str,
    vector_token: SecretStr,
    openai_api_key: SecretStr,
) -> ResearchProviderFingerprints:
    """Return stable, domain-separated fingerprints without exposing inputs.

    Matching the OpenAI-key fingerprint confirms only that two environments use
    the same key. It does not independently prove which OpenAI project owns it.
    """

    try:
        normalized_endpoint = _normalized_vector_endpoint(vector_endpoint)
        vector_secret = _secret_value(vector_token)
        openai_secret = _secret_value(openai_api_key)
    except ValueError as error:
        vector_endpoint = ""
        vector_token = SecretStr("")
        openai_api_key = SecretStr("")
        error = ValueError(str(error))
        raise error from None
    result = ResearchProviderFingerprints(
        vector_endpoint=_fingerprint(_VECTOR_ENDPOINT_DOMAIN, normalized_endpoint),
        vector_token=_fingerprint(_VECTOR_TOKEN_DOMAIN, vector_secret),
        openai_api_key=_fingerprint(_OPENAI_KEY_DOMAIN, openai_secret),
    )
    vector_secret = ""
    openai_secret = ""
    return result


__all__ = ["ResearchProviderFingerprints", "research_provider_fingerprints"]
