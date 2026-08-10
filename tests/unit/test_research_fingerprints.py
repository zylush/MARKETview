from __future__ import annotations

import inspect
import re

import pytest
from pydantic import SecretStr

from app.research.fingerprints import research_provider_fingerprints


def test_provider_fingerprints_are_fixed_length_domain_separated_and_secret_safe() -> None:
    shared_secret = "private-provider-secret-marker"
    fingerprints = research_provider_fingerprints(
        vector_endpoint="https://Research-Vector.Upstash.io/",
        vector_token=SecretStr(shared_secret),
        openai_api_key=SecretStr(shared_secret),
    )

    assert (
        fingerprints.vector_endpoint
        == research_provider_fingerprints(
            vector_endpoint="https://research-vector.upstash.io",
            vector_token=SecretStr(shared_secret),
            openai_api_key=SecretStr(shared_secret),
        ).vector_endpoint
    )
    assert fingerprints.vector_token != fingerprints.openai_api_key
    assert all(
        re.fullmatch(r"mvfp1-[0-9a-f]{32}", value)
        for value in (
            fingerprints.vector_endpoint,
            fingerprints.vector_token,
            fingerprints.openai_api_key,
        )
    )
    assert shared_secret not in repr(fingerprints)
    assert shared_secret not in str(fingerprints)
    assert fingerprints.openai_project_confirmation == "not_independently_verified"


@pytest.mark.parametrize(
    ("endpoint", "vector_token", "openai_key"),
    [
        ("http://research-vector.upstash.io", "vector-secret", "openai-secret"),
        ("https://evil.example", "vector-secret", "openai-secret"),
        ("https://research-vector.upstash.io?token=secret", "vector-secret", "openai-secret"),
        ("https://research-vector.upstash.io", "", "openai-secret"),
        ("https://research-vector.upstash.io", "vector-secret", ""),
    ],
)
def test_provider_fingerprints_fail_closed_without_echoing_invalid_values(
    endpoint: str,
    vector_token: str,
    openai_key: str,
) -> None:
    with pytest.raises(ValueError, match="invalid") as captured:
        research_provider_fingerprints(
            vector_endpoint=endpoint,
            vector_token=SecretStr(vector_token),
            openai_api_key=SecretStr(openai_key),
        )

    rendered = str(captured.value)
    assert endpoint not in rendered
    if vector_token:
        assert vector_token not in rendered
    if openai_key:
        assert openai_key not in rendered


def test_provider_fingerprint_validation_discards_traceback_locals() -> None:
    sentinel = "private-invalid-endpoint-and-token"

    with pytest.raises(ValueError, match="endpoint is invalid") as captured:
        research_provider_fingerprints(
            vector_endpoint=f"https://research-vector.upstash.io?token={sentinel}",
            vector_token=SecretStr(f"{sentinel}\n"),
            openai_api_key=SecretStr(sentinel),
        )

    traceback = captured.value.__traceback__
    rendered = ""
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.replace("\\", "/").endswith(
            "app/research/fingerprints.py"
        ):
            rendered += "".join(
                repr(value)
                for value in traceback.tb_frame.f_locals.values()
                if not inspect.iscoroutine(value)
            )
        traceback = traceback.tb_next
    assert sentinel not in rendered
