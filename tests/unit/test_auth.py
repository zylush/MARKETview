from __future__ import annotations

import hashlib

import pytest
from itsdangerous import BadSignature, SignatureExpired

from app.auth import SessionSigner, verify_app_key


def test_default_session_signer_uses_market_data_salt() -> None:
    signer = SessionSigner("a-long-test-secret", max_age_seconds=60)

    assert signer.salt == "marketdata-session-v1"


def test_app_key_is_checked_against_its_sha256_digest() -> None:
    digest = hashlib.sha256(b"right-key").hexdigest()

    assert verify_app_key("right-key", digest) is True
    assert verify_app_key("wrong-key", digest) is False


@pytest.mark.parametrize("digest", ["", "not-a-digest", "a" * 63, "g" * 64])
def test_app_key_rejects_an_invalid_configured_digest(digest: str) -> None:
    assert verify_app_key("anything", digest) is False


def test_signed_session_round_trip_and_tamper_detection() -> None:
    signer = SessionSigner("a-long-test-secret", max_age_seconds=60)
    token = signer.dumps("analyst")

    assert signer.loads(token) == "analyst"
    with pytest.raises((BadSignature, SignatureExpired)):
        signer.loads(f"{token}tampered")
