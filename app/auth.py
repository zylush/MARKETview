from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from itsdangerous import URLSafeTimedSerializer


def verify_app_key(candidate: str | None, expected_sha256: str | None) -> bool:
    """Verify a raw application key without retaining it or comparing it directly."""
    if not candidate or not expected_sha256 or len(expected_sha256) != 64:
        return False
    try:
        bytes.fromhex(expected_sha256)
    except ValueError:
        return False
    actual = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    return hmac.compare_digest(actual, expected_sha256.lower())


@dataclass(frozen=True, slots=True)
class SessionSigner:
    secret: str
    max_age_seconds: int
    salt: str = "marketstack-session-v1"

    def _serializer(self) -> URLSafeTimedSerializer:
        return URLSafeTimedSerializer(self.secret, salt=self.salt)

    def dumps(self, subject: str) -> str:
        token = self._serializer().dumps({"sub": subject, "version": 1})
        if not isinstance(token, str):
            raise TypeError("session serializer returned an invalid token")
        return token

    def loads(self, token: str) -> str:
        payload = self._serializer().loads(token, max_age=self.max_age_seconds)
        if not isinstance(payload, dict):
            raise ValueError("invalid session payload")
        subject = payload.get("sub")
        if not isinstance(subject, str):
            raise ValueError("invalid session payload")
        return subject
