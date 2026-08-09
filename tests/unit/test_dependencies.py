from types import SimpleNamespace

import pytest
from starlette.requests import Request

from app.dependencies import rate_limit_identity


def _request(*, headers: dict[str, str], client_host: str = "127.0.0.1") -> Request:
    return Request(
        {
            "type": "http",
            "headers": [
                (name.lower().encode("ascii"), value.encode("ascii"))
                for name, value in headers.items()
            ],
            "client": (client_host, 12345),
        }
    )


@pytest.mark.unit
def test_deployed_identity_ignores_spoofable_vercel_forwarded_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VERCEL", raising=False)
    request = _request(
        headers={
            "X-Vercel-Forwarded-For": "203.0.113.7",
            "X-Forwarded-For": "198.51.100.4",
        }
    )

    assert rate_limit_identity(request, SimpleNamespace(deployment_platform="vercel")) == (
        "198.51.100.4"
    )


@pytest.mark.unit
def test_deployed_identity_does_not_fall_back_to_vercel_forwarded_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VERCEL", raising=False)
    request = _request(headers={"X-Vercel-Forwarded-For": "203.0.113.7"})

    assert (
        rate_limit_identity(request, SimpleNamespace(deployment_platform="vercel")) == "127.0.0.1"
    )


@pytest.mark.unit
def test_non_deployed_identity_ignores_forwarded_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VERCEL", raising=False)
    request = _request(headers={"X-Forwarded-For": "198.51.100.4"})
    settings = SimpleNamespace(deployment_platform="local", trust_proxy_headers=True)

    assert rate_limit_identity(request, settings) == "127.0.0.1"


@pytest.mark.unit
def test_deployed_identity_uses_official_forwarded_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VERCEL", raising=False)
    request = _request(headers={"X-Forwarded-For": "2001:0db8:0:0:0:0:0:1"})

    assert rate_limit_identity(request, SimpleNamespace(deployment_platform="deployed")) == (
        "2001:db8::1"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "forwarded_value",
    [
        "not-an-ip",
        "198.51.100.4, 203.0.113.7",
        "198.51.100.4, malformed",
    ],
)
def test_deployed_identity_falls_back_for_malformed_forwarded_header(
    monkeypatch: pytest.MonkeyPatch,
    forwarded_value: str,
) -> None:
    monkeypatch.delenv("VERCEL", raising=False)
    request = _request(headers={"X-Forwarded-For": forwarded_value})

    assert (
        rate_limit_identity(request, SimpleNamespace(deployment_platform="vercel")) == "127.0.0.1"
    )
