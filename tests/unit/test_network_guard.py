from __future__ import annotations

import socket

import pytest


def test_test_harness_blocks_external_dns_but_allows_loopback() -> None:
    with pytest.raises(RuntimeError, match="external network access is disabled"):
        socket.getaddrinfo("api.openai.com", 443)

    assert socket.getaddrinfo("localhost", 0)
