from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Callable
from typing import Protocol

from pydantic import SecretStr

from app.config import get_settings
from app.providers.upstash_vector_smoke import VectorSmokeProbe, VectorSmokeResult
from app.research.deadline import RequestDeadline
from app.research.fingerprints import ResearchProviderFingerprints, research_provider_fingerprints

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 3


class _SettingsLike(Protocol):
    upstash_vector_rest_url: str | None
    upstash_vector_rest_token: SecretStr | None
    openai_api_key: SecretStr | None


class _ProbeLike(Protocol):
    async def run(self, *, deadline: RequestDeadline) -> VectorSmokeResult: ...

    async def aclose(self) -> None: ...


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run an isolated Vector smoke probe.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--acknowledge-live-vector-smoke", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def _validated_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 1 <= float(value) <= 120
    ):
        raise ValueError("the vector smoke timeout is invalid")
    return float(value)


def _credentials(settings: _SettingsLike) -> tuple[str, SecretStr, ResearchProviderFingerprints]:
    url = settings.upstash_vector_rest_url
    token = settings.upstash_vector_rest_token
    openai_key = settings.openai_api_key
    if url is None or token is None or openai_key is None:
        raise ValueError("the vector smoke configuration is incomplete")
    fingerprints = research_provider_fingerprints(
        vector_endpoint=url,
        vector_token=token,
        openai_api_key=openai_key,
    )
    return url, token, fingerprints


def _fingerprint_payload(fingerprints: ResearchProviderFingerprints) -> dict[str, object]:
    return {
        "fingerprints": {
            "openai_api_key": fingerprints.openai_api_key,
            "vector_endpoint": fingerprints.vector_endpoint,
            "vector_token": fingerprints.vector_token,
        },
        "openai_project_confirmation": fingerprints.openai_project_confirmation,
    }


async def _close_probe(probe: _ProbeLike | None) -> bool:
    if probe is None:
        return True
    try:
        await probe.aclose()
    except Exception:
        return False
    return True


async def async_main(
    argv: list[str] | None = None,
    *,
    settings_factory: Callable[[], _SettingsLike] = get_settings,
    probe_factory: Callable[..., _ProbeLike] = VectorSmokeProbe,
) -> int:
    args = _parser().parse_args(argv)
    acknowledged = bool(args.acknowledge_live_vector_smoke)
    applied = bool(args.apply)
    if applied != acknowledged:
        print(
            json.dumps(
                {
                    "applied": False,
                    "dry_run": True,
                    "error_category": "authorization",
                    "errors": ["live vector smoke requires both acknowledgements"],
                    "network_calls": 0,
                },
                sort_keys=True,
            )
        )
        return EXIT_CONFIG

    try:
        timeout = _validated_timeout(args.timeout_seconds)
        url, token, fingerprints = _credentials(settings_factory())
    except Exception:
        print(
            json.dumps(
                {
                    "applied": False,
                    "dry_run": True,
                    "error_category": "configuration",
                    "errors": ["vector smoke configuration failure"],
                    "network_calls": 0,
                },
                sort_keys=True,
            )
        )
        return EXIT_CONFIG

    if not applied:
        print(
            json.dumps(
                {
                    "applied": False,
                    "dry_run": True,
                    "network_calls": 0,
                    "planned_point_count": 1,
                    **_fingerprint_payload(fingerprints),
                },
                sort_keys=True,
            )
        )
        return EXIT_SUCCESS

    probe: _ProbeLike | None = None
    result: VectorSmokeResult | None = None
    provider_failed = False
    try:
        probe = probe_factory(url, token)
        result = await probe.run(deadline=RequestDeadline.after(timeout))
    except Exception:
        provider_failed = True
    finally:
        close_succeeded = await _close_probe(probe)
    if provider_failed or not close_succeeded or result is None:
        print(
            json.dumps(
                {
                    "applied": True,
                    "dry_run": False,
                    "error_category": "provider",
                    "errors": ["vector smoke provider failure"],
                    "network_calls": None,
                    **_fingerprint_payload(fingerprints),
                },
                sort_keys=True,
            )
        )
        return EXIT_FAILURE

    payload = {
        "applied": True,
        "cleanup_verified": result.cleanup_verified,
        "dry_run": False,
        "expected_point_count": result.expected_point_count,
        "null_point_count": result.null_point_count,
        "observed_point_count": result.observed_point_count,
        "passed": result.passed,
        "reason": result.reason.value,
        "request_counts": {
            "delete": result.delete_request_count,
            "fetch": result.fetch_request_count,
            "upsert": result.upsert_request_count,
        },
        **_fingerprint_payload(fingerprints),
    }
    print(json.dumps(payload, sort_keys=True))
    return EXIT_SUCCESS if result.passed else EXIT_FAILURE


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["EXIT_CONFIG", "EXIT_FAILURE", "EXIT_SUCCESS", "async_main", "main"]
