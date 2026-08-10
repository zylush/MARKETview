from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Never

from app.config import get_settings
from app.research.control import Reservation
from app.research.deadline import RequestDeadline
from app.research.diagnostic_runtime import (
    DiagnosticPreflight,
    DiagnosticRuntime,
    RetrievalDiagnosticRuntime,
    _Configuration,
    _configuration,
    _SettingsLike,
    build_retrieval_diagnostic_runtime,
)
from app.research.retrieval import (
    SafeHitClassification,
    SafeHitRejectionCounts,
)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 3

DIAGNOSTIC_CASE = "aapl_latest_10k_risks_v1"
_SYMBOL = "AAPL"
_FILING_TYPE = "10-K"
_QUESTION = "What material risks did Apple disclose in its latest Form 10-K?"


@dataclass(frozen=True, slots=True)
class _CallCounts:
    embedding: int = 0
    vector_search: int = 0
    answer_generation: int = 0


_ZERO_CALL_COUNTS = _CallCounts()


class _UsageError(ValueError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise _UsageError("retrieval diagnostic arguments are invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Plan or run the fixed operator retrieval diagnostic.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--acknowledge-live-retrieval-diagnostic", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def _validated_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 1 <= float(value) <= 120
    ):
        raise ValueError("retrieval diagnostic timeout is invalid")
    return float(value)


def _score_range(value: tuple[float, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"maximum": round(value[1], 6), "minimum": round(value[0], 6)}


def _payload(
    *,
    applied: bool,
    passed: bool,
    configuration: _Configuration | None,
    preflight: DiagnosticPreflight | None = None,
    classification: SafeHitClassification | None = None,
    calls: _CallCounts = _ZERO_CALL_COUNTS,
    budget_units_committed: int = 0,
    error_category: str | None = None,
    error_message: str | None = None,
) -> dict[str, object]:
    rejections = (
        classification.rejection_counts if classification is not None else SafeHitRejectionCounts()
    )
    payload: dict[str, object] = {
        "accepted_count": classification.accepted_count if classification is not None else 0,
        "accepted_score_range": _score_range(
            classification.accepted_score_range if classification is not None else None
        ),
        "active_generation_count": (
            preflight.active_generation_count if preflight is not None else None
        ),
        "applied": applied,
        "budget_units_committed": budget_units_committed,
        "call_counts": {
            "answer_generation": calls.answer_generation,
            "embedding": calls.embedding,
            "vector_search": calls.vector_search,
        },
        "candidate_count": classification.candidate_count if classification is not None else 0,
        "configured_minimum_score": (
            configuration.minimum_score if configuration is not None else None
        ),
        "diagnostic_case": DIAGNOSTIC_CASE,
        "dry_run": not applied,
        "expected_point_count": (preflight.expected_point_count if preflight is not None else None),
        "filing_type": _FILING_TYPE,
        "inspection_state": preflight.inspection_state if preflight is not None else None,
        "network_calls": 0 if not applied else None,
        "observed_point_count": (preflight.observed_point_count if preflight is not None else None),
        "passed": passed,
        "pending_cleanup_count": (
            preflight.pending_cleanup_count if preflight is not None else None
        ),
        "raw_score_range": _score_range(
            classification.raw_score_range if classification is not None else None
        ),
        "rejection_counts": rejections.as_dict(),
        "request_counts": {
            "embedding": calls.embedding,
            "generation": calls.answer_generation,
            "search": calls.vector_search,
        },
        "requested_candidate_limit": (
            configuration.requested_candidate_limit if configuration is not None else None
        ),
        "symbol": _SYMBOL,
    }
    if error_category is not None:
        payload["error_category"] = error_category
        payload["errors"] = [error_message or "retrieval diagnostic failed"]
    return payload


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _classification_passes(classification: SafeHitClassification) -> bool:
    rejections = classification.rejection_counts
    return bool(
        classification.accepted_count > 0
        and rejections.metadata_integrity == 0
        and rejections.prompt_injection == 0
    )


async def _release_before_commit(
    runtime: DiagnosticRuntime,
    reservation: Reservation,
    deadline: RequestDeadline,
) -> bool:
    try:
        return await runtime.release(reservation=reservation, deadline=deadline)
    except Exception:
        return False


async def _release_cancelled_reservation(
    runtime: DiagnosticRuntime,
    reservation: Reservation,
) -> None:
    release_task = asyncio.create_task(
        runtime.release(
            reservation=reservation,
            deadline=RequestDeadline.after(0.25),
        )
    )
    try:
        async with asyncio.timeout(0.3):
            await asyncio.shield(release_task)
    except asyncio.CancelledError:
        return
    except Exception:
        return
    finally:
        if not release_task.done():
            release_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await release_task


@dataclass(frozen=True, slots=True)
class _LiveOutcome:
    exit_code: int
    preflight: DiagnosticPreflight | None
    classification: SafeHitClassification | None
    calls: _CallCounts
    budget_units_committed: int
    error_category: str | None = None
    error_message: str | None = None


class _LiveFailureError(Exception):
    def __init__(
        self,
        category: str,
        message: str,
        *,
        exit_code: int = EXIT_FAILURE,
        release_before_commit: bool = False,
        budget_units_committed: int = 0,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.safe_message = message
        self.exit_code = exit_code
        self.release_before_commit = release_before_commit
        self.budget_units_committed = budget_units_committed


async def _run_live(
    settings: _SettingsLike,
    *,
    timeout_seconds: float,
    runtime_factory: Callable[[_SettingsLike], DiagnosticRuntime],
) -> _LiveOutcome:
    runtime: DiagnosticRuntime | None = None
    snapshot: DiagnosticPreflight | None = None
    reservation: Reservation | None = None
    classification: SafeHitClassification | None = None
    calls = _ZERO_CALL_COUNTS
    failure: _LiveFailureError | None = None
    commit_started = False
    try:
        try:
            runtime = runtime_factory(settings)
        except Exception:
            raise _LiveFailureError(
                "configuration",
                "retrieval diagnostic configuration failure",
                exit_code=EXIT_CONFIG,
            ) from None
        deadline = RequestDeadline.after(timeout_seconds)
        try:
            snapshot = await runtime.preflight(
                symbol=_SYMBOL,
                filing_type=_FILING_TYPE,
                deadline=deadline,
            )
        except Exception:
            raise _LiveFailureError(
                "preflight",
                "retrieval diagnostic preflight failure",
            ) from None
        if not isinstance(snapshot, DiagnosticPreflight) or not snapshot.is_exact:
            raise _LiveFailureError(
                "preflight",
                "retrieval diagnostic preflight rejected",
                exit_code=EXIT_CONFIG,
            )
        try:
            authorized = await runtime.authorize(deadline=deadline)
        except Exception:
            raise _LiveFailureError(
                "authorization",
                "retrieval diagnostic authorization failure",
            ) from None
        if not isinstance(authorized, Reservation):
            raise _LiveFailureError(
                "budget",
                "retrieval diagnostic budget rejected",
                exit_code=EXIT_CONFIG,
            )
        reservation = authorized
        try:
            revalidated = await runtime.revalidate(preflight=snapshot, deadline=deadline)
        except Exception:
            raise _LiveFailureError(
                "revalidation",
                "retrieval diagnostic revalidation failure",
                release_before_commit=True,
            ) from None
        if not revalidated:
            raise _LiveFailureError(
                "preflight",
                "retrieval diagnostic safety state changed",
                exit_code=EXIT_CONFIG,
                release_before_commit=True,
            )
        commit_started = True
        try:
            committed = await runtime.commit(reservation=reservation, deadline=deadline)
        except Exception:
            raise _LiveFailureError(
                "commit",
                "retrieval diagnostic commit failure",
                budget_units_committed=1,
            ) from None
        if not committed:
            raise _LiveFailureError(
                "budget",
                "retrieval diagnostic budget commit rejected",
                exit_code=EXIT_CONFIG,
                release_before_commit=True,
            )
        calls = replace(calls, embedding=1)
        try:
            vector = await runtime.embed(text=_QUESTION, deadline=deadline)
        except Exception as error:
            raise _LiveFailureError(
                "timeout" if isinstance(error, TimeoutError) else "embedding",
                "retrieval diagnostic embedding failure",
                budget_units_committed=1,
            ) from None
        calls = replace(calls, vector_search=1)
        try:
            result = await runtime.search(
                symbol=_SYMBOL,
                filing_type=_FILING_TYPE,
                vector=vector,
                preflight=snapshot,
                deadline=deadline,
            )
        except Exception as error:
            raise _LiveFailureError(
                "timeout" if isinstance(error, TimeoutError) else "search",
                "retrieval diagnostic search failure",
                budget_units_committed=1,
            ) from None
        if not isinstance(result, SafeHitClassification):
            raise _LiveFailureError(
                "search",
                "retrieval diagnostic search failure",
                budget_units_committed=1,
            )
        classification = result
        if not _classification_passes(classification):
            raise _LiveFailureError(
                "retrieval",
                "retrieval diagnostic safe evidence gate failed",
                budget_units_committed=1,
            )
    except asyncio.CancelledError:
        if runtime is not None and reservation is not None and not commit_started:
            await _release_cancelled_reservation(runtime, reservation)
        raise
    except _LiveFailureError as error:
        failure = error
        if (
            error.release_before_commit
            and runtime is not None
            and reservation is not None
            and not await _release_before_commit(runtime, reservation, deadline)
        ):
            failure = _LiveFailureError(
                "provider",
                "retrieval diagnostic reservation release failure",
            )
    finally:
        if runtime is not None:
            try:
                await runtime.aclose()
            except Exception:
                failure = _LiveFailureError(
                    "provider",
                    "retrieval diagnostic provider close failure",
                    budget_units_committed=(
                        failure.budget_units_committed if failure is not None else 1
                    ),
                )
    if failure is not None:
        return _LiveOutcome(
            exit_code=failure.exit_code,
            preflight=snapshot,
            classification=classification,
            calls=calls,
            budget_units_committed=failure.budget_units_committed,
            error_category=failure.category,
            error_message=failure.safe_message,
        )
    return _LiveOutcome(
        exit_code=EXIT_SUCCESS,
        preflight=snapshot,
        classification=classification,
        calls=calls,
        budget_units_committed=1,
    )


async def async_main(
    argv: list[str] | None = None,
    *,
    settings_factory: Callable[[], _SettingsLike] = get_settings,
    runtime_factory: Callable[[_SettingsLike], DiagnosticRuntime] = (
        build_retrieval_diagnostic_runtime
    ),
) -> int:
    try:
        args = _parser().parse_args(argv)
        timeout = _validated_timeout(args.timeout_seconds)
    except (TypeError, ValueError, _UsageError):
        _emit(
            _payload(
                applied=False,
                passed=False,
                configuration=None,
                error_category="configuration",
                error_message="retrieval diagnostic configuration failure",
            )
        )
        return EXIT_CONFIG
    applied = bool(args.apply)
    acknowledged = bool(args.acknowledge_live_retrieval_diagnostic)
    if applied != acknowledged:
        _emit(
            _payload(
                applied=False,
                passed=False,
                configuration=None,
                error_category="authorization",
                error_message="live retrieval diagnostic requires both acknowledgements",
            )
        )
        return EXIT_CONFIG

    try:
        settings = settings_factory()
        configuration = _configuration(settings)
    except Exception:
        _emit(
            _payload(
                applied=False,
                passed=False,
                configuration=None,
                error_category="configuration",
                error_message="retrieval diagnostic configuration failure",
            )
        )
        return EXIT_CONFIG
    if not applied:
        _emit(_payload(applied=False, passed=False, configuration=configuration))
        return EXIT_SUCCESS

    outcome = await _run_live(
        settings,
        timeout_seconds=timeout,
        runtime_factory=runtime_factory,
    )
    _emit(
        _payload(
            applied=True,
            passed=(
                outcome.exit_code == EXIT_SUCCESS
                and outcome.preflight is not None
                and outcome.preflight.is_exact
                and outcome.classification is not None
                and _classification_passes(outcome.classification)
            ),
            configuration=configuration,
            preflight=outcome.preflight,
            classification=outcome.classification,
            calls=outcome.calls,
            budget_units_committed=outcome.budget_units_committed,
            error_category=outcome.error_category,
            error_message=outcome.error_message,
        )
    )
    return outcome.exit_code


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DIAGNOSTIC_CASE",
    "EXIT_CONFIG",
    "EXIT_FAILURE",
    "EXIT_SUCCESS",
    "DiagnosticPreflight",
    "RetrievalDiagnosticRuntime",
    "async_main",
    "build_retrieval_diagnostic_runtime",
    "main",
]
