from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import date
from typing import Protocol, cast

from app.config import Settings, get_settings
from app.research.control import IngestionCheckpoint
from app.research.deadline import RequestDeadline
from app.research.ingestion import (
    IngestionCheckpointStore,
    ResearchIngestionRequest,
)

EXIT_SUCCESS = 0
EXIT_UNEXPECTED = 1
EXIT_PARTIAL = 2
EXIT_CONFIG = 3
EXIT_TIMEOUT = 4
EXIT_LOCK_RECOVERY = 5
EXIT_PROVIDER = 6


class RunnerLike(Protocol):
    async def run(
        self,
        request: ResearchIngestionRequest,
        *,
        deadline: RequestDeadline,
    ) -> object: ...


async def _close_owned_resources(resources: tuple[object, ...]) -> None:
    first_error: Exception | None = None
    for resource in reversed(resources):
        closer = getattr(resource, "aclose", None)
        if closer is None:
            continue
        try:
            await cast(Awaitable[None], closer())
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


@dataclass(frozen=True, slots=True)
class OwnedResearchIngestionRunner:
    """Runner plus the providers exclusively owned by the CLI invocation."""

    runner: RunnerLike
    owned_resources: tuple[object, ...]

    async def run(
        self,
        request: ResearchIngestionRequest,
        *,
        deadline: RequestDeadline,
    ) -> object:
        return await self.runner.run(request, deadline=deadline)

    async def aclose(self) -> None:
        await _close_owned_resources(self.owned_resources)


class ResultLike(Protocol):
    dry_run: bool
    errors: tuple[str, ...]
    failed_count: int
    inserted_count: int
    opaque_job_id: str
    planned_count: int
    processed_count: int
    removed_count: int
    skipped_count: int


class ControlCheckpointBackend(Protocol):
    async def load_checkpoint(
        self,
        *,
        job_digest: str,
        deadline: RequestDeadline,
    ) -> IngestionCheckpoint | None: ...

    async def save_checkpoint(
        self,
        *,
        checkpoint: IngestionCheckpoint,
        expected_previous: IngestionCheckpoint | None,
        deadline: RequestDeadline,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 1 <= self.timeout_seconds <= 900
        ):
            raise ValueError("timeout must be between 1 and 900 seconds")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


class ControlCheckpointStore:
    Record = IngestionCheckpointStore.Record

    def __init__(self, control: ControlCheckpointBackend) -> None:
        self._control = control
        self._previous: dict[str, IngestionCheckpoint | None] = {}

    async def load(
        self,
        job_digest: str,
        *,
        deadline: RequestDeadline,
    ) -> IngestionCheckpointStore.Record | None:
        checkpoint = await self._control.load_checkpoint(
            job_digest=job_digest,
            deadline=deadline,
        )
        self._previous[job_digest] = checkpoint
        if checkpoint is None:
            return None
        return IngestionCheckpointStore.Record(
            job_digest=checkpoint.job_digest,
            cursor_digest=checkpoint.cursor_digest,
            processed_count=checkpoint.processed_count,
            failed_count=checkpoint.failed_count,
            complete=checkpoint.complete,
        )

    async def save(
        self,
        record: IngestionCheckpointStore.Record,
        *,
        deadline: RequestDeadline,
    ) -> None:
        checkpoint = IngestionCheckpoint(
            job_digest=record.job_digest,
            cursor_digest=record.cursor_digest,
            processed_count=record.processed_count,
            failed_count=record.failed_count,
            complete=record.complete,
        )
        saved = await self._control.save_checkpoint(
            checkpoint=checkpoint,
            expected_previous=self._previous.get(record.job_digest),
            deadline=deadline,
        )
        if not saved:
            raise RuntimeError("research ingestion checkpoint conflict")
        self._previous[record.job_digest] = checkpoint


def parse_iso_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError("date must use ISO YYYY-MM-DD") from None
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run SEC research ingestion.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--cik", required=True)
    parser.add_argument("--forms", required=True)
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser


def _request_from_args(args: argparse.Namespace) -> ResearchIngestionRequest:
    return ResearchIngestionRequest(
        symbol=args.symbol,
        cik=args.cik,
        filing_types=tuple(item for item in args.forms.split(",") if item),
        date_from=parse_iso_date(args.date_from),
        date_to=parse_iso_date(args.date_to),
        limit=args.limit,
        apply=bool(args.apply),
    )


def _settings() -> Settings:
    return get_settings()


def _config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(timeout_seconds=args.timeout_seconds)


def _require_live_settings(settings: Settings) -> Settings:
    if (
        settings.openai_api_key is None
        or settings.upstash_vector_rest_url is None
        or settings.upstash_vector_rest_token is None
        or settings.upstash_redis_rest_url is None
        or settings.upstash_redis_rest_token is None
        or not settings.sec_user_agent
    ):
        raise ValueError("research ingestion configuration is incomplete")
    return settings


async def _default_runner_factory(config: RunnerConfig) -> OwnedResearchIngestionRunner:
    from app.providers.openai_research import OpenAIAnswerGenerator, OpenAIEmbedder
    from app.providers.redis_research import RedisResearchControl
    from app.providers.sec_filing_parser import SecFilingParser
    from app.providers.sec_filings import SecFilingSource
    from app.providers.upstash_vector import UpstashVectorStore
    from app.research.chunker import DeterministicSectionChunker
    from app.research.domain import CorpusDescriptor
    from app.research.ingestion import ResearchIngestionRunner as Runner

    research_service = importlib.import_module("app.research.service")

    settings = _require_live_settings(_settings())
    openai_key = settings.openai_api_key
    redis_url = settings.upstash_redis_rest_url
    redis_token = settings.upstash_redis_rest_token
    vector_url = settings.upstash_vector_rest_url
    vector_token = settings.upstash_vector_rest_token
    if (
        openai_key is None
        or redis_url is None
        or redis_token is None
        or vector_url is None
        or vector_token is None
    ):
        raise ValueError("research ingestion configuration is incomplete")
    owned_resources: tuple[object, ...] = ()
    try:
        embedder = OpenAIEmbedder(api_key=openai_key)
        owned_resources = (*owned_resources, embedder)
        generator = OpenAIAnswerGenerator(
            api_key=openai_key,
            max_output_tokens=settings.research_generation_max_output_tokens,
        )
        owned_resources = (*owned_resources, generator)
        control_plane = RedisResearchControl(
            redis_url,
            redis_token,
            timeout_seconds=min(3.0, config.timeout_seconds),
        )
        owned_resources = (*owned_resources, control_plane)
        vector_store = UpstashVectorStore(
            vector_url,
            vector_token,
            namespace=settings.research_vector_namespace,
        )
        owned_resources = (*owned_resources, vector_store)
        core = research_service.ResearchCoreService(
            corpus=CorpusDescriptor(
                corpus_version=settings.research_index_schema_version,
                chunker_version=(
                    "tokens-"
                    f"{settings.research_chunk_tokens}-"
                    f"{settings.research_chunk_overlap_tokens}-v1"
                ),
                embedding=embedder.descriptor,
            ),
            chunker=DeterministicSectionChunker(
                max_tokens=settings.research_chunk_tokens,
                overlap_tokens=settings.research_chunk_overlap_tokens,
            ),
            embedder=embedder,
            store=vector_store,
            control_plane=control_plane,
            generator=generator,
        )
        source = SecFilingSource(user_agent=settings.sec_user_agent)
        owned_resources = (*owned_resources, source)
        runner = Runner(
            source=source,
            parser=SecFilingParser(),
            core=core,
            checkpoints=ControlCheckpointStore(control_plane),
        )
    except BaseException as error:
        try:
            await _close_owned_resources(owned_resources)
        except Exception as close_error:
            error.add_note(f"research provider cleanup also failed: {type(close_error).__name__}")
        raise
    return OwnedResearchIngestionRunner(
        runner=runner,
        owned_resources=owned_resources,
    )


def _result_payload(result: ResultLike) -> dict[str, object]:
    return {
        "applied": not result.dry_run,
        "dry_run": result.dry_run,
        "errors": list(result.errors),
        "failed_count": result.failed_count,
        "inserted_count": result.inserted_count,
        "job_id": result.opaque_job_id,
        "planned_count": result.planned_count,
        "processed_count": result.processed_count,
        "removed_count": result.removed_count,
        "skipped_count": result.skipped_count,
    }


def _failure_category(error: Exception) -> str:
    name = type(error).__name__
    message = str(error).lower()
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, ValueError):
        return "configuration"
    if "checkpoint" in message or "control" in name.lower() or "lock" in message:
        return "lock_recovery"
    if (
        "provider" in name.lower()
        or "unavailable" in name.lower()
        or "openai" in name.lower()
        or "upstash" in name.lower()
    ):
        return "provider"
    return "unexpected"


def _failure_exit_code(category: str) -> int:
    return {
        "configuration": EXIT_CONFIG,
        "timeout": EXIT_TIMEOUT,
        "lock_recovery": EXIT_LOCK_RECOVERY,
        "provider": EXIT_PROVIDER,
    }.get(category, EXIT_UNEXPECTED)


def _failure_payload(request: ResearchIngestionRequest, category: str) -> dict[str, object]:
    return {
        "applied": bool(request.apply),
        "dry_run": not request.apply,
        "error_category": category,
        "errors": [f"research ingestion {category} failure"],
        "failed_count": 1,
        "inserted_count": 0,
        "job_id": None,
        "planned_count": 0,
        "processed_count": 0,
        "removed_count": 0,
        "skipped_count": 0,
    }


async def async_main(
    argv: list[str] | None = None,
    *,
    runner_factory: object | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        request = _request_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        config = _config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        if runner_factory is None:
            owned_runner = await _default_runner_factory(config)
            try:
                result = cast(
                    ResultLike,
                    await owned_runner.run(
                        request,
                        deadline=RequestDeadline.after(config.timeout_seconds),
                    ),
                )
            finally:
                await owned_runner.aclose()
        else:
            runner = cast(RunnerLike, runner_factory(config))  # type: ignore[operator]
            result = cast(
                ResultLike,
                await runner.run(
                    request,
                    deadline=RequestDeadline.after(config.timeout_seconds),
                ),
            )
    except Exception as error:
        category = _failure_category(error)
        print(json.dumps(_failure_payload(request, category), sort_keys=True))
        return _failure_exit_code(category)
    payload = _result_payload(result)
    print(json.dumps(payload, sort_keys=True))
    return EXIT_PARTIAL if result.failed_count else EXIT_SUCCESS


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
