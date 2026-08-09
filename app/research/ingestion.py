from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from app.research.deadline import RequestDeadline
from app.research.domain import (
    FilingDiscoveryRequest,
    FilingDocument,
    FilingReference,
)
from app.research.ports import FilingParser, FilingSource


class IngestOutcome(Protocol):
    @property
    def inserted_count(self) -> int: ...

    @property
    def removed_count(self) -> int: ...


class CoreIngestor(Protocol):
    async def ingest(
        self,
        document: FilingDocument,
        *,
        deadline: RequestDeadline | None = None,
    ) -> IngestOutcome: ...


_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")
_CIK = re.compile(r"^\d{10}$")
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_ALLOWED_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A"})


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_symbol(value: object) -> str:
    normalized = value.strip().upper() if isinstance(value, str) else ""
    if _SYMBOL.fullmatch(normalized) is None:
        raise ValueError("symbol is invalid")
    return normalized


def _validate_cik(value: object) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _CIK.fullmatch(normalized) is None:
        raise ValueError("CIK is invalid")
    return normalized


def _validate_forms(values: tuple[str, ...]) -> tuple[str, ...]:
    forms = tuple(item.strip().upper() for item in values if isinstance(item, str) and item.strip())
    if not forms or len(forms) > len(_ALLOWED_FORMS):
        raise ValueError("filing type allowlist is invalid")
    if len(set(forms)) != len(forms) or any(item not in _ALLOWED_FORMS for item in forms):
        raise ValueError("filing type allowlist is invalid")
    return forms


@dataclass(frozen=True, slots=True)
class ResearchIngestionRequest:
    symbol: str
    cik: str
    filing_types: tuple[str, ...]
    date_from: date
    date_to: date
    limit: int
    apply: bool = False

    def __post_init__(self) -> None:
        symbol = _validate_symbol(self.symbol)
        cik = _validate_cik(self.cik)
        forms = _validate_forms(tuple(self.filing_types))
        if type(self.date_from) is not date or type(self.date_to) is not date:
            raise ValueError("date allowlist is invalid")
        if self.date_from > self.date_to:
            raise ValueError("date allowlist is invalid")
        if type(self.limit) is not int or not 1 <= self.limit <= 100:
            raise ValueError("count allowlist is invalid")
        if type(self.apply) is not bool:
            raise ValueError("apply flag is invalid")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "cik", cik)
        object.__setattr__(self, "filing_types", forms)

    def discovery_request(self) -> FilingDiscoveryRequest:
        return FilingDiscoveryRequest(
            symbol=self.symbol,
            cik=self.cik,
            filing_types=self.filing_types,
            date_from=self.date_from,
            date_to=self.date_to,
            limit=self.limit,
        )


@dataclass(frozen=True, slots=True)
class ResearchIngestionJob:
    request: ResearchIngestionRequest
    references: tuple[FilingReference, ...] = field(repr=False)
    job_digest: str

    @classmethod
    def from_plan(
        cls,
        request: ResearchIngestionRequest,
        references: tuple[FilingReference, ...],
    ) -> ResearchIngestionJob:
        if len(references) > request.limit or any(
            not isinstance(item, FilingReference) for item in references
        ):
            raise ValueError("ingestion plan is invalid")
        material = "|".join(
            (
                request.symbol,
                request.cik,
                ",".join(request.filing_types),
                request.date_from.isoformat(),
                request.date_to.isoformat(),
                str(request.limit),
                ",".join(item.accession_number for item in references),
            )
        )
        return cls(request=request, references=references, job_digest=_digest(material))

    def cursor_for(self, accession_number: str) -> str:
        if not isinstance(accession_number, str) or _ACCESSION.fullmatch(accession_number) is None:
            raise ValueError("checkpoint cursor accession is invalid")
        if all(item.accession_number != accession_number for item in self.references):
            raise ValueError("checkpoint cursor is outside the ingestion plan")
        return _digest(f"{self.job_digest}|{accession_number}")

    def cursor_index(self, cursor_digest: str | None) -> int:
        if cursor_digest is None:
            return 0
        if (
            not isinstance(cursor_digest, str)
            or re.fullmatch(r"^[0-9a-f]{64}$", cursor_digest) is None
        ):
            raise ValueError("checkpoint cursor is invalid")
        for index, reference in enumerate(self.references, start=1):
            if self.cursor_for(reference.accession_number) == cursor_digest:
                return index
        raise ValueError("checkpoint cursor is outside the ingestion plan")


class IngestionCheckpointStore(Protocol):
    @dataclass(frozen=True, slots=True)
    class Record:
        job_digest: str
        cursor_digest: str | None
        processed_count: int
        failed_count: int
        complete: bool

        def __post_init__(self) -> None:
            if not isinstance(self.job_digest, str) or not re.fullmatch(
                r"^[0-9a-f]{64}$", self.job_digest
            ):
                raise ValueError("checkpoint job ID is invalid")
            if self.cursor_digest is not None and (
                not isinstance(self.cursor_digest, str)
                or re.fullmatch(r"^[0-9a-f]{64}$", self.cursor_digest) is None
            ):
                raise ValueError("checkpoint cursor is invalid")
            if type(self.processed_count) is not int or self.processed_count < 0:
                raise ValueError("checkpoint processed count is invalid")
            if type(self.failed_count) is not int or self.failed_count < 0:
                raise ValueError("checkpoint failed count is invalid")
            if type(self.complete) is not bool:
                raise ValueError("checkpoint completion is invalid")

    async def load(self, job_digest: str, *, deadline: RequestDeadline) -> Record | None: ...
    async def save(self, record: Record, *, deadline: RequestDeadline) -> None: ...


@dataclass(frozen=True, slots=True)
class ResearchIngestionResult:
    dry_run: bool
    job: ResearchIngestionJob = field(repr=False)
    planned_count: int
    processed_count: int
    skipped_count: int
    failed_count: int
    inserted_count: int
    removed_count: int
    errors: tuple[str, ...] = ()

    @property
    def opaque_job_id(self) -> str:
        return self.job.job_digest

    @property
    def opaque_accession_ids(self) -> tuple[str, ...]:
        return tuple(_digest(item.accession_number) for item in self.job.references)


class ResearchIngestionRunner:
    def __init__(
        self,
        *,
        source: FilingSource,
        parser: FilingParser,
        core: CoreIngestor,
        checkpoints: IngestionCheckpointStore | None = None,
    ) -> None:
        self._source = source
        self._parser = parser
        self._core = core
        self._checkpoints = checkpoints

    async def run(
        self,
        request: ResearchIngestionRequest,
        *,
        deadline: RequestDeadline,
    ) -> ResearchIngestionResult:
        if not isinstance(request, ResearchIngestionRequest):
            raise ValueError("ingestion request is invalid")
        deadline.raise_if_expired()
        page = await self._source.discover(request.discovery_request(), deadline=deadline)
        job = ResearchIngestionJob.from_plan(request, tuple(page.references))
        if not request.apply:
            return ResearchIngestionResult(
                dry_run=True,
                job=job,
                planned_count=len(job.references),
                processed_count=0,
                skipped_count=0,
                failed_count=0,
                inserted_count=0,
                removed_count=0,
            )
        checkpoint = await self._load_checkpoint(job.job_digest, deadline=deadline)
        resume_index = job.cursor_index(checkpoint.cursor_digest) if checkpoint is not None else 0
        failed_count = checkpoint.failed_count if checkpoint is not None else 0
        total_success_count = checkpoint.processed_count if checkpoint is not None else 0
        processed_count = 0
        inserted_count = 0
        removed_count = 0
        errors: tuple[str, ...] = ()
        cursor_digest = checkpoint.cursor_digest if checkpoint is not None else None
        for index, reference in enumerate(job.references):
            if index < resume_index:
                continue
            try:
                raw = await self._source.fetch(reference, deadline=deadline)
                document = self._parser.parse(raw)
                result = await self._core.ingest(document, deadline=deadline)
                inserted_count += result.inserted_count
                removed_count += result.removed_count
                processed_count += 1
                total_success_count += 1
            except Exception:
                failed_count += 1
                errors = (*errors, "ingestion failed for one filing")
            cursor_digest = job.cursor_for(reference.accession_number)
            await self._save_checkpoint(
                IngestionCheckpointStore.Record(
                    job_digest=job.job_digest,
                    cursor_digest=cursor_digest,
                    processed_count=total_success_count,
                    failed_count=failed_count,
                    complete=False,
                ),
                deadline=deadline,
            )
        new_failure_count = failed_count - (
            checkpoint.failed_count if checkpoint is not None else 0
        )
        complete = resume_index + processed_count + new_failure_count >= len(job.references)
        await self._save_checkpoint(
            IngestionCheckpointStore.Record(
                job_digest=job.job_digest,
                cursor_digest=cursor_digest,
                processed_count=total_success_count,
                failed_count=failed_count,
                complete=complete,
            ),
            deadline=deadline,
        )
        return ResearchIngestionResult(
            dry_run=False,
            job=job,
            planned_count=len(job.references),
            processed_count=processed_count,
            skipped_count=resume_index,
            failed_count=failed_count,
            inserted_count=inserted_count,
            removed_count=removed_count,
            errors=tuple(dict.fromkeys(errors)),
        )

    async def _load_checkpoint(
        self,
        job_digest: str,
        *,
        deadline: RequestDeadline,
    ) -> IngestionCheckpointStore.Record | None:
        if self._checkpoints is None:
            return None
        return await self._checkpoints.load(job_digest, deadline=deadline)

    async def _save_checkpoint(
        self,
        record: IngestionCheckpointStore.Record,
        *,
        deadline: RequestDeadline,
    ) -> None:
        if self._checkpoints is not None:
            await self._checkpoints.save(record, deadline=deadline)
