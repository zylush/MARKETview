from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from urllib.parse import unquote, urlsplit

_SEC_ARCHIVE_HOST = "www.sec.gov"
_DISCLAIMER = "Informational only, not investment advice."
_ACCESSION_NUMBER = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_SUPPORTED_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A"})
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,31}$")
_CIK = re.compile(r"^\d{10}$")
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_FILING_TYPE = re.compile(r"^(?:10-K|10-Q|8-K)(?:/A)?$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_SEC_ARCHIVE_DOCUMENT_PATH = re.compile(
    r"^/Archives/edgar/data/(?P<cik>[1-9][0-9]{0,9})/"
    r"(?P<accession>[0-9]{18})/(?P<filename>[A-Za-z0-9][A-Za-z0-9._-]{0,254})$"
)


def _is_control_like(character: str) -> bool:
    return unicodedata.category(character) in {"Cc", "Cf"}


def _plain_display_excerpt(value: str, *, maximum: int) -> str:
    translated = "".join(
        " "
        if _is_control_like(character)
        else "\u2039"
        if character == "<"
        else "\u203a"
        if character == ">"
        else character
        for character in value
    )
    return " ".join(translated.split())[:maximum].rstrip()


def is_trusted_sec_archive_url(
    url: str,
    *,
    cik: str | None = None,
    accession_number: str | None = None,
) -> bool:
    """Return whether *url* is a canonical public SEC filing URL."""

    try:
        parsed = urlsplit(url)
        decoded_path = unquote(parsed.path)
    except (TypeError, ValueError):
        return False
    match = _SEC_ARCHIVE_DOCUMENT_PATH.fullmatch(parsed.path)
    if not bool(
        parsed.scheme == "https"
        and parsed.hostname == _SEC_ARCHIVE_HOST
        and parsed.netloc == _SEC_ARCHIVE_HOST
        and parsed.username is None
        and parsed.password is None
        and parsed.port is None
        and not parsed.query
        and not parsed.fragment
        and decoded_path == parsed.path
        and match is not None
        and "/./" not in parsed.path
        and "/../" not in parsed.path
    ):
        return False
    if match is None:  # pragma: no cover - narrowed by the condition above
        return False
    path_cik = match.group("cik")
    path_accession = match.group("accession")
    if cik is not None and (_CIK.fullmatch(cik) is None or path_cik != str(int(cik))):
        return False
    if accession_number is not None:
        if _ACCESSION.fullmatch(accession_number) is None:
            return False
        if path_accession != accession_number.replace("-", ""):
            return False
        if path_cik != str(int(accession_number[:10])):
            return False
    return True


def _require_text(value: object, name: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise ValueError(f"{name} must contain 1-{maximum} characters")
    return normalized


@dataclass(frozen=True, slots=True)
class FilingReference:
    """Trusted immutable metadata used by a future SEC filing-source adapter."""

    symbol: str
    cik: str
    accession_number: str
    filing_type: str
    title: str
    filed_date: date
    source_url: str

    def __post_init__(self) -> None:
        symbol = _require_text(self.symbol, "symbol", maximum=32).upper()
        cik = _require_text(self.cik, "CIK", maximum=10)
        accession = _require_text(self.accession_number, "accession number", maximum=20)
        filing_type = _require_text(self.filing_type, "filing type", maximum=6).upper()
        title = _require_text(self.title, "title", maximum=200)
        if not _SYMBOL.fullmatch(symbol):
            raise ValueError("filing symbol is malformed")
        if (
            not _CIK.fullmatch(cik)
            or not _ACCESSION_NUMBER.fullmatch(accession)
            or accession[:10] != cik
        ):
            raise ValueError("filing CIK or accession number is invalid")
        if filing_type not in _SUPPORTED_FORMS:
            raise ValueError("filing type is not supported")
        if any(character in title for character in "<>") or any(
            ord(character) < 32 or ord(character) == 127 for character in title
        ):
            raise ValueError("filing title must contain plain display text")
        if type(self.filed_date) is not date:
            raise ValueError("filed date must be a date")
        if not is_trusted_sec_archive_url(
            self.source_url,
            cik=cik,
            accession_number=accession,
        ):
            raise ValueError("research sources must use SEC Archives HTTPS URLs")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "cik", cik)
        object.__setattr__(self, "accession_number", accession)
        object.__setattr__(self, "filing_type", filing_type)
        object.__setattr__(self, "title", title)


@dataclass(frozen=True, slots=True)
class FilingDiscoveryCursor:
    """Opaque, bounded continuation token owned by a filing-source adapter."""

    value: str

    def __post_init__(self) -> None:
        value = _require_text(self.value, "filing discovery cursor", maximum=512)
        if any(_is_control_like(character) for character in value):
            raise ValueError("filing discovery cursor contains control characters")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class FilingDiscoveryRequest:
    symbol: str
    cik: str
    filing_types: tuple[str, ...]
    date_from: date
    date_to: date
    limit: int
    cursor: FilingDiscoveryCursor | None = None

    def __post_init__(self) -> None:
        symbol = _require_text(self.symbol, "symbol", maximum=32).upper()
        cik = _require_text(self.cik, "CIK", maximum=10)
        filing_types = tuple(
            _require_text(item, "filing type", maximum=6).upper() for item in self.filing_types
        )
        if not _SYMBOL.fullmatch(symbol) or not _CIK.fullmatch(cik):
            raise ValueError("filing discovery identity is malformed")
        if not filing_types or len(filing_types) > len(_SUPPORTED_FORMS):
            raise ValueError("filing discovery requires a bounded filing type list")
        if len(set(filing_types)) != len(filing_types) or any(
            item not in _SUPPORTED_FORMS for item in filing_types
        ):
            raise ValueError("filing discovery contains duplicate or unsupported filing types")
        if type(self.date_from) is not date or type(self.date_to) is not date:
            raise ValueError("filing discovery dates must be dates")
        if self.date_from > self.date_to:
            raise ValueError("filing discovery date range is invalid")
        if type(self.limit) is not int or not 1 <= self.limit <= 100:
            raise ValueError("filing discovery limit must be between 1 and 100")
        if self.cursor is not None and not isinstance(self.cursor, FilingDiscoveryCursor):
            raise ValueError("filing discovery cursor is invalid")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "cik", cik)
        object.__setattr__(self, "filing_types", filing_types)


@dataclass(frozen=True, slots=True)
class FilingDiscoveryPage:
    references: tuple[FilingReference, ...]
    next_cursor: FilingDiscoveryCursor | None = None

    def __post_init__(self) -> None:
        references = tuple(self.references)
        if len(references) > 100 or any(
            not isinstance(reference, FilingReference) for reference in references
        ):
            raise ValueError("filing discovery page is invalid or exceeds 100 references")
        identities = tuple((reference.cik, reference.accession_number) for reference in references)
        if len(set(identities)) != len(identities):
            raise ValueError("filing discovery page contains duplicate references")
        if self.next_cursor is not None and not isinstance(self.next_cursor, FilingDiscoveryCursor):
            raise ValueError("filing discovery page cursor is invalid")
        object.__setattr__(self, "references", references)


@dataclass(frozen=True, slots=True)
class RawFiling:
    """Fetched filing payload passed to a separately configured parser."""

    reference: FilingReference
    media_type: str
    body: bytes = field(repr=False)
    content_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference, FilingReference):
            raise ValueError("raw filing must be bound to trusted filing metadata")
        media_type = _require_text(self.media_type, "raw filing media type", maximum=64).lower()
        if media_type not in _MEDIA_TYPES:
            raise ValueError("raw filing media type is unsupported")
        if not isinstance(self.body, bytes) or not self.body:
            raise ValueError("raw filing body must not be empty")
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "content_hash", hashlib.sha256(self.body).hexdigest())


@dataclass(frozen=True, slots=True)
class FilingDocument:
    symbol: str
    cik: str
    accession_number: str
    filing_type: str
    title: str
    filed_date: date
    source_url: str
    text: str = field(repr=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        symbol = _require_text(self.symbol, "symbol", maximum=32).upper()
        cik = _require_text(self.cik, "CIK", maximum=20)
        accession = _require_text(self.accession_number, "accession number", maximum=32)
        filing_type = _require_text(self.filing_type, "filing type", maximum=12)
        title = _require_text(self.title, "title", maximum=200)
        text = _require_text(self.text, "filing text", maximum=5_000_000)
        if not _SYMBOL.fullmatch(symbol):
            raise ValueError("filing symbol is malformed")
        if not _CIK.fullmatch(cik):
            raise ValueError("filing CIK must be ten digits")
        if not _ACCESSION.fullmatch(accession):
            raise ValueError("filing accession number is malformed")
        if not _FILING_TYPE.fullmatch(filing_type):
            raise ValueError("filing type is unsupported")
        if any(character in title for character in "<>") or any(
            ord(character) < 32 or ord(character) == 127 for character in title
        ):
            raise ValueError("filing title must contain plain display text")
        if type(self.filed_date) is not date:
            raise ValueError("filing date must be a date")
        if accession[:10] != cik or not is_trusted_sec_archive_url(
            self.source_url,
            cik=cik,
            accession_number=accession,
        ):
            raise ValueError("research sources must use SEC Archives HTTPS URLs")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "cik", cik)
        object.__setattr__(self, "accession_number", accession)
        object.__setattr__(self, "filing_type", filing_type)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "content_hash", hashlib.sha256(text.encode("utf-8")).hexdigest())


@dataclass(frozen=True, slots=True)
class EmbeddingDescriptor:
    provider: str
    model: str
    version: str
    dimensions: int

    def __post_init__(self) -> None:
        provider = _require_text(self.provider, "embedding provider", maximum=64).lower()
        model = _require_text(self.model, "embedding model", maximum=128)
        version = _require_text(self.version, "embedding version", maximum=128)
        if not all(_VERSION.fullmatch(item) for item in (provider, model, version)):
            raise ValueError("embedding provider, model, or version is malformed")
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= 65_536:
            raise ValueError("embedding dimensions must be between 1 and 65536")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "version", version)

    @property
    def canonical_key(self) -> str:
        return f"{self.provider}:{self.model}:{self.version}:{self.dimensions}"


@dataclass(frozen=True, slots=True)
class CorpusDescriptor:
    corpus_version: str
    chunker_version: str
    embedding: EmbeddingDescriptor

    def __post_init__(self) -> None:
        corpus_version = _require_text(self.corpus_version, "corpus version", maximum=128).lower()
        chunker_version = _require_text(
            self.chunker_version, "chunker version", maximum=128
        ).lower()
        if not _VERSION.fullmatch(corpus_version) or not _VERSION.fullmatch(chunker_version):
            raise ValueError("corpus or chunker version is malformed")
        if not isinstance(self.embedding, EmbeddingDescriptor):
            raise ValueError("corpus embedding descriptor is invalid")
        object.__setattr__(self, "corpus_version", corpus_version)
        object.__setattr__(self, "chunker_version", chunker_version)

    @property
    def canonical_key(self) -> str:
        return f"{self.corpus_version}:{self.chunker_version}:{self.embedding.canonical_key}"


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    descriptor: EmbeddingDescriptor
    values: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, EmbeddingDescriptor):
            raise ValueError("embedding descriptor is invalid")
        values = tuple(self.values)
        if len(values) != self.descriptor.dimensions or any(
            type(value) is not float or not math.isfinite(value) for value in values
        ):
            raise ValueError(
                "embedding values must be finite floats matching the configured dimensions"
            )
        object.__setattr__(self, "values", values)


def _canonical_digest(*parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    reference: FilingReference
    corpus: CorpusDescriptor
    generation_id: str
    chunk_id: str
    content_hash: str
    ordinal: int
    text: str = field(repr=False)
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference, FilingReference) or not isinstance(
            self.corpus, CorpusDescriptor
        ):
            raise ValueError("evidence filing or corpus metadata is invalid")
        generation_id = _require_text(self.generation_id, "evidence generation ID", maximum=80)
        chunk_id = _require_text(self.chunk_id, "evidence chunk ID", maximum=80)
        content_hash = _require_text(self.content_hash, "evidence content hash", maximum=64)
        text = _require_text(self.text, "evidence text", maximum=20_000)
        if not re.fullmatch(r"gen-[0-9a-f]{64}", generation_id):
            raise ValueError("evidence generation ID is not canonical")
        if not re.fullmatch(r"chunk-[0-9a-f]{64}", chunk_id):
            raise ValueError("evidence chunk ID is not canonical")
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise ValueError("evidence content hash is malformed")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("evidence chunk ordinal must not be negative")
        expected_generation = self.canonical_generation_id(
            self.reference, self.corpus, content_hash
        )
        evidence_digest = self.canonical_evidence_digest(self.reference, text)
        expected_chunk = self.canonical_chunk_id(
            self.reference,
            self.corpus,
            content_hash,
            self.ordinal,
            text=text,
        )
        legacy_chunk = self.canonical_chunk_id(
            self.reference,
            self.corpus,
            content_hash,
            self.ordinal,
        )
        if generation_id != expected_generation or chunk_id not in {expected_chunk, legacy_chunk}:
            raise ValueError("evidence identifiers do not match canonical metadata")
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "evidence_digest", evidence_digest)

    @property
    def symbol(self) -> str:
        return self.reference.symbol

    @property
    def cik(self) -> str:
        return self.reference.cik

    @property
    def accession_number(self) -> str:
        return self.reference.accession_number

    @property
    def filing_type(self) -> str:
        return self.reference.filing_type

    @property
    def title(self) -> str:
        return self.reference.title

    @property
    def filed_date(self) -> date:
        return self.reference.filed_date

    @property
    def source_url(self) -> str:
        return self.reference.source_url

    @classmethod
    def canonical_generation_id(
        cls,
        reference: FilingReference,
        corpus: CorpusDescriptor,
        content_hash: str,
    ) -> str:
        return "gen-" + _canonical_digest(
            corpus.canonical_key,
            reference.symbol,
            cls.canonical_citation_digest(reference),
            reference.accession_number,
            content_hash,
        )

    @staticmethod
    def canonical_citation_digest(reference: FilingReference) -> str:
        if not isinstance(reference, FilingReference):
            raise ValueError("evidence filing metadata is invalid")
        return _canonical_digest(
            reference.cik,
            reference.accession_number,
            reference.filing_type,
            reference.filed_date.isoformat(),
            reference.title,
            reference.source_url,
        )

    @classmethod
    def canonical_evidence_digest(cls, reference: FilingReference, text: str) -> str:
        checked_text = _require_text(text, "evidence text", maximum=20_000)
        return _canonical_digest(cls.canonical_citation_digest(reference), checked_text)

    @classmethod
    def canonical_chunk_id(
        cls,
        reference: FilingReference,
        corpus: CorpusDescriptor,
        content_hash: str,
        ordinal: int,
        *,
        text: str | None = None,
    ) -> str:
        evidence_digest = (
            cls.canonical_evidence_digest(reference, text) if text is not None else "legacy"
        )
        return "chunk-" + _canonical_digest(
            corpus.canonical_key,
            reference.symbol,
            cls.canonical_citation_digest(reference),
            content_hash,
            ordinal,
            evidence_digest,
        )

    @classmethod
    def from_document(
        cls,
        document: FilingDocument,
        *,
        corpus: CorpusDescriptor,
        ordinal: int,
        text: str,
    ) -> EvidenceChunk:
        reference = FilingReference(
            symbol=document.symbol,
            cik=document.cik,
            accession_number=document.accession_number,
            filing_type=document.filing_type,
            title=document.title,
            filed_date=document.filed_date,
            source_url=document.source_url,
        )
        return cls(
            reference=reference,
            corpus=corpus,
            generation_id=cls.canonical_generation_id(reference, corpus, document.content_hash),
            chunk_id=cls.canonical_chunk_id(
                reference,
                corpus,
                document.content_hash,
                ordinal,
                text=text,
            ),
            content_hash=document.content_hash,
            ordinal=ordinal,
            text=text,
        )


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    evidence: EvidenceChunk
    embedding: EmbeddingVector = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EvidenceChunk) or not isinstance(
            self.embedding, EmbeddingVector
        ):
            raise ValueError("embedded chunk payload is invalid")
        if self.embedding.descriptor != self.evidence.corpus.embedding:
            raise ValueError("embedded chunk descriptor does not match its corpus")


@dataclass(frozen=True, slots=True)
class SearchHit:
    evidence: EvidenceChunk
    score: float
    embedding_descriptor: EmbeddingDescriptor
    active_generation_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EvidenceChunk) or not isinstance(
            self.embedding_descriptor, EmbeddingDescriptor
        ):
            raise ValueError("research search hit metadata is invalid")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("research similarity score must be between zero and one")
        active_generation_id = _require_text(
            self.active_generation_id, "active generation ID", maximum=80
        )
        if not re.fullmatch(r"gen-[0-9a-f]{64}", active_generation_id):
            raise ValueError("active generation ID is malformed")
        object.__setattr__(self, "active_generation_id", active_generation_id)


@dataclass(frozen=True, slots=True)
class EvidenceQuote:
    chunk_id: str
    quote: str

    def __post_init__(self) -> None:
        chunk_id = _require_text(self.chunk_id, "evidence quote chunk ID", maximum=80)
        quote = _require_text(self.quote, "evidence quote", maximum=500)
        if not re.fullmatch(r"chunk-[0-9a-f]{64}", chunk_id):
            raise ValueError("evidence quote chunk ID is malformed")
        if any(_is_control_like(character) for character in quote):
            raise ValueError("evidence quote contains control characters")
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "quote", quote)


@dataclass(frozen=True, slots=True)
class GeneratedClaim:
    text: str
    supporting_chunk_ids: tuple[str, ...]
    evidence_quotes: tuple[EvidenceQuote, ...] = ()

    def __post_init__(self) -> None:
        text = _require_text(self.text, "generated claim", maximum=1000)
        supporting_chunk_ids = tuple(
            _require_text(item, "generated claim chunk ID", maximum=80)
            for item in self.supporting_chunk_ids
        )
        evidence_quotes = tuple(self.evidence_quotes)
        if not supporting_chunk_ids:
            raise ValueError("every generated claim must reference at least one chunk ID")
        if len(set(supporting_chunk_ids)) != len(supporting_chunk_ids):
            raise ValueError("generated claim chunk IDs must be unique")
        if any(not isinstance(item, EvidenceQuote) for item in evidence_quotes):
            raise ValueError("generated claim evidence quotes are invalid")
        quoted_ids = {item.chunk_id for item in evidence_quotes}
        if evidence_quotes and quoted_ids != set(supporting_chunk_ids):
            raise ValueError("every generated claim chunk ID requires an evidence quote")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "supporting_chunk_ids", supporting_chunk_ids)
        object.__setattr__(self, "evidence_quotes", evidence_quotes)


class GeneratedAnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    status: GeneratedAnswerStatus
    claims: tuple[GeneratedClaim, ...] = ()

    def __post_init__(self) -> None:
        claims = tuple(self.claims)
        if not isinstance(self.status, GeneratedAnswerStatus):
            raise ValueError("generated answer status is invalid")
        if any(not isinstance(claim, GeneratedClaim) for claim in claims):
            raise ValueError("generated answer claims are invalid")
        if self.status is GeneratedAnswerStatus.ANSWERED and not claims:
            raise ValueError("answered generated outcome requires at least one claim")
        if self.status is not GeneratedAnswerStatus.ANSWERED and claims:
            raise ValueError("non-answer generated outcome must not contain claims")
        object.__setattr__(self, "claims", claims)


@dataclass(frozen=True, slots=True)
class ResearchCitation:
    chunk_id: str
    title: str
    url: str
    filed_date: date
    filing_type: str
    accession_number: str
    snippet: str

    def __post_init__(self) -> None:
        chunk_id = _require_text(self.chunk_id, "citation chunk ID", maximum=100)
        title = _require_text(self.title, "citation title", maximum=200)
        filing_type = _require_text(self.filing_type, "citation filing type", maximum=6).upper()
        accession = _require_text(
            self.accession_number,
            "citation accession number",
            maximum=20,
        )
        snippet = _require_text(self.snippet, "citation snippet", maximum=300)
        if not is_trusted_sec_archive_url(
            self.url,
            accession_number=accession,
        ):
            raise ValueError("research citations must use SEC Archives HTTPS URLs")
        if not _FILING_TYPE.fullmatch(filing_type) or not _ACCESSION.fullmatch(accession):
            raise ValueError("citation filing metadata is malformed")
        if type(self.filed_date) is not date:
            raise ValueError("citation filed date must be a date")
        if any(character in title + snippet for character in "<>") or any(
            _is_control_like(character) for character in title + snippet
        ):
            raise ValueError("citation title or snippet must contain plain display text")
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "filing_type", filing_type)
        object.__setattr__(self, "accession_number", accession)
        object.__setattr__(self, "snippet", snippet)

    @classmethod
    def from_chunk(cls, chunk: EvidenceChunk) -> ResearchCitation:
        return cls(
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            url=chunk.source_url,
            filed_date=chunk.filed_date,
            filing_type=chunk.filing_type,
            accession_number=chunk.accession_number,
            snippet=_plain_display_excerpt(chunk.text, maximum=300),
        )


class ResearchOutcome(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class ResearchAnswer:
    symbol: str
    answer: str
    outcome: ResearchOutcome
    insufficient_evidence: bool
    refused: bool
    claims: tuple[GeneratedClaim, ...] = ()
    citations: tuple[ResearchCitation, ...] = ()
    disclaimer: str = _DISCLAIMER

    def __post_init__(self) -> None:
        symbol = _require_text(self.symbol, "symbol", maximum=32).upper()
        if not _SYMBOL.fullmatch(symbol):
            raise ValueError("research answer symbol is malformed")
        answer = _require_text(self.answer, "research answer", maximum=2000)
        claims = tuple(self.claims)
        citations = tuple(self.citations)
        if any(not isinstance(claim, GeneratedClaim) for claim in claims) or any(
            not isinstance(citation, ResearchCitation) for citation in citations
        ):
            raise ValueError("research claim and citation types are invalid")
        if self.disclaimer != _DISCLAIMER:
            raise ValueError("research disclaimer must use the approved informational text")
        if self.outcome is ResearchOutcome.ANSWERED:
            referenced = frozenset(
                chunk_id for claim in claims for chunk_id in claim.supporting_chunk_ids
            )
            cited = frozenset(citation.chunk_id for citation in citations)
            if (
                self.insufficient_evidence
                or self.refused
                or not claims
                or not citations
                or not referenced.issubset(cited)
                or answer != " ".join(claim.text for claim in claims)
            ):
                raise ValueError("answered research requires grounded claims and citations")
        elif self.outcome is ResearchOutcome.INSUFFICIENT_EVIDENCE:
            if (
                not self.insufficient_evidence
                or self.refused
                or claims
                or citations
                or answer != "Insufficient evidence."
            ):
                raise ValueError("insufficient research outcome is inconsistent")
        elif self.outcome is ResearchOutcome.REFUSED:
            if (
                self.insufficient_evidence
                or not self.refused
                or claims
                or citations
                or answer != "I cannot provide personalized buy or sell recommendations."
            ):
                raise ValueError("refused research outcome is inconsistent")
        else:
            raise ValueError("research outcome is invalid")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "citations", citations)

    @classmethod
    def answered(
        cls,
        symbol: str,
        claims: tuple[GeneratedClaim, ...],
        citations: tuple[ResearchCitation, ...],
    ) -> ResearchAnswer:
        return cls(
            symbol=symbol,
            answer=" ".join(claim.text for claim in claims),
            outcome=ResearchOutcome.ANSWERED,
            insufficient_evidence=False,
            refused=False,
            claims=claims,
            citations=citations,
        )

    @classmethod
    def insufficient(cls, symbol: str) -> ResearchAnswer:
        return cls(
            symbol=symbol,
            answer="Insufficient evidence.",
            outcome=ResearchOutcome.INSUFFICIENT_EVIDENCE,
            insufficient_evidence=True,
            refused=False,
        )

    @classmethod
    def refusal(cls, symbol: str) -> ResearchAnswer:
        return cls(
            symbol=symbol,
            answer="I cannot provide personalized buy or sell recommendations.",
            outcome=ResearchOutcome.REFUSED,
            insufficient_evidence=False,
            refused=True,
        )


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    corpus: CorpusDescriptor
    symbol: str
    accession_number: str
    generation_id: str
    content_hash: str
    chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, CorpusDescriptor):
            raise ValueError("generation manifest corpus is invalid")
        symbol = _require_text(self.symbol, "manifest symbol", maximum=32).upper()
        accession = _require_text(self.accession_number, "manifest accession number", maximum=20)
        generation_id = _require_text(self.generation_id, "manifest generation ID", maximum=80)
        content_hash = _require_text(self.content_hash, "manifest content hash", maximum=64)
        chunk_ids = tuple(self.chunk_ids)
        if not _SYMBOL.fullmatch(symbol) or not _ACCESSION.fullmatch(accession):
            raise ValueError("generation manifest filing identity is malformed")
        if not re.fullmatch(r"gen-[0-9a-f]{64}", generation_id):
            raise ValueError("generation manifest ID is malformed")
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise ValueError("generation manifest content hash is malformed")
        if (
            not chunk_ids
            or len(set(chunk_ids)) != len(chunk_ids)
            or any(not re.fullmatch(r"chunk-[0-9a-f]{64}", item) for item in chunk_ids)
        ):
            raise ValueError("generation manifest chunk IDs are malformed or duplicated")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "accession_number", accession)
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "chunk_ids", chunk_ids)


@dataclass(frozen=True, slots=True)
class GenerationVerification:
    generation_id: str
    point_count: int
    point_ids_hash: str

    def __post_init__(self) -> None:
        generation_id = _require_text(self.generation_id, "verification generation ID", maximum=80)
        point_ids_hash = _require_text(
            self.point_ids_hash, "verification point IDs hash", maximum=64
        )
        if not re.fullmatch(r"gen-[0-9a-f]{64}", generation_id):
            raise ValueError("verification generation ID is malformed")
        if type(self.point_count) is not int or self.point_count < 1:
            raise ValueError("verification point count must be positive")
        if not re.fullmatch(r"[0-9a-f]{64}", point_ids_hash):
            raise ValueError("verification point IDs hash is malformed")
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "point_ids_hash", point_ids_hash)

    @classmethod
    def from_point_ids(
        cls,
        generation_id: str,
        point_ids: tuple[str, ...],
    ) -> GenerationVerification:
        checked_ids = tuple(point_ids)
        if not checked_ids or len(set(checked_ids)) != len(checked_ids):
            raise ValueError("verification point IDs must be nonempty and unique")
        return cls(
            generation_id=generation_id,
            point_count=len(checked_ids),
            point_ids_hash=_canonical_digest(*sorted(checked_ids)),
        )

    def proves(self, manifest: GenerationManifest) -> bool:
        expected = self.from_point_ids(manifest.generation_id, manifest.chunk_ids)
        return self == expected


@dataclass(frozen=True, slots=True)
class IngestionResult:
    outcome: str
    inserted_count: int
    removed_count: int
    cleanup_pending_count: int = 0

    def __post_init__(self) -> None:
        if self.outcome not in {"created", "replaced", "unchanged"}:
            raise ValueError("unsupported ingestion outcome")
        if self.inserted_count < 0 or self.removed_count < 0 or self.cleanup_pending_count < 0:
            raise ValueError("ingestion counts must not be negative")
