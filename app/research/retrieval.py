from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from app.research.domain import CorpusDescriptor, GenerationManifest, SearchHit

_PROMPT_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
        r"\breveal\s+(?:the\s+)?(?:system|developer)\s+prompt\b",
        r"\b(?:system|developer)\s+message\s*:",
        r"\bdisclose\s+(?:your\s+)?hidden\s+instructions\b",
        r"\bact\s+as\s+(?:the\s+)?system\b",
        r"\b(?:ignore|forget|disregard|override|bypass|supersede)\b.{0,80}"
        r"\b(?:instructions?|directions?|rules?|guidance|safeguards?)\b",
        r"\b(?:reveal|expose|print|output|disclose|divulge)\b.{0,80}"
        r"\b(?:hidden|private|confidential|system|developer)\b.{0,40}"
        r"\b(?:prompts?|instructions?|directives?|configuration)\b",
        r"\btreat\b.{0,60}\b(?:higher[- ]priority|authoritative)\b.{0,40}"
        r"\b(?:instructions?|guidance|commands?)\b",
    )
)


@dataclass(frozen=True, slots=True)
class SafeHitRejectionCounts:
    inactive_manifest: int = 0
    duplicate_chunk: int = 0
    metadata_integrity: int = 0
    below_threshold: int = 0
    prompt_injection: int = 0
    result_limit: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in self.as_tuple()):
            raise ValueError("safe-hit rejection count is invalid")

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.inactive_manifest,
            self.duplicate_chunk,
            self.metadata_integrity,
            self.below_threshold,
            self.prompt_injection,
            self.result_limit,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "below_threshold": self.below_threshold,
            "duplicate_chunk": self.duplicate_chunk,
            "inactive_manifest": self.inactive_manifest,
            "metadata_integrity": self.metadata_integrity,
            "prompt_injection": self.prompt_injection,
            "result_limit": self.result_limit,
        }

    @property
    def total(self) -> int:
        return sum(self.as_tuple())


@dataclass(frozen=True, slots=True)
class SafeHitClassification:
    accepted_hits: tuple[SearchHit, ...]
    rejection_counts: SafeHitRejectionCounts
    candidate_count: int
    accepted_count: int
    raw_score_range: tuple[float, float] | None
    accepted_score_range: tuple[float, float] | None

    def __post_init__(self) -> None:
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise ValueError("safe-hit candidate count is invalid")
        if type(self.accepted_count) is not int or self.accepted_count < 0:
            raise ValueError("safe-hit accepted count is invalid")
        if self.accepted_count != len(self.accepted_hits):
            raise ValueError("safe-hit accepted count does not match accepted hits")
        if self.accepted_count + self.rejection_counts.total != self.candidate_count:
            raise ValueError("safe-hit classification does not account for every candidate")
        self._validate_range(self.raw_score_range, present=self.candidate_count > 0)
        self._validate_range(self.accepted_score_range, present=self.accepted_count > 0)

    @staticmethod
    def _validate_range(value: tuple[float, float] | None, *, present: bool) -> None:
        if (value is not None) != present:
            raise ValueError("safe-hit score range presence is invalid")
        if value is not None and (
            len(value) != 2
            or any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in value)
            or value[0] > value[1]
        ):
            raise ValueError("safe-hit score range is invalid")


def contains_prompt_injection(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text)
    safe_text = "".join(
        ""
        if unicodedata.category(character) == "Cf"
        else " "
        if unicodedata.category(character) == "Cc"
        else character
        for character in normalized
    )
    return any(pattern.search(safe_text) for pattern in _PROMPT_INJECTION_PATTERNS)


def classify_safe_hits(
    symbol: str,
    corpus: CorpusDescriptor,
    active_manifests: tuple[GenerationManifest, ...],
    hits: tuple[SearchHit, ...],
    *,
    minimum_score: float,
    max_results: int,
) -> SafeHitClassification:
    """Classify every ordered candidate while preserving production acceptance semantics."""

    if not isinstance(symbol, str) or not symbol:
        raise ValueError("safe-hit symbol is invalid")
    if not isinstance(corpus, CorpusDescriptor):
        raise ValueError("safe-hit corpus is invalid")
    if (
        isinstance(minimum_score, bool)
        or not isinstance(minimum_score, (int, float))
        or not math.isfinite(float(minimum_score))
        or not 0.0 <= float(minimum_score) <= 1.0
    ):
        raise ValueError("safe-hit minimum score is invalid")
    if type(max_results) is not int or not 1 <= max_results <= 20:
        raise ValueError("safe-hit result limit is invalid")
    active = tuple(active_manifests)
    candidates = tuple(hits)
    if any(not isinstance(item, GenerationManifest) for item in active):
        raise ValueError("safe-hit active manifest scope is invalid")
    if any(not isinstance(item, SearchHit) for item in candidates):
        raise ValueError("safe-hit candidate is invalid")

    active_by_generation = {item.generation_id: item for item in active}
    accepted: list[SearchHit] = []
    seen_chunks: set[str] = set()
    counts = {
        "inactive_manifest": 0,
        "duplicate_chunk": 0,
        "metadata_integrity": 0,
        "below_threshold": 0,
        "prompt_injection": 0,
        "result_limit": 0,
    }
    for hit in candidates:
        evidence = hit.evidence
        manifest = active_by_generation.get(hit.active_generation_id)
        if manifest is None:
            counts["inactive_manifest"] += 1
        elif evidence.chunk_id in seen_chunks:
            counts["duplicate_chunk"] += 1
        elif (
            evidence.symbol != symbol
            or evidence.corpus != corpus
            or hit.embedding_descriptor != corpus.embedding
            or hit.active_generation_id != evidence.generation_id
            or manifest.accession_number != evidence.accession_number
            or manifest.content_hash != evidence.content_hash
            or evidence.chunk_id not in manifest.chunk_ids
        ):
            counts["metadata_integrity"] += 1
        elif hit.score < float(minimum_score):
            counts["below_threshold"] += 1
        elif contains_prompt_injection(evidence.text):
            counts["prompt_injection"] += 1
        elif len(accepted) >= max_results:
            counts["result_limit"] += 1
        else:
            accepted.append(hit)
            seen_chunks.add(evidence.chunk_id)

    raw_scores = tuple(item.score for item in candidates)
    accepted_scores = tuple(item.score for item in accepted)
    rejection_counts = SafeHitRejectionCounts(**counts)
    return SafeHitClassification(
        accepted_hits=tuple(accepted),
        rejection_counts=rejection_counts,
        candidate_count=len(candidates),
        accepted_count=len(accepted),
        raw_score_range=(min(raw_scores), max(raw_scores)) if raw_scores else None,
        accepted_score_range=(min(accepted_scores), max(accepted_scores))
        if accepted_scores
        else None,
    )


__all__ = [
    "SafeHitClassification",
    "SafeHitRejectionCounts",
    "classify_safe_hits",
    "contains_prompt_injection",
]
