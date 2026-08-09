from __future__ import annotations

import re

from app.research.domain import FilingDocument

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.'_-][A-Za-z0-9]+)*|[^\s]", re.UNICODE)
_HEADING = re.compile(r"^(?:(?i:item\s+\d+[a-z]?\.?|part\s+[ivx]+)|[A-Z][A-Z0-9 &/.-]{3,})")


class DeterministicSectionChunker:
    """Deterministic, section-aware bounded tokenizer/chunker for normalized filings."""

    def __init__(
        self,
        *,
        max_tokens: int = 800,
        overlap_tokens: int = 100,
        max_chunks: int = 10_000,
    ) -> None:
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4_000:
            raise ValueError("maximum chunk tokens must be between 1 and 4000")
        if type(overlap_tokens) is not int or overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("chunk overlap must be nonnegative and smaller than the maximum")
        if type(max_chunks) is not int or not 1 <= max_chunks <= 10_000:
            raise ValueError("maximum chunk count must be between 1 and 10000")
        self._maximum = max_tokens
        self._overlap = overlap_tokens
        self._max_chunks = max_chunks

    def chunk(self, document: FilingDocument) -> tuple[str, ...]:
        if not isinstance(document, FilingDocument):
            raise ValueError("chunker requires a filing document")
        sections = self._sections(document.text)
        chunks: list[str] = []
        for section in sections:
            chunks.extend(self._section_chunks(section))
            if len(chunks) > self._max_chunks:
                raise ValueError("filing produced too many chunks")
        result = tuple(chunk for chunk in chunks if chunk)
        if not result:
            raise ValueError("filing produced no usable chunks")
        return result

    @staticmethod
    def tokens(value: str) -> tuple[str, ...]:
        return tuple(match.group(0) for match in _TOKEN.finditer(value))

    @classmethod
    def count_tokens(cls, value: str) -> int:
        return len(cls.tokens(value))

    @staticmethod
    def _sections(value: str) -> tuple[str, ...]:
        paragraphs = tuple(part.strip() for part in re.split(r"\n\s*\n", value) if part.strip())
        sections: list[str] = []
        current: tuple[str, ...] = ()
        for paragraph in paragraphs:
            if _HEADING.match(paragraph) and current:
                sections.append("\n\n".join(current))
                current = (paragraph,)
            else:
                current = (*current, paragraph)
        if current:
            sections.append("\n\n".join(current))
        return tuple(sections)

    def _section_chunks(self, section: str) -> list[str]:
        matches = tuple(_TOKEN.finditer(section))
        if not matches:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(matches):
            end = min(start + self._maximum, len(matches))
            start_character = matches[start].start()
            end_character = matches[end - 1].end()
            chunks.append(section[start_character:end_character].strip())
            if end == len(matches):
                break
            start = end - self._overlap
        return chunks


__all__ = ["DeterministicSectionChunker"]
