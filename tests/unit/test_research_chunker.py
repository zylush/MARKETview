from __future__ import annotations

from datetime import date
from itertools import pairwise

import pytest

from app.research.chunker import DeterministicSectionChunker
from app.research.domain import FilingDocument


def _document(text: str) -> FilingDocument:
    return FilingDocument(
        symbol="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000003",
        filing_type="10-K",
        title="Apple 2025 Form 10-K",
        filed_date=date(2025, 10, 31),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000003/aapl-20250927.htm"
        ),
        text=text,
    )


def test_chunker_is_deterministic_bounded_overlapping_and_section_aware() -> None:
    text = (
        "Item 1. Business\n\n"
        + " ".join(f"business{i}" for i in range(12))
        + "\n\nItem 1A. Risk Factors\n\n"
        + " ".join(f"risk{i}" for i in range(18))
    )
    chunker = DeterministicSectionChunker(max_tokens=12, overlap_tokens=3)

    first = chunker.chunk(_document(text))
    second = chunker.chunk(_document(text))

    assert first == second
    assert isinstance(first, tuple)
    assert len(first) >= 3
    assert all(chunker.count_tokens(chunk) <= 12 for chunk in first)
    assert first[0].startswith("Item 1. Business")
    assert any(chunk.startswith("Item 1A. Risk Factors") for chunk in first)
    for previous, current in pairwise(first):
        if current.startswith("Item "):
            continue
        previous_tokens = chunker.tokens(previous)
        current_tokens = chunker.tokens(current)
        assert set(previous_tokens[-3:]).intersection(current_tokens[:6])


def test_chunker_splits_a_single_oversized_section_without_losing_content() -> None:
    text = " ".join(f"token{i}" for i in range(31))
    chunker = DeterministicSectionChunker(max_tokens=10, overlap_tokens=2)
    chunks = chunker.chunk(_document(text))
    observed = {token for chunk in chunks for token in chunker.tokens(chunk)}
    assert observed == set(text.split())
    assert all(chunker.count_tokens(chunk) <= 10 for chunk in chunks)


def test_chunker_groups_small_paragraphs_until_a_real_section_boundary() -> None:
    document = _document("Alpha one.\n\nBeta two.\n\nGamma three.\n\nDelta four.")
    chunks = DeterministicSectionChunker(max_tokens=20, overlap_tokens=2).chunk(document)
    assert chunks == ("Alpha one.\n\nBeta two.\n\nGamma three.\n\nDelta four.",)


@pytest.mark.parametrize(
    ("maximum", "overlap"),
    [(True, 0), (0, 0), (4001, 0), (10, True), (10, -1), (10, 10)],
)
def test_chunker_rejects_unsafe_configuration(maximum: object, overlap: object) -> None:
    with pytest.raises(ValueError, match=r"chunk|maximum"):
        DeterministicSectionChunker(
            max_tokens=maximum,  # type: ignore[arg-type]
            overlap_tokens=overlap,  # type: ignore[arg-type]
        )


def test_chunker_rejects_invalid_document_and_chunk_count() -> None:
    with pytest.raises(ValueError, match="filing document"):
        DeterministicSectionChunker().chunk(object())  # type: ignore[arg-type]
    document = _document("Item 1\n\n" + " ".join(f"token{i}" for i in range(20)))
    with pytest.raises(ValueError, match="too many chunks"):
        DeterministicSectionChunker(
            max_tokens=4,
            overlap_tokens=1,
            max_chunks=1,
        ).chunk(document)
