"""Provider-neutral, citation-grounded research contracts.

This package deliberately contains no network, model, or vector-database adapter.
Production research stays unavailable until those dependencies are explicitly
configured and approved.
"""

from app.research.domain import (
    CorpusDescriptor,
    EmbeddedChunk,
    EmbeddingDescriptor,
    EmbeddingVector,
    EvidenceChunk,
    EvidenceQuote,
    FilingDiscoveryCursor,
    FilingDiscoveryPage,
    FilingDiscoveryRequest,
    FilingDocument,
    FilingReference,
    GeneratedAnswer,
    GeneratedClaim,
    GenerationManifest,
    GenerationVerification,
    IngestionResult,
    RawFiling,
    ResearchAnswer,
    ResearchCitation,
    ResearchOutcome,
    SearchHit,
)
from app.research.service import (
    ResearchCoreService,
    ResearchCorpusUnavailableError,
    ResearchPolicy,
)

__all__ = [
    "CorpusDescriptor",
    "EmbeddedChunk",
    "EmbeddingDescriptor",
    "EmbeddingVector",
    "EvidenceChunk",
    "EvidenceQuote",
    "FilingDiscoveryCursor",
    "FilingDiscoveryPage",
    "FilingDiscoveryRequest",
    "FilingDocument",
    "FilingReference",
    "GeneratedAnswer",
    "GeneratedClaim",
    "GenerationManifest",
    "GenerationVerification",
    "IngestionResult",
    "RawFiling",
    "ResearchAnswer",
    "ResearchCitation",
    "ResearchCoreService",
    "ResearchCorpusUnavailableError",
    "ResearchOutcome",
    "ResearchPolicy",
    "SearchHit",
]
