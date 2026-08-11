from __future__ import annotations

from typing import Protocol

from paper_rag.domain import EvidenceForest, QuerySpec, SearchHit


class EvidenceRetriever(Protocol):
    """Rank candidates and select the final evidence with one shared pipeline contract."""

    def rank_hits(self, query: QuerySpec, hits: list[SearchHit]) -> list[SearchHit]: ...

    def retrieve(self, query: QuerySpec, hits: list[SearchHit]) -> EvidenceForest: ...
