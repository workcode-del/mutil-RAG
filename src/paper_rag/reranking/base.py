from __future__ import annotations

from typing import Mapping, Protocol, Sequence


RerankDocument = str | Mapping[str, object]


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[RerankDocument]) -> list[float]: ...


class NoOpReranker:
    def score(self, query: str, documents: Sequence[RerankDocument]) -> list[float]:
        return [0.0] * len(documents)
