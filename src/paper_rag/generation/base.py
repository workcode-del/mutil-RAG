from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from paper_rag.domain import EvidenceForest, QuerySpec
from paper_rag.evidence_graph import EvidenceGraph


@dataclass(slots=True)
class Answer:
    text: str
    evidence_ids: list[str] | None = None
    raw: dict = field(default_factory=dict)


class AnswerGenerator(Protocol):
    def generate(
        self, query: QuerySpec, forest: EvidenceForest, graph: EvidenceGraph
    ) -> Answer: ...
