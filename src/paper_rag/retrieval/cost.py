from __future__ import annotations

import re
from dataclasses import dataclass

from paper_rag.domain import NodeType
from paper_rag.evidence_graph import EvidenceGraph


def estimate_text_tokens(text: str) -> int:
    """Stable model-independent proxy; actual generator usage must be logged separately."""
    latin_tokens = len(re.findall(r"[A-Za-z0-9_]+|[^\x00-\x7F]", text))
    return max(1, latin_tokens)


@dataclass(frozen=True, slots=True)
class CostModel:
    image_unit: int = 512

    def node_cost(self, graph: EvidenceGraph, node_id: str) -> int:
        node = graph.nodes[node_id]
        if node.node_type is NodeType.FIGURE:
            return self.image_unit
        return estimate_text_tokens(node.searchable_text)

    def set_cost(self, graph: EvidenceGraph, node_ids: set[str]) -> int:
        return sum(self.node_cost(graph, node_id) for node_id in node_ids)

