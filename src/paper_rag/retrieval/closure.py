from __future__ import annotations

from dataclasses import dataclass

from paper_rag.domain import NodeType, RelationType
from paper_rag.evidence_graph import EvidenceGraph


@dataclass(frozen=True, slots=True)
class ClosurePolicy:
    figure_requires_caption: bool = True
    chart_requires_figure: bool = True
    sentence_reference_requires_figure: bool = True
    include_locator_nodes: bool = False


def evidence_closure(
    graph: EvidenceGraph,
    seed_ids: set[str],
    policy: ClosurePolicy = ClosurePolicy(),
) -> set[str]:
    """Compute the least fixed point of typed scientific-evidence dependencies."""
    unknown = seed_ids.difference(graph.nodes)
    if unknown:
        raise KeyError(f"Unknown evidence nodes: {sorted(unknown)}")
    closed = set(seed_ids)
    changed = True
    while changed:
        changed = False
        additions: set[str] = set()
        for node_id in closed:
            node = graph.nodes[node_id]
            incident = graph.incident_edges(node_id)
            if node.node_type is NodeType.FIGURE and policy.figure_requires_caption:
                additions.update(
                    edge.src
                    for edge in incident
                    if edge.relation is RelationType.CAPTION_OF and edge.dst == node_id
                )
            if node.node_type is NodeType.CHART_DATA and policy.chart_requires_figure:
                additions.update(
                    edge.dst
                    for edge in incident
                    if edge.relation is RelationType.DERIVED_FROM and edge.src == node_id
                )
            if node.node_type is NodeType.SENTENCE and policy.sentence_reference_requires_figure:
                additions.update(
                    edge.dst
                    for edge in incident
                    if edge.relation is RelationType.REFERS_TO and edge.src == node_id
                )
        before = len(closed)
        closed.update(additions)
        changed = len(closed) != before
    return closed


def validate_closure(graph: EvidenceGraph, node_ids: set[str], policy: ClosurePolicy) -> bool:
    return evidence_closure(graph, node_ids, policy) == node_ids

