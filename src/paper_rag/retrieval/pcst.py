from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paper_rag.domain import RelationType
from paper_rag.evidence_graph import EvidenceGraph


@dataclass(frozen=True, slots=True)
class PCSTResult:
    node_ids: set[str]
    edge_pairs: set[tuple[str, str]]
    backend: str


DEFAULT_RELATION_COSTS = {
    RelationType.CAPTION_OF: 0.10,
    RelationType.REFERS_TO: 0.20,
    RelationType.DERIVED_FROM: 0.10,
    RelationType.NEXT_SENTENCE: 0.50,
    RelationType.CONTAINS: 0.30,
    RelationType.SEMANTICALLY_SIMILAR: 1.20,
}


def solve_pcst(
    graph: EvidenceGraph,
    prizes: dict[str, float],
    relation_costs: dict[RelationType, float] | None = None,
    cost_scale: float = 1.0,
    allow_fallback: bool = True,
) -> PCSTResult:
    """Convert the directed typed evidence graph to pcst_fast's undirected view."""
    relation_costs = relation_costs or DEFAULT_RELATION_COSTS
    node_ids = sorted(graph.nodes)
    if not node_ids:
        return PCSTResult(set(), set(), "empty")
    index = {node_id: position for position, node_id in enumerate(node_ids)}

    # Merge reverse typed edges only for optimization. Original graph remains unchanged.
    cheapest_edges: dict[tuple[int, int], float] = {}
    for edge in graph.edges:
        left, right = sorted((index[edge.src], index[edge.dst]))
        if left == right:
            continue
        cost = relation_costs.get(edge.relation, 1.0) * cost_scale
        pair = (left, right)
        cheapest_edges[pair] = min(cheapest_edges.get(pair, float("inf")), cost)
    edge_pairs = list(cheapest_edges)
    edge_array = np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2)
    costs = np.asarray([cheapest_edges[pair] for pair in edge_pairs], dtype=np.float64)
    node_prizes = np.asarray([max(0.0, prizes.get(node_id, 0.0)) for node_id in node_ids])

    try:
        from pcst_fast import pcst_fast
    except ImportError:
        if not allow_fallback:
            raise RuntimeError("pcst_fast is required in graph-env")
        return _positive_prize_fallback(graph, prizes)

    vertices, selected_edge_indices = pcst_fast(
        edge_array,
        node_prizes.astype(np.float64),
        costs,
        -1,  # unrooted
        1,
        "gw",
        0,
    )
    selected_nodes = {node_ids[int(position)] for position in vertices}
    selected_edges = {
        (node_ids[edge_pairs[int(i)][0]], node_ids[edge_pairs[int(i)][1]])
        for i in selected_edge_indices
    }
    return PCSTResult(selected_nodes, selected_edges, "pcst_fast")


def _positive_prize_fallback(graph: EvidenceGraph, prizes: dict[str, float]) -> PCSTResult:
    """Availability fallback, deliberately not reported as the PCST experimental baseline."""
    positive = {node_id for node_id, prize in prizes.items() if prize > 0 and node_id in graph.nodes}
    if not positive and prizes:
        positive = {max(prizes, key=prizes.get)}
    edges = {
        tuple(sorted((edge.src, edge.dst)))
        for edge in graph.edges
        if edge.src in positive and edge.dst in positive
    }
    return PCSTResult(positive, edges, "positive_prize_fallback")

