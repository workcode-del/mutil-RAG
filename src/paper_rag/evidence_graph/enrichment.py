from __future__ import annotations

from paper_rag.domain import EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.evidence_graph.graph import EvidenceGraph


def attach_chart_data(
    graph: EvidenceGraph,
    figure_id: str,
    linearized_table: str,
    parse_status: str,
    confidence: float = 0.5,
    extractor: str = "unknown",
    uncertainty: float | None = None,
) -> str:
    figure = graph.nodes[figure_id]
    if figure.node_type is not NodeType.FIGURE:
        raise ValueError("ChartData can only be derived from a Figure")
    node_id = f"{figure.paper_id}:chart_data:{figure_id.rsplit(':', 1)[-1]}"
    node = EvidenceNode(
        node_id=node_id,
        paper_id=figure.paper_id,
        node_type=NodeType.CHART_DATA,
        text=linearized_table,
        page=figure.page,
        bbox=figure.bbox,
        confidence=confidence,
        provenance={
            "extractor": extractor,
            "derived": True,
            "parse_status": parse_status,
            "uncertainty": uncertainty,
        },
    )
    graph.add_node(node)
    graph.add_edge(
        EvidenceEdge(
            node_id,
            figure_id,
            RelationType.DERIVED_FROM,
            confidence=confidence,
            mandatory_for_closure=True,
        )
    )
    return node_id


def build_figure_text_views(graph: EvidenceGraph) -> None:
    """Serialize linked text for a text-only reranker without replacing image evidence."""
    for node in graph.nodes.values():
        if node.node_type is not NodeType.FIGURE:
            continue
        parts: list[str] = []
        for edge in graph.incident_edges(node.node_id):
            other_id: str | None = None
            if edge.relation is RelationType.CAPTION_OF and edge.dst == node.node_id:
                other_id = edge.src
            elif edge.relation is RelationType.REFERS_TO and edge.dst == node.node_id:
                other_id = edge.src
            elif edge.relation is RelationType.DERIVED_FROM and edge.dst == node.node_id:
                other_id = edge.src
            if other_id:
                text = graph.nodes[other_id].searchable_text.strip()
                if text:
                    parts.append(text)
        node.attributes["text_view"] = "\n".join(dict.fromkeys(parts))
