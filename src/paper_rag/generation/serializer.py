from __future__ import annotations

from paper_rag.domain import EvidenceForest, NodeType
from paper_rag.evidence_graph import EvidenceGraph


def serialize_forest(forest: EvidenceForest, graph: EvidenceGraph) -> tuple[str, list[str]]:
    """Create auditable text context and return image paths separately."""
    lines: list[str] = []
    image_paths: list[str] = []
    for tree_index, tree in enumerate(forest.trees, start=1):
        lines.append(f"## Evidence component {tree_index}; paper={tree.paper_id}")
        ordered = sorted(
            tree.node_ids,
            key=lambda node_id: (
                graph.nodes[node_id].page or 10**9,
                graph.nodes[node_id].node_type.value,
                node_id,
            ),
        )
        for node_id in ordered:
            node = graph.nodes[node_id]
            location = f"page={node.page or 'unknown'}"
            if node.node_type is NodeType.FIGURE:
                image_paths.append(node.image_path or "")
                text_view = node.attributes.get("text_view", "")
                lines.append(f"[{node_id}] Figure ({location}); text_view={text_view}")
            else:
                lines.append(f"[{node_id}] {node.node_type.value} ({location}): {node.text}")
    return "\n".join(lines), [path for path in image_paths if path]

