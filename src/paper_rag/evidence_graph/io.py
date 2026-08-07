from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper_rag.domain import BoundingBox, EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.evidence_graph.graph import EvidenceGraph


def save_graph(graph: EvidenceGraph, path: str | Path) -> None:
    payload = {
        "schema_version": "1.0",
        "nodes": [node.to_dict() for node in graph.nodes.values()],
        "edges": [
            {
                "src": edge.src,
                "dst": edge.dst,
                "relation": edge.relation.value,
                "confidence": edge.confidence,
                "mandatory_for_closure": edge.mandatory_for_closure,
                "directional": edge.directional,
                "attributes": edge.attributes,
            }
            for edge in graph.edges
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph(path: str | Path) -> EvidenceGraph:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported graph schema: {payload.get('schema_version')}")
    graph = EvidenceGraph()
    for raw in payload.get("nodes", []):
        bbox = raw.get("bbox")
        graph.add_node(
            EvidenceNode(
                node_id=raw["node_id"],
                paper_id=raw["paper_id"],
                node_type=NodeType(raw["node_type"]),
                text=raw.get("text"),
                image_path=raw.get("image_path"),
                page=raw.get("page"),
                bbox=BoundingBox(*bbox) if bbox else None,
                parser_block_id=raw.get("parser_block_id"),
                confidence=float(raw.get("confidence", 1.0)),
                provenance=raw.get("provenance", {}),
                attributes=raw.get("attributes", {}),
            )
        )
    for raw in payload.get("edges", []):
        graph.add_edge(
            EvidenceEdge(
                src=raw["src"],
                dst=raw["dst"],
                relation=RelationType(raw["relation"]),
                confidence=float(raw.get("confidence", 1.0)),
                mandatory_for_closure=bool(raw.get("mandatory_for_closure", False)),
                directional=bool(raw.get("directional", True)),
                attributes=raw.get("attributes", {}),
            )
        )
    return graph

