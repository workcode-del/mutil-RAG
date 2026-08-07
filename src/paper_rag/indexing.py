from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paper_rag.domain import EvidenceNode, NodeType
from paper_rag.embedding.base import Embedder
from paper_rag.evidence_graph import EvidenceGraph, build_figure_text_views


@dataclass(frozen=True, slots=True)
class IndexingReport:
    text_nodes: int
    figure_nodes: int
    dimension: int


def compute_base_embeddings(
    graph: EvidenceGraph, embedder: Embedder, batch_size: int = 16
) -> tuple[dict[str, np.ndarray], IndexingReport]:
    build_figure_text_views(graph)
    texts = [node for node in graph.nodes.values() if node.node_type is not NodeType.FIGURE]
    figures = [node for node in graph.nodes.values() if node.node_type is NodeType.FIGURE]
    result: dict[str, np.ndarray] = {}
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = embedder.embed_texts([node.searchable_text for node in batch])
        result.update({node.node_id: vector for node, vector in zip(batch, vectors, strict=True)})
    for start in range(0, len(figures), batch_size):
        batch = figures[start : start + batch_size]
        vectors = embedder.embed_images([node.image_path or "" for node in batch])
        result.update({node.node_id: vector for node, vector in zip(batch, vectors, strict=True)})
    return result, IndexingReport(len(texts), len(figures), embedder.dimension)


def upsert_base_embeddings(store, graph: EvidenceGraph, embeddings: dict[str, np.ndarray]) -> None:
    nodes: list[EvidenceNode] = []
    vectors: list[np.ndarray] = []
    for node_id, vector in embeddings.items():
        nodes.append(graph.nodes[node_id])
        vectors.append(vector)
    if vectors:
        store.ensure_collection()
        store.upsert(nodes, np.stack(vectors))
