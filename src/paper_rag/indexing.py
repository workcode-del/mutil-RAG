from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from paper_rag.domain import EvidenceNode, NodeType
from paper_rag.embedding.base import Embedder
from paper_rag.evidence_graph import EvidenceGraph, build_figure_text_views


logger = logging.getLogger(__name__)


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
    logger.info(
        "Embedding nodes: text=%d figures=%d batch_size=%d", len(texts), len(figures), batch_size
    )
    _embed_batches(texts, batch_size, embedder.embed_texts, result, "text")
    _embed_batches(figures, batch_size, embedder.embed_images, result, "figure")
    return result, IndexingReport(len(texts), len(figures), embedder.dimension)


def _embed_batches(nodes, batch_size, encode, result, label: str) -> None:
    total = len(nodes)
    for start in range(0, total, batch_size):
        batch = nodes[start : start + batch_size]
        values = [node.image_path or "" for node in batch] if label == "figure" else [
            node.searchable_text for node in batch
        ]
        vectors = encode(values)
        result.update({node.node_id: vector for node, vector in zip(batch, vectors, strict=True)})
        done = min(start + batch_size, total)
        if done == total or done // max(1, total // 10) != start // max(1, total // 10):
            logger.info("Embedding progress: %s=%d/%d", label, done, total)


def upsert_base_embeddings(
    store, graph: EvidenceGraph, embeddings: dict[str, np.ndarray], batch_size: int = 512
) -> None:
    items = list(embeddings.items())
    if not items:
        return
    store.ensure_collection()
    logger.info("Vector upsert: nodes=%d batch_size=%d", len(items), batch_size)
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        store.upsert(
            [graph.nodes[node_id] for node_id, _ in batch],
            np.stack([vector for _, vector in batch]),
        )
        done = min(start + batch_size, len(items))
        if done == len(items) or done // max(1, len(items) // 10) != start // max(
            1, len(items) // 10
        ):
            logger.info("Vector upsert progress: %d/%d", done, len(items))
