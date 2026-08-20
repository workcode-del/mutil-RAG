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
    table_nodes: int = 0


def compute_base_embeddings(
    graph: EvidenceGraph,
    embedder: Embedder,
    batch_size: int = 16,
    image_batch_size: int | None = None,
) -> tuple[dict[str, np.ndarray], IndexingReport]:
    build_figure_text_views(graph)
    tables = [node for node in graph.nodes.values() if node.node_type is NodeType.TABLE]
    texts = [
        node
        for node in graph.nodes.values()
        if node.node_type not in {NodeType.FIGURE, NodeType.TABLE}
    ]
    figures = [node for node in graph.nodes.values() if node.node_type is NodeType.FIGURE]
    result: dict[str, np.ndarray] = {}
    image_batch_size = image_batch_size or batch_size
    logger.info(
        "Embedding nodes: text=%d tables=%d figures=%d text_batch=%d image_batch=%d",
        len(texts),
        len(tables),
        len(figures),
        batch_size,
        image_batch_size,
    )
    _embed_batches(texts, batch_size, embedder.embed_texts, result, "text")
    _embed_tables(tables, batch_size, embedder, result)
    _embed_batches(figures, image_batch_size, embedder.embed_images, result, "figure")
    return result, IndexingReport(len(texts), len(figures), embedder.dimension, len(tables))


def _embed_tables(nodes, batch_size: int, embedder: Embedder, result) -> None:
    for start in range(0, len(nodes), batch_size):
        batch = nodes[start : start + batch_size]
        items = [
            {
                **({"text": node.searchable_text} if node.searchable_text else {}),
                **({"image": node.image_path} if node.image_path else {}),
            }
            for node in batch
        ]
        if hasattr(embedder, "embed_mixed"):
            vectors = embedder.embed_mixed(items)
        else:  # Compatibility for third-party text/image embedders.
            vectors = np.stack(
                [
                    embedder.embed_texts([node.searchable_text])[0]
                    if node.searchable_text
                    else embedder.embed_images([node.image_path or ""])[0]
                    for node in batch
                ]
            )
        result.update({node.node_id: vector for node, vector in zip(batch, vectors, strict=True)})


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
    store, graph: EvidenceGraph, embeddings: dict[str, np.ndarray], batch_size: int = 4096
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
