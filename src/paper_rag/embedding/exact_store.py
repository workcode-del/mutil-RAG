from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from paper_rag.domain import EvidenceNode, NodeType, SearchHit
from paper_rag.evidence_graph import EvidenceGraph


class ExactEmbeddingStore:
    """Exact cosine search over cached benchmark embeddings."""

    def __init__(self, graph: EvidenceGraph, embeddings: Mapping[str, np.ndarray]) -> None:
        self.nodes: tuple[EvidenceNode, ...] = tuple(graph.nodes.values())
        self.positions = {node.node_id: index for index, node in enumerate(self.nodes)}
        self.vectors = np.stack([embeddings[node.node_id] for node in self.nodes]).astype(
            np.float32, copy=False
        )
        self.vectors /= np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)
        grouped: dict[tuple[str, NodeType], list[int]] = defaultdict(list)
        by_type: dict[NodeType, list[int]] = defaultdict(list)
        for index, node in enumerate(self.nodes):
            grouped[(node.paper_id, node.node_type)].append(index)
            by_type[node.node_type].append(index)
        self.by_paper_type = {key: np.asarray(value) for key, value in grouped.items()}
        self.by_type = {key: np.asarray(value) for key, value in by_type.items()}

    @classmethod
    def from_npz(cls, graph: EvidenceGraph, path: str | Path) -> "ExactEmbeddingStore":
        with np.load(path) as embeddings:
            return cls(graph, embeddings)

    def search(
        self,
        query: str,
        query_vector: np.ndarray | None,
        node_types: Iterable[NodeType],
        per_type_top_k: int = 25,
        paper_ids: set[str] | None = None,
        candidate_node_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        if query_vector is None:
            raise ValueError("Embedding search requires a query vector")
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        hits: list[SearchHit] = []
        for node_type in node_types:
            positions = self._scope(node_type, paper_ids, candidate_node_ids)
            if not len(positions):
                continue
            scores = self.vectors[positions] @ vector
            for offset in np.argsort(-scores, kind="stable")[:per_type_top_k]:
                node = self.nodes[int(positions[offset])]
                score = float(scores[offset])
                hits.append(
                    SearchHit(
                        node.node_id,
                        node.paper_id,
                        node.node_type,
                        score,
                        {"embedding": score},
                    )
                )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)

    def _scope(
        self,
        node_type: NodeType,
        paper_ids: set[str] | None,
        candidate_node_ids: set[str] | None,
    ) -> np.ndarray:
        if candidate_node_ids:
            return np.asarray(
                [
                    self.positions[node_id]
                    for node_id in sorted(candidate_node_ids)
                    if node_id in self.positions
                    and self.nodes[self.positions[node_id]].node_type is node_type
                    and (
                        not paper_ids
                        or self.nodes[self.positions[node_id]].paper_id in paper_ids
                    )
                ]
            )
        if paper_ids:
            groups = [
                self.by_paper_type[(paper_id, node_type)]
                for paper_id in sorted(paper_ids)
                if (paper_id, node_type) in self.by_paper_type
            ]
            return np.concatenate(groups) if groups else np.asarray([], dtype=np.int64)
        return self.by_type.get(node_type, np.asarray([], dtype=np.int64))
