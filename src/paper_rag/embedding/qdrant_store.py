from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from paper_rag.domain import EvidenceNode, NodeType, SearchHit


class QdrantEvidenceStore:
    def __init__(
        self,
        collection: str = "scientific_evidence",
        dimension: int = 2048,
        path: str | Path | None = "data/index/qdrant",
        url: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install qdrant-client in vector/graph environment") from exc
        if bool(path) == bool(url):
            raise ValueError("Provide exactly one of path or url")
        self.models = models
        self.client = QdrantClient(url=url) if url else QdrantClient(path=str(path))
        self.collection = collection
        self.dimension = dimension

    def ensure_collection(self) -> None:
        models = self.models
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(self, nodes: Sequence[EvidenceNode], vectors: np.ndarray) -> None:
        if vectors.shape != (len(nodes), self.dimension):
            raise ValueError(
                f"Expected vectors shape {(len(nodes), self.dimension)}, got {vectors.shape}"
            )
        points = []
        for node, vector in zip(nodes, vectors, strict=True):
            # Qdrant point IDs are UUID/int. Stable UUID5 keeps EvidenceNode IDs in payload.
            import uuid

            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, node.node_id))
            points.append(
                self.models.PointStruct(
                    id=point_id,
                    vector=vector.astype(float).tolist(),
                    payload={
                        "node_id": node.node_id,
                        "paper_id": node.paper_id,
                        "node_type": node.node_type.value,
                        "page": node.page,
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(
        self,
        query_vector: np.ndarray,
        node_types: Iterable[NodeType],
        per_type_top_k: int = 25,
    ) -> list[SearchHit]:
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Query dimension must be {self.dimension}")
        hits: list[SearchHit] = []
        for node_type in node_types:
            query_filter = self.models.Filter(
                must=[
                    self.models.FieldCondition(
                        key="node_type", match=self.models.MatchValue(value=node_type.value)
                    )
                ]
            )
            # query_points is the current client API; adapting future changes stays here.
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector.tolist(),
                query_filter=query_filter,
                limit=per_type_top_k,
                with_payload=True,
            )
            for point in response.points:
                payload = point.payload or {}
                hits.append(
                    SearchHit(
                        node_id=str(payload["node_id"]),
                        paper_id=str(payload["paper_id"]),
                        node_type=NodeType(str(payload["node_type"])),
                        score=float(point.score),
                        score_components={"embedding": float(point.score)},
                    )
                )
        return sorted(hits, key=lambda item: item.score, reverse=True)
