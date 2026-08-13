from __future__ import annotations

import uuid
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
        upload_batch_size: int = 4096,
        upload_parallel: int = 1,
        prefer_grpc: bool = False,
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install qdrant-client in vector/graph environment") from exc
        if bool(path) == bool(url):
            raise ValueError("Provide exactly one of path or url")
        self.models = models
        self.client = (
            QdrantClient(url=url, prefer_grpc=prefer_grpc)
            if url
            else QdrantClient(path=str(path))
        )
        self.collection = collection
        self.dimension = dimension
        self.upload_batch_size = upload_batch_size
        self.upload_parallel = upload_parallel

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
        self.client.upload_collection(
            collection_name=self.collection,
            vectors=vectors,
            ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, node.node_id)) for node in nodes],
            payload=[
                {
                    "node_id": node.node_id,
                    "paper_id": node.paper_id,
                    "node_type": node.node_type.value,
                    "page": node.page,
                }
                for node in nodes
            ],
            batch_size=self.upload_batch_size,
            parallel=self.upload_parallel,
            wait=True,
        )

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
            raise ValueError("Qdrant search requires a query vector")
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Query dimension must be {self.dimension}")
        hits: list[SearchHit] = []
        for node_type in node_types:
            conditions = [
                self.models.FieldCondition(
                    key="node_type", match=self.models.MatchValue(value=node_type.value)
                )
            ]
            if paper_ids:
                conditions.append(
                    self.models.FieldCondition(
                        key="paper_id",
                        match=self.models.MatchAny(any=sorted(paper_ids)),
                    )
                )
            if candidate_node_ids:
                conditions.append(
                    self.models.FieldCondition(
                        key="node_id",
                        match=self.models.MatchAny(any=sorted(candidate_node_ids)),
                    )
                )
            query_filter = self.models.Filter(
                must=conditions
            )
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
