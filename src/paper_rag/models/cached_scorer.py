from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_rag.domain import SearchHit


class CachedHGTScorer:
    """Online innovation-one scorer: Wq(query) against cached HGT node vectors."""

    def __init__(self, artifact_dir: str | Path, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install PyTorch in graph-env") from exc
        root = Path(artifact_dir)
        self.torch = torch
        self.device = device
        self.query_projector = torch.jit.load(str(root / "query_projector.pt"), map_location=device)
        self.query_projector.eval()
        self.node_ids = json.loads((root / "node_ids.json").read_text(encoding="utf-8"))
        matrix = np.load(root / "graph_embeddings.npy").astype(np.float32)
        metadata_path = root / "training.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {"hidden_dimension": 256}
        )
        if (
            matrix.ndim != 2
            or matrix.shape[1] != metadata["hidden_dimension"]
            or len(self.node_ids) != len(matrix)
        ):
            raise ValueError("Invalid HGT artifact dimensions")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.maximum(norms, 1e-12)
        self.positions = {node_id: index for index, node_id in enumerate(self.node_ids)}

    def __call__(self, query_vector: np.ndarray, hits: list[SearchHit]) -> dict[str, float]:
        tensor = self.torch.from_numpy(np.asarray(query_vector, dtype=np.float32)).to(self.device)
        with self.torch.no_grad():
            query = self.query_projector(tensor).float().cpu().numpy().reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        return {
            hit.node_id: float(self.matrix[self.positions[hit.node_id]] @ query)
            for hit in hits
            if hit.node_id in self.positions
        }
