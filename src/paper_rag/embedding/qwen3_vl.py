from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np


class Qwen3VLEmbedder:
    """Adapter for the official Qwen3-VL-Embedding implementation.

    The upstream repository currently exposes its model from ``src.models`` rather
    than a stable top-level import.  ``official_repo`` makes that dependency
    explicit and prevents this project from copying or silently modifying upstream
    model code.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        official_repo: str | Path | None = None,
        dimension: int = 2048,
        max_length: int = 8192,
        query_instruction: str = "Retrieve scientific evidence that answers the question.",
        device: str = "cuda",
    ) -> None:
        if official_repo is not None:
            repo = str(Path(official_repo).resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)
        try:
            import torch
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder as OfficialEmbedder
        except ImportError as exc:  # pragma: no cover - model environment guard
            raise RuntimeError(
                "Clone https://github.com/QwenLM/Qwen3-VL-Embedding and set "
                "QWEN3_VL_RETRIEVAL_REPO to that directory"
            ) from exc

        self.dimension = dimension
        self.query_instruction = query_instruction
        kwargs = {
            "model_name_or_path": model_name,
            "max_length": max_length,
            "torch_dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
        }
        if device.startswith("cuda"):
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = OfficialEmbedder(**kwargs)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        items = [{"text": text, "instruction": self.query_instruction} for text in texts]
        return self._normalize(self.model.process(items))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._normalize(self.model.process([{"text": text} for text in texts]))

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        return self._normalize(self.model.process([{"image": path} for path in image_paths]))

    def embed_mixed(self, items: Sequence[dict]) -> np.ndarray:
        """Encode text-image mixtures for later fine-grained experiments."""
        return self._normalize(self.model.process(list(items)))

    def _normalize(self, vectors: object) -> np.ndarray:
        if hasattr(vectors, "detach"):
            vectors = vectors.detach().float().cpu().numpy()
        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] < self.dimension:
            raise ValueError(
                f"Expected [batch, >= {self.dimension}] Qwen3-VL embeddings, got {result.shape}"
            )
        # MRL permits using a prefix dimension; 2048 keeps the complete 2B representation.
        result = result[:, : self.dimension]
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-12)
