from __future__ import annotations

from typing import Sequence

import numpy as np


class HTTPEmbedder:
    def __init__(self, base_url: str, dimension: int = 2048, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        self.timeout = timeout

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._post("query", {"values": list(texts)})

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._post("text", {"values": list(texts)})

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        return self._post("image", {"values": list(image_paths)})

    def _post(self, kind: str, payload: dict) -> np.ndarray:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install requests in the caller environment") from exc
        response = requests.post(
            f"{self.base_url}/embed/{kind}", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        vectors = np.asarray(response.json()["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Embedding service returned invalid shape {vectors.shape}")
        return vectors
