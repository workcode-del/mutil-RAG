from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dimension: int

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray: ...

    def embed_mixed(self, items: Sequence[dict]) -> np.ndarray: ...


def deterministic_mock_embedding(value: str, dimension: int = 2048) -> np.ndarray:
    """Repeatable normalized vector for interface tests; never use in experiments."""
    seed = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")
    vector = np.random.default_rng(seed).standard_normal(dimension).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class MockEmbedder:
    def __init__(self, dimension: int = 2048) -> None:
        self.dimension = dimension

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([deterministic_mock_embedding(f"query:{x}", self.dimension) for x in texts])

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([deterministic_mock_embedding(f"text:{x}", self.dimension) for x in texts])

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        return np.stack(
            [deterministic_mock_embedding(f"image:{x}", self.dimension) for x in image_paths]
        )

    def embed_mixed(self, items: Sequence[dict]) -> np.ndarray:
        return np.stack(
            [deterministic_mock_embedding(f"mixed:{item!r}", self.dimension) for item in items]
        )
