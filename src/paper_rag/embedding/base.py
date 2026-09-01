from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dimension: int

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray: ...

    def embed_mixed(self, items: Sequence[dict]) -> np.ndarray: ...
