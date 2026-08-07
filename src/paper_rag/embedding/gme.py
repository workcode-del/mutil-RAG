from __future__ import annotations

from typing import Sequence

import numpy as np


class GMEEmbedder:
    """Thin adapter over Alibaba-NLP GME official remote-code interface.

    Keep this class in embedding-env with transformers==4.51.3. The graph environment
    consumes cached arrays and never imports this model.
    """

    dimension = 1536

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        query_instruction: str = "Retrieve scientific evidence that answers the question.",
    ) -> None:
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install embedding-env dependencies before loading GME") from exc
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(device)
        self.model.eval()
        self.query_instruction = query_instruction

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.get_text_embeddings(
            texts=list(texts),
            instruction=self.query_instruction,
            is_query=True,
        )
        return self._numpy(vectors)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.get_text_embeddings(texts=list(texts), is_query=False)
        return self._numpy(vectors)

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        vectors = self.model.get_image_embeddings(images=list(image_paths), is_query=False)
        return self._numpy(vectors)

    @staticmethod
    def _numpy(vectors: object) -> np.ndarray:
        if hasattr(vectors, "detach"):
            vectors = vectors.detach().float().cpu().numpy()
        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != GMEEmbedder.dimension:
            raise ValueError(f"Expected [batch, 1536] GME embeddings, got {result.shape}")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-12)

