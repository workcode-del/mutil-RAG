from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Sequence


MultimodalDocument = str | Mapping[str, object]


class Qwen3VLReranker:
    """Pointwise multimodal reranker backed by the official Qwen implementation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Reranker-2B",
        official_repo: str | Path | None = None,
        max_length: int = 8192,
        device: str = "cuda",
        instruction: str = "Retrieve scientific evidence that answers the query.",
    ) -> None:
        if official_repo is not None:
            repo = str(Path(official_repo).resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)
        try:
            import torch
            from src.models.qwen3_vl_reranker import Qwen3VLReranker as OfficialReranker
        except ImportError as exc:  # pragma: no cover - model environment guard
            raise RuntimeError(
                "Clone https://github.com/QwenLM/Qwen3-VL-Embedding and set "
                "QWEN3_VL_RETRIEVAL_REPO to that directory"
            ) from exc
        kwargs = {
            "model_name_or_path": model_name,
            "max_length": max_length,
            "torch_dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
        }
        if device.startswith("cuda"):
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = OfficialReranker(**kwargs)
        self.instruction = instruction

    def score(self, query: str, documents: Sequence[MultimodalDocument]) -> list[float]:
        normalized = [self._normalize_document(document) for document in documents]
        values = self.model.process(
            {
                "instruction": self.instruction,
                "query": {"text": query},
                "documents": normalized,
            }
        )
        return [float(value) for value in values]

    @staticmethod
    def _normalize_document(document: MultimodalDocument) -> dict[str, object]:
        if isinstance(document, str):
            return {"text": document}
        result = {key: value for key, value in document.items() if value not in (None, "")}
        if not result or not ({"text", "image"} & result.keys()):
            raise ValueError("A rerank document must contain text and/or image")
        return result
