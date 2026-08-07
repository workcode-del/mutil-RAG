from __future__ import annotations

import pytest

from paper_rag.reranking.qwen3_vl import Qwen3VLReranker


def test_multimodal_document_normalization() -> None:
    assert Qwen3VLReranker._normalize_document("plain text") == {"text": "plain text"}
    assert Qwen3VLReranker._normalize_document(
        {"image": "figure.png", "text": "caption", "unused": None}
    ) == {"image": "figure.png", "text": "caption"}


def test_multimodal_document_requires_content() -> None:
    with pytest.raises(ValueError):
        Qwen3VLReranker._normalize_document({"text": "", "image": None})
