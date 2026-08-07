from __future__ import annotations

from typing import Sequence

from .base import RerankDocument


class HTTPReranker:
    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def score(self, query: str, documents: Sequence[RerankDocument]) -> list[float]:
        import requests

        response = requests.post(
            f"{self.base_url}/rerank",
            json={"query": query, "documents": list(documents)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        scores = [float(value) for value in response.json()["scores"]]
        if len(scores) != len(documents):
            raise ValueError("Reranker service returned the wrong number of scores")
        return scores
