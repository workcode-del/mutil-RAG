from __future__ import annotations

import json
import os
import base64
import mimetypes
from pathlib import Path

from paper_rag.domain import EvidenceForest, QuerySpec
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.generation.base import Answer
from paper_rag.generation.serializer import serialize_forest


class OpenAICompatibleGenerator:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "PAPER_RAG_API_KEY",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout

    def generate(
        self, query: QuerySpec, forest: EvidenceForest, graph: EvidenceGraph
    ) -> Answer:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install app dependencies for generation HTTP calls") from exc
        context, image_paths = serialize_forest(forest, graph)
        prompt = (
            "Answer only from the supplied evidence. Return JSON with keys answer and "
            "evidence_ids. Every atomic claim must cite existing bracketed evidence IDs.\n\n"
            f"Question: {query.query}\n\n{context}"
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            path = Path(image_path)
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
            )
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()
        content = raw["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        evidence_ids = [str(value) for value in parsed.get("evidence_ids", [])]
        invalid = set(evidence_ids) - forest.node_ids
        if invalid:
            raise ValueError(f"Generator cited evidence outside retrieved forest: {sorted(invalid)}")
        return Answer(str(parsed.get("answer", "")), evidence_ids, raw)
