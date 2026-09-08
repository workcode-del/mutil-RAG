from __future__ import annotations

import base64
import mimetypes
import os
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
        extra_body: dict | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.extra_body = extra_body or {}

    def generate(
        self, query: QuerySpec, forest: EvidenceForest, graph: EvidenceGraph
    ) -> Answer:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install app dependencies for generation HTTP calls") from exc
        context, image_paths = serialize_forest(forest, graph)
        prompt = (
            "Answer the question using only the supplied evidence. Give only the answer, "
            "without discussing these instructions.\n\n"
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
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            **self.extra_body,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()
        return Answer(_response_text(raw), None, raw)


def _response_text(response: dict) -> str:
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Invalid OpenAI chat completion response") from exc
    if not isinstance(message, dict):
        raise ValueError("Invalid OpenAI chat completion response")
    for field in ("content", "reasoning_content", "reasoning"):
        text = _message_text(message.get(field))
        if text:
            return text
    raise ValueError(
        "OpenAI chat completion returned no text in content, reasoning_content, or reasoning; "
        f"finish_reason={choice.get('finish_reason')!r}"
    )


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(_message_text(part) for part in value).strip()
    if isinstance(value, dict):
        for field in ("text", "content", "value"):
            text = _message_text(value.get(field))
            if text:
                return text
    return ""
