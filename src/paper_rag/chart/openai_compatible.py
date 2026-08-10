from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from .pp_chart2table import ChartExtractionResult


class OpenAICompatibleChartExtractor:
    """Convert a line-chart image to CSV through a multimodal chat API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "PAPER_RAG_API_KEY",
        timeout: float = 120.0,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.temperature = temperature

    def extract(self, image_path: str | Path) -> ChartExtractionResult:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the unified dependencies before chart extraction") from exc

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Chart image does not exist: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = (
            "This image is a scientific line chart. Convert every visible line series to CSV. "
            "Preserve axis labels, units, legend names, and numeric values. Output CSV only, "
            "without Markdown fences or explanation."
        )
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            },
        ]
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
                "temperature": self.temperature,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        table = str(response.json()["choices"][0]["message"]["content"]).strip()
        if table.startswith("```"):
            lines = table.splitlines()
            table = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else ""
            if table.lower().startswith("csv\n"):
                table = table[4:].strip()
        return ChartExtractionResult(
            linearized_table=table,
            parse_status="ok" if table else "suspect",
            extractor="openai-compatible-vlm",
        )
