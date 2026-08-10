from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paper_rag.model_source import resolve_model_reference


@dataclass(frozen=True, slots=True)
class ChartExtractionResult:
    linearized_table: str
    parse_status: str
    confidence: float | None = None
    uncertainty: float | None = None
    extractor: str = "unknown"


class PPChart2TableExtractor:
    """Local chart-to-table backend added to Transformers in 2026."""

    def __init__(
        self,
        model_name: str = "PaddlePaddle/PP-Chart2Table_safetensors",
        device: int | str = 0,
        model_source: str = "modelscope",
        local_path: str | Path | None = None,
        modelscope_id: str | None = None,
        model_cache_dir: str | Path = "data/models",
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the unified dependencies in the paper-rag environment") from exc
        resolved_model = resolve_model_reference(
            model_name,
            local_path=local_path,
            modelscope_id=modelscope_id,
            source=model_source,
            cache_dir=model_cache_dir,
        )
        self.pipe = pipeline("image-text-to-text", model=resolved_model, device=device)

    def extract(self, image_path: str | Path) -> ChartExtractionResult:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image", "url": str(Path(image_path).resolve())}],
            }
        ]
        output = self.pipe(text=conversation)
        generated = output[0].get("generated_text", "")
        if isinstance(generated, list):
            generated = generated[-1].get("content", "") if generated else ""
        table = str(generated).strip()
        status = "ok" if table else "suspect"
        return ChartExtractionResult(table, status, extractor="pp-chart2table")
