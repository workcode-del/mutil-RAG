from __future__ import annotations

from pathlib import Path

from .pp_chart2table import ChartExtractionResult


DePlotResult = ChartExtractionResult


class DePlotExtractor:
    prompt = "Generate underlying data table of the figure below:"

    def __init__(self, model_name: str = "google/deplot", device: str = "cuda") -> None:
        try:
            from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install embedding-env dependencies before loading DePlot") from exc
        self.processor = Pix2StructProcessor.from_pretrained(model_name)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def extract(self, image_path: str | Path, max_new_tokens: int = 512) -> DePlotResult:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install Pillow in embedding-env") from exc
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        table = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        status = "ok" if table and "<0x" not in table else "suspect"
        return DePlotResult(table, status, extractor="deplot")
