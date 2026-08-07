from __future__ import annotations

import csv
import io
import statistics
from collections.abc import Callable
from pathlib import Path

from .pp_chart2table import ChartExtractionResult


def _parse_rows(value: str) -> list[list[str]]:
    lines = [line.strip().strip("|") for line in value.splitlines() if line.strip()]
    if not lines:
        return []
    delimiter = "\t" if any("\t" in line for line in lines) else ("|" if any("|" in line for line in lines) else ",")
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
    return [[cell.strip() for cell in row] for row in rows if not all(set(cell) <= {"-", ":", " "} for cell in row)]


def _numeric(value: str) -> float | None:
    try:
        return float(value.replace(",", "").rstrip("%"))
    except ValueError:
        return None


class SelfEnsemblingChartExtractor:
    """Training-free repeated extraction with cell-wise median and uncertainty.

    It follows the 2026 self-ensembling idea while keeping the backend injectable;
    the callable may use Qwen3-VL, PP-Chart2Table, or an external multimodal API.
    """

    def __init__(self, sample: Callable[[str | Path], ChartExtractionResult], repeats: int = 3):
        if repeats < 2:
            raise ValueError("Self-ensembling requires at least two samples")
        self.sample = sample
        self.repeats = repeats

    def extract(self, image_path: str | Path) -> ChartExtractionResult:
        samples = [self.sample(image_path) for _ in range(self.repeats)]
        tables = [_parse_rows(sample.linearized_table) for sample in samples]
        shapes = [(len(table), max((len(row) for row in table), default=0)) for table in tables]
        target_shape = statistics.mode(shapes)
        aligned = [table for table, shape in zip(tables, shapes, strict=True) if shape == target_shape]
        if not aligned or target_shape == (0, 0):
            return ChartExtractionResult("", "suspect", 0.0, 1.0, "self-ensemble")

        rows, columns = target_shape
        merged: list[list[str]] = []
        disagreements = 0
        comparable = 0
        for row_index in range(rows):
            row: list[str] = []
            for column_index in range(columns):
                values = [
                    table[row_index][column_index] if column_index < len(table[row_index]) else ""
                    for table in aligned
                ]
                numbers = [_numeric(value) for value in values]
                present_numbers = [value for value in numbers if value is not None]
                if len(present_numbers) == len(values):
                    median = statistics.median(present_numbers)
                    row.append(f"{median:g}")
                    comparable += 1
                    disagreements += int(max(present_numbers) != min(present_numbers))
                else:
                    row.append(statistics.mode(values))
            merged.append(row)

        uncertainty = disagreements / comparable if comparable else 0.0
        buffer = io.StringIO()
        csv.writer(buffer).writerows(merged)
        confidence = max(0.0, 1.0 - uncertainty)
        status = "ok" if confidence >= 0.67 else "suspect"
        return ChartExtractionResult(
            buffer.getvalue().strip(), status, confidence, uncertainty, "self-ensemble"
        )
