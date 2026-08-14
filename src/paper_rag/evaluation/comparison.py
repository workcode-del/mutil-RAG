from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_METRICS = (
    "macro_mrr",
    "macro_recall_at_10",
    "macro_ndcg_at_10",
    "macro_joint_recall_at_10",
    "macro_evidence_f1",
    "micro_sentence_evidence_f1",
    "micro_figure_evidence_f1",
    "macro_closure_validity",
    "macro_budget_violation",
    "macro_evidence_cost",
    "macro_retrieval_latency_ms",
    "macro_query_embedding_amortized_ms",
    "macro_latency_ms",
)


def save_comparison(
    reports: Iterable[str | Path],
    output: str | Path,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> tuple[Path, str]:
    metric_names = tuple(metrics)
    rows = [_report_row(Path(path), metric_names) for path in reports]
    fields = [
        "run",
        "candidate_backend",
        "dense_search_backend",
        "retrieval_method",
        "reranker",
        "hgt",
        "latency_mode",
        *metric_names,
    ]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target.resolve(), _markdown_table(rows, fields)


def _report_row(path: Path, metrics: tuple[str, ...]) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    summary = report.get("summary", {})
    return {
        "run": path.stem,
        "candidate_backend": metadata.get("candidate_backend", ""),
        "dense_search_backend": metadata.get("dense_search_backend", ""),
        "retrieval_method": metadata.get("retrieval_method", ""),
        "reranker": metadata.get("reranker", ""),
        "hgt": metadata.get("hgt", ""),
        "latency_mode": metadata.get("latency_mode", ""),
        **{metric: summary.get(metric, "") for metric in metrics},
    }


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    display = fields
    lines = [
        "| " + " | ".join(display) + " |",
        "| " + " | ".join("---" for _ in display) + " |",
    ]
    for row in rows:
        values = [_format(row.get(field, "")) for field in display]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)
