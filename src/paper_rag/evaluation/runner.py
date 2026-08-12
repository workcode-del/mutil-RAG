from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from paper_rag.domain import NodeType, QuerySpec
from paper_rag.evaluation.metrics import result_metrics, serialize_hit, summarize
from paper_rag.pipeline import ScientificRAGPipeline


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    query_id: str
    query: QuerySpec
    relevant_node_ids: set[str]
    paper_ids: set[str]
    candidate_node_ids: set[str] = field(default_factory=set)
    answer: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "EvaluationSample":
        gold = {str(value) for value in data.get("relevant_node_ids", [])}
        if not gold:
            raise ValueError(f"Sample {index} has no relevant_node_ids")
        paper_ids = {str(value) for value in data.get("paper_ids", [])}
        if data.get("paper_id") is not None:
            paper_ids.add(str(data["paper_id"]))
        return cls(
            query_id=str(data.get("query_id", index)),
            query=QuerySpec(
                query=str(data["query"]),
                answer_type=str(data.get("answer_type", "free_text")),
                entity_type=data.get("entity_type"),
                metric=data.get("metric"),
                operator=data.get("operator"),
                value=data.get("value"),
                unit=data.get("unit"),
                conditions=[str(value) for value in data.get("conditions", [])],
                required_modalities=[
                    str(value) for value in data.get("required_modalities", ["text", "figure"])
                ],
            ),
            relevant_node_ids=gold,
            paper_ids=paper_ids,
            candidate_node_ids={str(value) for value in data.get("candidate_node_ids", [])},
            answer=str(data["answer"]) if data.get("answer") is not None else None,
        )


def load_samples(path: str | Path) -> list[EvaluationSample]:
    with Path(path).open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    return [EvaluationSample.from_dict(row, index) for index, row in enumerate(rows)]


def evaluate(
    pipeline: ScientificRAGPipeline,
    samples: list[EvaluationSample],
    *,
    cutoffs: tuple[int, ...] = (1, 3, 5, 10),
    per_type_top_k: int | None = None,
    scope_to_sample_papers: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph = pipeline.graph
    details: list[dict[str, Any]] = []
    total = len(samples)
    logger.info("Evaluation start: samples=%d", total)
    for index, sample in enumerate(samples, 1):
        unknown = sample.relevant_node_ids - graph.nodes.keys()
        if sample.candidate_node_ids:
            unknown.update(sample.candidate_node_ids - graph.nodes.keys())
            unknown.update(sample.relevant_node_ids - sample.candidate_node_ids)
        if unknown:
            raise KeyError(f"Sample {sample.query_id} has invalid evidence IDs: {sorted(unknown)}")
        started = perf_counter()
        result = pipeline.run(
            sample.query,
            per_type_top_k=per_type_top_k,
            paper_ids=sample.paper_ids if scope_to_sample_papers and sample.paper_ids else None,
            candidate_node_ids=sample.candidate_node_ids or None,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        metrics = result_metrics(
            graph,
            result,
            sample.relevant_node_ids,
            cutoffs=cutoffs,
            latency_ms=latency_ms,
            reference_answer=sample.answer,
        )
        details.append(
            {
                "query_id": sample.query_id,
                "metrics": metrics,
                "gold_node_ids": sorted(sample.relevant_node_ids),
                "ranked_hits": [serialize_hit(hit) for hit in result.hits],
                "selected_node_ids": sorted(result.forest.node_ids),
                "answer": result.answer.text if result.answer else None,
                "evidence_ids": result.answer.evidence_ids if result.answer else [],
            }
        )
        if index == total or index % 50 == 0:
            logger.info("Evaluation progress: %d/%d", index, total)
    return {
        "metadata": metadata or {},
        "summary": {
            "queries": len(details),
            **summarize(row["metrics"] for row in details),
            **micro_evidence_metrics(graph, details),
        },
        "details": details,
    }


def micro_evidence_metrics(
    graph, details: list[dict[str, Any]]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    node_types = (NodeType.SENTENCE, NodeType.FIGURE, NodeType.CAPTION, NodeType.CHART_DATA)
    for node_type in (None, *node_types):
        gold = {
            (row["query_id"], node_id)
            for row in details
            for node_id in row["gold_node_ids"]
            if node_type is None or graph.nodes[node_id].node_type is node_type
        }
        selected = {
            (row["query_id"], node_id)
            for row in details
            for node_id in row["selected_node_ids"]
            if node_type is None or graph.nodes[node_id].node_type is node_type
        }
        prefix = "evidence" if node_type is None else f"{node_type.value.lower()}_evidence"
        true_positive = len(gold & selected)
        precision = true_positive / len(selected) if selected else 0.0
        recall = true_positive / len(gold) if gold else None
        result[f"micro_{prefix}_precision"] = precision if gold else None
        result[f"micro_{prefix}_recall"] = recall
        result[f"micro_{prefix}_f1"] = (
            2 * precision * recall / (precision + recall)
            if recall is not None and precision + recall
            else 0.0 if recall is not None else None
        )
    return result


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target.resolve()
