from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from paper_rag.domain import NodeType, SearchHit
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.pipeline import PipelineResult
from paper_rag.retrieval.closure import ClosurePolicy, evidence_closure, validate_closure


def ranking_metrics(
    ranked_ids: list[str], gold_ids: set[str], cutoffs: Iterable[int]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for cutoff in sorted(set(cutoffs)):
        prefix = ranked_ids[:cutoff]
        relevant = sum(node_id in gold_ids for node_id in prefix)
        metrics[f"recall_at_{cutoff}"] = relevant / len(gold_ids)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_ids), cutoff) + 1))
        actual = sum(
            1.0 / math.log2(rank + 1)
            for rank, node_id in enumerate(prefix, 1)
            if node_id in gold_ids
        )
        metrics[f"ndcg_at_{cutoff}"] = actual / ideal if ideal else 0.0
        metrics[f"joint_recall_at_{cutoff}"] = float(gold_ids.issubset(prefix))
    metrics["mrr"] = next(
        (1.0 / rank for rank, node_id in enumerate(ranked_ids, 1) if node_id in gold_ids),
        0.0,
    )
    metrics["mrr_at_10"] = next(
        (1.0 / rank for rank, node_id in enumerate(ranked_ids[:10], 1) if node_id in gold_ids),
        0.0,
    )
    return metrics


def result_metrics(
    graph: EvidenceGraph,
    result: PipelineResult,
    gold_ids: set[str],
    *,
    cutoffs: Iterable[int],
    latency_ms: float,
    reference_answer: str | None = None,
) -> dict[str, float | None]:
    cutoffs = tuple(cutoffs)
    if not cutoffs:
        raise ValueError("At least one ranking cutoff is required")
    ranked_ids = [hit.node_id for hit in result.hits]
    selected = result.forest.node_ids
    true_positive = len(selected & gold_ids)
    precision = true_positive / len(selected) if selected else 0.0
    recall = true_positive / len(gold_ids)
    metrics: dict[str, float | None] = {
        **ranking_metrics(ranked_ids, gold_ids, cutoffs),
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": harmonic_mean(precision, recall),
        "closure_validity": float(validate_closure(graph, selected, ClosurePolicy())),
        "dependency_completeness": dependency_completeness(graph, selected),
        "budget_violation": float(result.forest.total_cost > result.forest.budget),
        "evidence_cost": float(result.forest.total_cost),
        "selected_nodes": float(len(selected)),
        "latency_ms": latency_ms,
    }
    node_types = (
        NodeType.SENTENCE,
        NodeType.FIGURE,
        NodeType.TABLE,
        NodeType.CAPTION,
        NodeType.CHART_DATA,
    )
    for node_type in node_types:
        typed_gold = {
            node_id for node_id in gold_ids if graph.nodes[node_id].node_type is node_type
        }
        prefix = node_type.value.lower()
        for cutoff in sorted(set(cutoffs)):
            ranked_prefix = set(ranked_ids[:cutoff])
            metrics[f"{prefix}_recall_at_{cutoff}"] = (
                len(typed_gold & ranked_prefix) / len(typed_gold) if typed_gold else None
            )
        typed_selected = {
            node_id for node_id in selected if graph.nodes[node_id].node_type is node_type
        }
        typed_true_positive = len(typed_selected & typed_gold)
        typed_precision = (
            typed_true_positive / len(typed_selected) if typed_selected else 0.0
        )
        typed_recall = typed_true_positive / len(typed_gold) if typed_gold else None
        metrics[f"{prefix}_evidence_precision"] = typed_precision if typed_gold else None
        metrics[f"{prefix}_evidence_recall"] = typed_recall
        metrics[f"{prefix}_evidence_f1"] = (
            harmonic_mean(typed_precision, typed_recall) if typed_recall is not None else None
        )
    if result.answer and reference_answer is not None:
        metrics["answer_exact_match"] = float(
            normalize_answer(result.answer.text) == normalize_answer(reference_answer)
        )
        metrics["answer_token_f1"] = token_f1(result.answer.text, reference_answer)
        metrics["answer_rouge_l_f1"] = rouge_l_f1(result.answer.text, reference_answer)
        cited = set(result.answer.evidence_ids)
        citation_precision = len(cited & gold_ids) / len(cited) if cited else 0.0
        citation_recall = len(cited & gold_ids) / len(gold_ids)
        metrics["citation_precision"] = citation_precision
        metrics["citation_recall"] = citation_recall
        metrics["citation_f1"] = harmonic_mean(citation_precision, citation_recall)
    return metrics


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for name, value in row.items():
            if (
                value is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                values.setdefault(name, []).append(float(value))
    return {f"macro_{name}": sum(items) / len(items) for name, items in sorted(values.items())}


def dependency_completeness(graph: EvidenceGraph, selected: set[str]) -> float | None:
    requirements = []
    for node_id in selected:
        required = evidence_closure(graph, {node_id}) - {node_id}
        if required:
            requirements.append(required)
    if not requirements:
        return None
    return sum(required.issubset(selected) for required in requirements) / len(requirements)


def harmonic_mean(left: float, right: float) -> float:
    return 2.0 * left * right / (left + right) if left + right else 0.0


def normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(normalize_answer(prediction).split())
    expected = Counter(normalize_answer(reference).split())
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    return harmonic_mean(precision, recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return 0.0
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for index, expected_token in enumerate(expected, 1):
            current.append(
                previous[index - 1] + 1
                if token == expected_token
                else max(previous[index], current[-1])
            )
        previous = current
    common = previous[-1]
    return harmonic_mean(common / len(predicted), common / len(expected))


def serialize_hit(hit: SearchHit) -> dict[str, Any]:
    return {
        "node_id": hit.node_id,
        "paper_id": hit.paper_id,
        "node_type": hit.node_type.value,
        "score": hit.score,
        "score_components": hit.score_components,
    }
