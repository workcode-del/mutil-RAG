import math

import pytest

from paper_rag.domain import (
    EvidenceForest,
    EvidenceNode,
    EvidenceTree,
    NodeType,
    QuerySpec,
    SearchHit,
)
from paper_rag.evaluation.metrics import ranking_metrics, result_metrics, rouge_l_f1
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.pipeline import PipelineResult


def test_ranking_metrics_cover_first_hit_and_complete_evidence() -> None:
    metrics = ranking_metrics(["wrong", "gold-a", "gold-b"], {"gold-a", "gold-b"}, (1, 3))

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["joint_recall_at_3"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["ndcg_at_3"] == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )


def test_result_metrics_separate_ranked_hits_from_selected_evidence() -> None:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("p:s1", "p", NodeType.SENTENCE, text="answer"),
            EvidenceNode("p:s2", "p", NodeType.SENTENCE, text="distractor"),
        ],
        [],
    )
    result = PipelineResult(
        QuerySpec("question"),
        [
            SearchHit("p:s2", "p", NodeType.SENTENCE, 1.0),
            SearchHit("p:s1", "p", NodeType.SENTENCE, 0.9),
        ],
        EvidenceForest([EvidenceTree("p", {"p:s1"}, cost=1)], total_cost=1, budget=10),
    )

    metrics = result_metrics(graph, result, {"p:s1"}, cutoffs=(1, 2), latency_ms=2.0)

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_2"] == 1.0
    assert metrics["evidence_f1"] == 1.0
    assert metrics["budget_violation"] == 0.0


def test_rouge_l_f1() -> None:
    assert rouge_l_f1("a b c", "a x c") == pytest.approx(2 / 3)
