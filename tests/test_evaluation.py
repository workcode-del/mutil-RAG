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
from paper_rag.evaluation.comparison import DEFAULT_METRICS
from paper_rag.evaluation.evidence_mapping import map_evidence, normalize_evidence
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


def test_result_metrics_report_each_modality_at_every_cutoff() -> None:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("p:s", "p", NodeType.SENTENCE, text="text evidence"),
            EvidenceNode("p:f", "p", NodeType.FIGURE, image_path="figure.png"),
            EvidenceNode("p:t", "p", NodeType.TABLE, text="table evidence"),
            EvidenceNode("p:x", "p", NodeType.SENTENCE, text="distractor"),
        ],
        [],
    )
    result = PipelineResult(
        QuerySpec("question"),
        [
            SearchHit("p:s", "p", NodeType.SENTENCE, 1.0),
            SearchHit("p:x", "p", NodeType.SENTENCE, 0.9),
            SearchHit("p:f", "p", NodeType.FIGURE, 0.8),
            SearchHit("p:t", "p", NodeType.TABLE, 0.7),
        ],
        EvidenceForest(
            [EvidenceTree("p", {"p:s", "p:f", "p:t"}, cost=3)], total_cost=3, budget=10
        ),
    )

    metrics = result_metrics(
        graph,
        result,
        {"p:s", "p:f", "p:t"},
        cutoffs=(1, 2, 3, 4),
        latency_ms=1.0,
    )

    assert metrics["sentence_recall_at_1"] == 1.0
    assert metrics["figure_recall_at_1"] == 0.0
    assert metrics["figure_recall_at_2"] == 0.0
    assert metrics["figure_recall_at_3"] == 1.0
    assert metrics["table_recall_at_3"] == 0.0
    assert metrics["table_recall_at_4"] == 1.0


def test_rouge_l_f1() -> None:
    assert rouge_l_f1("a b c", "a x c") == pytest.approx(2 / 3)


def test_comparison_exports_figure_and_table_recall_at_three() -> None:
    assert "macro_figure_recall_at_3" in DEFAULT_METRICS
    assert "macro_table_recall_at_3" in DEFAULT_METRICS


def test_evidence_mapping_prefers_exact_then_fuzzy_match() -> None:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode(
                "p:sentence:1",
                "p",
                NodeType.SENTENCE,
                text="Block diffusion verifies several candidate tokens in parallel.",
            ),
            EvidenceNode(
                "p:sentence:2", "p", NodeType.SENTENCE, text="Unrelated sentence."
            ),
        ],
        [],
    )

    exact, fuzzy = map_evidence(
        graph,
        "p",
        [
            "Block diffusion verifies several candidate tokens in parallel.",
            "Block diffusion verifies candidate tokens in parallel.",
        ],
        min_score=0.75,
    )

    assert (exact.node_id, exact.method) == ("p:sentence:1", "exact")
    assert (fuzzy.node_id, fuzzy.method) == ("p:sentence:1", "fuzzy")


def test_normalize_evidence_normalizes_unicode_and_punctuation() -> None:
    assert normalize_evidence("ＤFlash:  Fast!\n") == "dflash fast"
