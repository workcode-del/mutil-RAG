from paper_rag.domain import EvidenceNode, NodeType
from paper_rag.evaluation.evidence_mapping import map_evidence, normalize_evidence
from paper_rag.evidence_graph import EvidenceGraph


def test_evidence_mapping_prefers_exact_then_fuzzy_match() -> None:
    graph = EvidenceGraph()
    graph.add_node(
        EvidenceNode(
            "p:sentence:1",
            "p",
            NodeType.SENTENCE,
            text="Block diffusion verifies several candidate tokens in parallel.",
        )
    )
    graph.add_node(
        EvidenceNode("p:sentence:2", "p", NodeType.SENTENCE, text="Unrelated sentence.")
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

    assert exact.node_id == "p:sentence:1"
    assert exact.method == "exact"
    assert fuzzy.node_id == "p:sentence:1"
    assert fuzzy.method == "fuzzy"


def test_normalize_evidence_normalizes_unicode_and_punctuation() -> None:
    assert normalize_evidence("ＤFlash:  Fast!\n") == "dflash fast"
