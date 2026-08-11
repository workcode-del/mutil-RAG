from paper_rag.domain import EvidenceNode, NodeType
from paper_rag.embedding.bm25_store import BM25EvidenceStore
from paper_rag.evidence_graph import EvidenceGraph


def test_bm25_respects_paper_scope_and_node_type() -> None:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("p1:s", "p1", NodeType.SENTENCE, text="block diffusion decoding"),
            EvidenceNode("p2:s", "p2", NodeType.SENTENCE, text="block diffusion decoding"),
            EvidenceNode("p1:c", "p1", NodeType.CAPTION, text="unrelated caption"),
        ],
        [],
    )

    hits = BM25EvidenceStore(graph).search(
        "block decoding",
        None,
        [NodeType.SENTENCE, NodeType.CAPTION],
        paper_ids={"p1"},
        candidate_node_ids={"p1:s"},
    )

    assert [hit.node_id for hit in hits] == ["p1:s"]
    assert hits[0].score_components.keys() == {"bm25"}
