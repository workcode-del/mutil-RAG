import numpy as np

from paper_rag.domain import EvidenceNode, NodeType
from paper_rag.embedding import ExactEmbeddingStore
from paper_rag.evidence_graph import EvidenceGraph


def test_exact_embedding_store_ranks_with_sample_scope() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:a", "p", NodeType.SENTENCE, text="a"))
    graph.add_node(EvidenceNode("p:b", "p", NodeType.SENTENCE, text="b"))
    graph.add_node(EvidenceNode("q:c", "q", NodeType.SENTENCE, text="c"))
    store = ExactEmbeddingStore(
        graph,
        {
            "p:a": np.asarray([1.0, 0.0]),
            "p:b": np.asarray([0.0, 1.0]),
            "q:c": np.asarray([1.0, 0.0]),
        },
    )

    hits = store.search(
        "query",
        np.asarray([1.0, 0.0]),
        [NodeType.SENTENCE],
        2,
        paper_ids={"p"},
        candidate_node_ids={"p:a", "p:b", "q:c"},
    )

    assert [hit.node_id for hit in hits] == ["p:a", "p:b"]
