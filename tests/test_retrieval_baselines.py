from paper_rag.domain import (
    EvidenceEdge,
    EvidenceNode,
    NodeType,
    QuerySpec,
    RelationType,
    SearchHit,
)
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.retrieval.baselines import PCSTEvidenceRetriever, RankedEvidenceRetriever
from paper_rag.retrieval.ec_bfr import ECBFRConfig


def graph_fixture() -> EvidenceGraph:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("p:s", "p", NodeType.SENTENCE, text="see figure"),
            EvidenceNode("p:f", "p", NodeType.FIGURE, image_path="figure.png"),
            EvidenceNode("p:c", "p", NodeType.CAPTION, text="Figure caption"),
        ],
        [
            EvidenceEdge("p:s", "p:f", RelationType.REFERS_TO),
            EvidenceEdge("p:c", "p:f", RelationType.CAPTION_OF),
        ],
    )
    return graph


def test_top_k_and_one_hop_share_selection_contract() -> None:
    graph = graph_fixture()
    hits = [SearchHit("p:s", "p", NodeType.SENTENCE, 1.0)]
    query = QuerySpec("figure")

    top_k = RankedEvidenceRetriever(graph, top_k=1, budget=1000, image_unit=20)
    one_hop = RankedEvidenceRetriever(graph, top_k=1, budget=1000, image_unit=20, hops=1)

    assert top_k.retrieve(query, hits).node_ids == {"p:s"}
    assert one_hop.retrieve(query, hits).node_ids == {"p:s", "p:f"}


def test_pcst_closure_is_an_explicit_baseline_variant() -> None:
    graph = graph_fixture()
    hits = [SearchHit("p:s", "p", NodeType.SENTENCE, 1.0)]
    config = ECBFRConfig(budget=1000, image_unit=20, candidate_hops=1, lambda_values=(1.0,))

    plain = PCSTEvidenceRetriever(graph, config, apply_closure=False).retrieve(QuerySpec("q"), hits)
    closed = PCSTEvidenceRetriever(graph, config, apply_closure=True).retrieve(QuerySpec("q"), hits)

    assert "p:s" in plain.node_ids
    assert closed.node_ids.issuperset({"p:s", "p:f", "p:c"})
