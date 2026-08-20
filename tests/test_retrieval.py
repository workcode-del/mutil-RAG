from paper_rag.domain import (
    EvidenceEdge,
    EvidenceNode,
    NodeType,
    QuerySpec,
    RelationType,
    SearchHit,
)
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.embedding.bm25_store import BM25EvidenceStore
from paper_rag.retrieval.baselines import PCSTEvidenceRetriever, RankedEvidenceRetriever
from paper_rag.retrieval.closure import ClosurePolicy, evidence_closure, validate_closure
from paper_rag.retrieval.ec_bfr import (
    ECBFRConfig,
    EvidenceClosureBudgetedForestRetriever,
)
from paper_rag.retrieval.fusion import reciprocal_rank_fusion


def _baseline_graph() -> EvidenceGraph:
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
    graph = _baseline_graph()
    hits = [SearchHit("p:s", "p", NodeType.SENTENCE, 1.0)]
    query = QuerySpec("figure")

    top_k = RankedEvidenceRetriever(graph, top_k=1, budget=1000, image_unit=20)
    one_hop = RankedEvidenceRetriever(graph, top_k=1, budget=1000, image_unit=20, hops=1)

    assert top_k.retrieve(query, hits).node_ids == {"p:s"}
    assert one_hop.retrieve(query, hits).node_ids == {"p:s", "p:f"}


def test_pcst_closure_is_an_explicit_baseline_variant() -> None:
    graph = _baseline_graph()
    hits = [SearchHit("p:s", "p", NodeType.SENTENCE, 1.0)]
    config = ECBFRConfig(budget=1000, image_unit=20, candidate_hops=1, lambda_values=(1.0,))

    plain = PCSTEvidenceRetriever(graph, config, apply_closure=False).retrieve(QuerySpec("q"), hits)
    closed = PCSTEvidenceRetriever(graph, config, apply_closure=True).retrieve(QuerySpec("q"), hits)

    assert "p:s" in plain.node_ids
    assert closed.node_ids.issuperset({"p:s", "p:f", "p:c"})


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


def _closure_graph() -> EvidenceGraph:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("p:s1", "p", NodeType.SENTENCE, text="As shown in Fig. 1."),
            EvidenceNode("p:f1", "p", NodeType.FIGURE, image_path="figure.png"),
            EvidenceNode("p:c1", "p", NodeType.CAPTION, text="Figure 1. Strength curves."),
            EvidenceNode("p:d1", "p", NodeType.CHART_DATA, text="sample | strength"),
            EvidenceNode("p:t1", "p", NodeType.TABLE, text="Model | F1"),
            EvidenceNode("p:tc1", "p", NodeType.CAPTION, text="Table 1. Results."),
        ],
        [
            EvidenceEdge("p:s1", "p:f1", RelationType.REFERS_TO, mandatory_for_closure=True),
            EvidenceEdge("p:c1", "p:f1", RelationType.CAPTION_OF, mandatory_for_closure=True),
            EvidenceEdge("p:d1", "p:f1", RelationType.DERIVED_FROM, mandatory_for_closure=True),
            EvidenceEdge("p:tc1", "p:t1", RelationType.CAPTION_OF, mandatory_for_closure=True),
        ],
    )
    return graph


def test_typed_evidence_closure_reaches_a_fixed_point() -> None:
    graph = _closure_graph()
    closed = evidence_closure(graph, {"p:s1"}, ClosurePolicy())
    assert closed == {"p:s1", "p:f1", "p:c1"}
    assert validate_closure(graph, closed, ClosurePolicy())
    assert evidence_closure(graph, closed) == closed
    assert evidence_closure(graph, {"p:d1"}) == {"p:d1", "p:f1", "p:c1"}
    assert evidence_closure(graph, {"p:t1"}) == {"p:t1", "p:tc1"}


def test_forest_is_closed_and_budgeted() -> None:
    graph = EvidenceGraph()
    for paper in ("p1", "p2"):
        graph.extend(
            [
                EvidenceNode(
                    f"{paper}:s", paper, NodeType.SENTENCE, text="alloy strength 500 MPa"
                ),
                EvidenceNode(f"{paper}:f", paper, NodeType.FIGURE, image_path=f"{paper}.png"),
                EvidenceNode(f"{paper}:c", paper, NodeType.CAPTION, text="Figure 1. strength"),
            ],
            [
                EvidenceEdge(f"{paper}:s", f"{paper}:f", RelationType.REFERS_TO),
                EvidenceEdge(f"{paper}:c", f"{paper}:f", RelationType.CAPTION_OF),
            ],
        )
    hits = [
        SearchHit("p1:s", "p1", NodeType.SENTENCE, 1.0),
        SearchHit("p2:s", "p2", NodeType.SENTENCE, 0.9),
    ]
    retriever = EvidenceClosureBudgetedForestRetriever(
        graph, ECBFRConfig(budget=80, image_unit=20, lambda_values=(1.0,))
    )
    forest = retriever.retrieve(
        QuerySpec("Which alloy reaches 500 MPa?", value=500, unit="MPa"), hits
    )

    forest.validate_budget()
    assert forest.trees
    for tree in forest.trees:
        assert {f"{tree.paper_id}:f", f"{tree.paper_id}:c"}.issubset(tree.node_ids)


def test_rrf_does_not_depend_on_raw_score_scale() -> None:
    result = reciprocal_rank_fusion({"gme": ["a", "b"], "reranker": ["b", "a"]})
    assert result["a"] == result["b"]
