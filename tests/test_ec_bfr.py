from paper_rag.domain import EvidenceEdge, EvidenceNode, NodeType, QuerySpec, RelationType, SearchHit
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.retrieval.ec_bfr import ECBFRConfig, EvidenceClosureBudgetedForestRetriever


def test_forest_is_closed_and_budgeted() -> None:
    graph = EvidenceGraph()
    for paper in ("p1", "p2"):
        graph.extend(
            [
                EvidenceNode(f"{paper}:s", paper, NodeType.SENTENCE, text="alloy strength 500 MPa"),
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
    forest = retriever.retrieve(QuerySpec("Which alloy reaches 500 MPa?", value=500, unit="MPa"), hits)
    forest.validate_budget()
    assert forest.trees
    for tree in forest.trees:
        assert f"{tree.paper_id}:f" in tree.node_ids
        assert f"{tree.paper_id}:c" in tree.node_ids

