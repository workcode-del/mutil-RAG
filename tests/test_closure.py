from paper_rag.domain import EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.retrieval.closure import ClosurePolicy, evidence_closure, validate_closure


def graph_fixture() -> EvidenceGraph:
    graph = EvidenceGraph()
    nodes = [
        EvidenceNode("p:s1", "p", NodeType.SENTENCE, text="As shown in Fig. 1."),
        EvidenceNode("p:f1", "p", NodeType.FIGURE, image_path="figure.png"),
        EvidenceNode("p:c1", "p", NodeType.CAPTION, text="Figure 1. Strength curves."),
        EvidenceNode("p:d1", "p", NodeType.CHART_DATA, text="sample | strength"),
    ]
    graph.extend(
        nodes,
        [
            EvidenceEdge("p:s1", "p:f1", RelationType.REFERS_TO, mandatory_for_closure=True),
            EvidenceEdge("p:c1", "p:f1", RelationType.CAPTION_OF, mandatory_for_closure=True),
            EvidenceEdge("p:d1", "p:f1", RelationType.DERIVED_FROM, mandatory_for_closure=True),
        ],
    )
    return graph


def test_sentence_closure_adds_figure_and_caption() -> None:
    graph = graph_fixture()
    closed = evidence_closure(graph, {"p:s1"}, ClosurePolicy())
    assert closed == {"p:s1", "p:f1", "p:c1"}
    assert validate_closure(graph, closed, ClosurePolicy())
    assert evidence_closure(graph, closed) == closed


def test_chart_closure_adds_figure_and_caption() -> None:
    graph = graph_fixture()
    assert evidence_closure(graph, {"p:d1"}) == {"p:d1", "p:f1", "p:c1"}

