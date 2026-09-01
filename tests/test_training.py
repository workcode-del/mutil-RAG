import numpy as np

from paper_rag.benchmarking.base import read_jsonl, write_jsonl
from paper_rag.domain import EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.training import (
    _pair_papers,
    _paper_graph_index,
    _paper_subgraph,
    _relation_triples,
    build_query_pairs,
)


def test_query_pairs_choose_same_type_hard_negative(tmp_path) -> None:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("p:gold", "p", NodeType.SENTENCE, text="gold"),
            EvidenceNode("p:hard", "p", NodeType.SENTENCE, text="hard"),
            EvidenceNode("p:easy", "p", NodeType.SENTENCE, text="easy"),
            EvidenceNode("p:figure", "p", NodeType.FIGURE, image_path="image.jpg"),
        ],
        [],
    )
    graph_path = tmp_path / "graph.json"
    save_graph(graph, graph_path)
    samples = write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "query_id": "q",
                "query": "question",
                "paper_id": "p",
                "relevant_node_ids": ["p:gold"],
                "candidate_node_ids": ["p:gold", "p:hard", "p:easy", "p:figure"],
            }
        ],
    )
    embeddings = tmp_path / "base.npz"
    np.savez_compressed(
        embeddings,
        **{
            "p:gold": np.array([1.0, 0.0]),
            "p:hard": np.array([0.9, 0.1]),
            "p:easy": np.array([0.0, 1.0]),
        },
    )

    output = build_query_pairs(
        graph_path,
        samples,
        tmp_path / "pairs.jsonl",
        embeddings_path=embeddings,
    )

    assert read_jsonl(output)[0]["negative_node_id"] == "p:hard"


def test_query_pairs_keep_all_gold_evidence(tmp_path) -> None:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("p:a", "p", NodeType.SENTENCE, text="a"),
            EvidenceNode("p:b", "p", NodeType.SENTENCE, text="b"),
            EvidenceNode("p:n", "p", NodeType.SENTENCE, text="negative"),
        ],
        [],
    )
    graph_path = tmp_path / "graph.json"
    save_graph(graph, graph_path)
    samples = write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "query_id": "q",
                "query": "question",
                "paper_id": "p",
                "relevant_node_ids": ["p:a", "p:b"],
                "candidate_node_ids": ["p:a", "p:b", "p:n"],
            }
        ],
    )

    output = build_query_pairs(graph_path, samples, tmp_path / "pairs.jsonl")

    assert {row["positive_node_id"] for row in read_jsonl(output)} == {"p:a", "p:b"}


def test_query_pairs_infer_candidates_from_gold_papers(tmp_path) -> None:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("p:gold", "p", NodeType.FIGURE, image_path="gold.png"),
            EvidenceNode("p:negative", "p", NodeType.FIGURE, image_path="negative.png"),
            EvidenceNode("other:node", "other", NodeType.FIGURE, image_path="other.png"),
        ],
        [],
    )
    graph_path = tmp_path / "graph.json"
    save_graph(graph, graph_path)
    samples = write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "query_id": "q",
                "query": "question",
                "relevant_node_ids": ["p:gold"],
            }
        ],
    )

    output = build_query_pairs(graph_path, samples, tmp_path / "pairs.jsonl")

    assert read_jsonl(output)[0]["negative_node_id"] == "p:negative"


def test_training_batch_includes_negative_papers() -> None:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("a:positive", "a", NodeType.SENTENCE, text="positive"),
            EvidenceNode("b:negative", "b", NodeType.SENTENCE, text="negative"),
            EvidenceNode("c:excluded", "c", NodeType.SENTENCE, text="excluded"),
        ],
        [],
    )
    pairs = [
        {
            "positive_node_id": "a:positive",
            "negative_node_id": "b:negative",
        }
    ]

    papers = _pair_papers(graph, pairs)
    subgraph = _paper_subgraph(graph, papers, index=_paper_graph_index(graph))

    assert papers == {"a", "b"}
    assert set(subgraph.nodes) == {"a:positive", "b:negative"}


def test_relation_supervision_ignores_low_confidence_edges() -> None:
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("p:c", "p", NodeType.CAPTION, text="caption"),
            EvidenceNode("p:f1", "p", NodeType.FIGURE, image_path="one.png"),
            EvidenceNode("p:f2", "p", NodeType.FIGURE, image_path="two.png"),
        ],
        [
            EvidenceEdge(
                "p:c", "p:f1", RelationType.CAPTION_OF, confidence=0.7
            )
        ],
    )

    assert _relation_triples(graph, {"p"}, seed=0) == []
