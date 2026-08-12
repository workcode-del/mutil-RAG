import numpy as np

from paper_rag.benchmarking.base import read_jsonl, write_jsonl
from paper_rag.domain import EvidenceNode, NodeType
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.training import build_query_pairs


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
