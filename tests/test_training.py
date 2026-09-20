import numpy as np
import pytest

from paper_rag.benchmarking.base import read_jsonl, write_jsonl
from paper_rag.domain import EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.training import (
    _pair_papers,
    _paper_aware_batches,
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
    assert embeddings.with_suffix(".matrix.npy").exists()


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


@pytest.mark.parametrize("explicit_candidates", [False, True])
def test_query_hard_negatives_respect_training_and_candidate_scopes(tmp_path, explicit_candidates):
    graph = EvidenceGraph()
    vectors = {
        "p:gold": [1.0, 0.0], "p:similar": [0.99, 0.1],
        "p:query_hard": [0.1, 0.9], "q:other_train": [0.0, 1.0],
        "heldout:node": [0.0, 1.0],
    }
    for node_id in vectors:
        graph.add_node(EvidenceNode(node_id, node_id.split(":")[0], NodeType.SENTENCE, text=node_id))
    graph_path = tmp_path / "graph.json"
    save_graph(graph, graph_path)
    row = {"query_id": "a", "query": "question", "relevant_node_ids": ["p:gold"]}
    if explicit_candidates:
        row["candidate_node_ids"] = ["p:gold", "p:similar", "p:query_hard"]
    samples = write_jsonl(tmp_path / "train.jsonl", [
        row, {"query_id": "b", "query": "second", "relevant_node_ids": ["q:other_train"]},
    ])
    base = tmp_path / "base.npz"
    np.savez_compressed(base, **vectors)
    queries = tmp_path / "queries.npz"
    np.savez_compressed(queries, a=[0.0, 1.0], b=[1.0, 0.0])
    output = build_query_pairs(
        graph_path, samples, tmp_path / "pairs.jsonl", embeddings_path=base,
        query_embeddings_path=queries, negative_scope="train",
    )
    rows = read_jsonl(output)
    assert rows[0]["negative_node_id"] == ("p:query_hard" if explicit_candidates else "q:other_train")
    assert all(row["negative_node_id"] != "heldout:node" for row in rows)
    assert rows[0]["negative_sampling"] == "query"


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


def test_paper_aware_batches_pack_multiple_papers_without_splitting_groups() -> None:
    graph = EvidenceGraph()
    rows = []
    for paper_id, pair_count in (("a", 2), ("b", 1), ("c", 1)):
        graph.extend(
            [
                EvidenceNode(f"{paper_id}:p", paper_id, NodeType.SENTENCE, text="positive"),
                EvidenceNode(f"{paper_id}:n", paper_id, NodeType.SENTENCE, text="negative"),
            ],
            [],
        )
        rows.extend(
            {
                "query_id": f"{paper_id}:{index}",
                "positive_node_id": f"{paper_id}:p",
                "negative_node_id": f"{paper_id}:n",
            }
            for index in range(pair_count)
        )

    batches = _paper_aware_batches(graph, rows, batch_size=3, seed=42)

    assert sorted(map(len, batches)) == [1, 3]
    assert all(len(batch) <= 3 for batch in batches)
    assert any(len(_pair_papers(graph, batch)) == 2 for batch in batches)
    assert any(
        sum(row["query_id"].startswith("a:") for row in batch) == 2
        for batch in batches
    )


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
