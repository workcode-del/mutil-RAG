from pathlib import Path

from paper_rag.benchmarking.base import grouped_split
from paper_rag.benchmarking.mmdocrag import _build_quote_graph, _sample, _string_list
from paper_rag.benchmarking.peerqa import _build_official_graph
from paper_rag.domain import NodeType


def test_peerqa_official_rows_build_stable_nodes() -> None:
    graph = _build_official_graph(
        [
            {
                "paper_id": "paper",
                "idx": 0,
                "pidx": 0,
                "sidx": 0,
                "type": "sentence",
                "content": "First sentence.",
            },
            {
                "paper_id": "paper",
                "idx": 1,
                "pidx": 0,
                "sidx": 1,
                "type": "caption",
                "content": "Figure caption.",
            },
        ]
    )

    assert graph.nodes["peerqa::paper::0"].node_type is NodeType.SENTENCE
    assert graph.nodes["peerqa::paper::1"].node_type is NodeType.CAPTION
    assert len(graph.edges) == 1


def test_mmdocrag_split_namespaces_prevent_qid_collisions() -> None:
    row = {
        "q_id": 1,
        "doc_name": "document",
        "question": "What is shown?",
        "text_quotes": [{"quote_id": "text1", "text": "Evidence", "page_id": 1}],
        "img_quotes": [
            {
                "quote_id": "image1",
                "img_path": "image.jpg",
                "img_description": "A chart",
                "page_id": 2,
            }
        ],
        "gold_quotes": ["text1", "image1"],
        "answer_short": "Answer",
    }

    graph, missing = _build_quote_graph(
        [("development", row), ("test", row)],
        {"image.jpg": Path("image.jpg")},
    )
    sample = _sample(row, "test")

    assert len(graph.nodes) == 4
    assert not missing
    assert set(sample["relevant_node_ids"]).issubset(sample["candidate_node_ids"])
    assert all(node_id.startswith("mmdocrag::test::") for node_id in sample["candidate_node_ids"])


def test_grouped_split_keeps_documents_together() -> None:
    rows = [
        {"paper_id": paper_id, "query_id": f"{paper_id}-{index}"}
        for paper_id in ("a", "b", "c", "d", "e")
        for index in range(3)
    ]

    split = grouped_split(rows, group_key="paper_id")
    allocation = {
        row["paper_id"]: name
        for name, items in split.items()
        for row in items
    }

    assert len(allocation) == 5
    for paper_id in allocation:
        assert sum(row["paper_id"] == paper_id for items in split.values() for row in items) == 3


def test_mmdocrag_modality_metadata_accepts_scalar_or_list() -> None:
    assert _string_list("image") == ["image"]
    assert _string_list(["text", "image"]) == ["text", "image"]
