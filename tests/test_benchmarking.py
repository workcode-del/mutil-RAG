import zipfile
from pathlib import Path

import pytest

import paper_rag.benchmarking.multimodalqa as multimodalqa
from paper_rag.benchmarking.base import (
    PROCESSED_SCHEMA_VERSION,
    BenchmarkLayout,
    connected_grouped_split,
    grouped_split,
    write_json,
)
from paper_rag.benchmarking.cli import _ranking_cutoffs, _report_summaries
from paper_rag.benchmarking.download import _valid_download, extract_zip
from paper_rag.benchmarking.mmdocrag import _build_quote_graph, _sample, _string_list
from paper_rag.benchmarking.multimodalqa import (
    _component_graph,
    _samples,
    prepare_multimodalqa,
)
from paper_rag.benchmarking.page_datasets import _mmlong_samples, _page_node_id
from paper_rag.benchmarking.peerqa import _build_official_graph
from paper_rag.benchmarking.runner import (
    DEFAULT_SYSTEMS,
    SYSTEMS,
    _official_split,
    _validate_preparation,
    _validate_processed_schema,
    _validate_training_split,
    benchmark_split_statistics,
)
from paper_rag.benchmarking.spiqa import _convert_splits
from paper_rag.domain import EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.io import read_jsonl
from paper_rag.io import write_jsonl
from paper_rag.training import count_relation_triples


def test_rgcn_is_an_explicit_nondefault_graph_baseline() -> None:
    assert SYSTEMS["rgcn"].graph_model == "rgcn"
    assert SYSTEMS["rgcn"].retrieval_method == SYSTEMS["full"].retrieval_method
    assert SYSTEMS["rgcn"].reranker == SYSTEMS["full"].reranker
    assert "rgcn" not in DEFAULT_SYSTEMS

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


def test_mmdocrag_keeps_question_local_quotes_distinct() -> None:
    first = {
        "q_id": 1,
        "doc_name": "document",
        "question": "First?",
        "text_quotes": [
            {"quote_id": "text1", "text": "Shared", "page_id": 1, "layout_id": 2}
        ],
        "img_quotes": [],
        "gold_quotes": ["text1"],
    }
    second = {**first, "q_id": 2, "question": "Second?", "gold_quotes": ["text7"]}
    second["text_quotes"] = [
        {"quote_id": "text7", "text": "Different view", "page_id": 1, "layout_id": 2}
    ]

    graph, _ = _build_quote_graph([("development", first), ("development", second)], {})

    assert len(graph.nodes) == 2
    assert _sample(first, "development")["candidate_node_ids"] != _sample(
        second, "development"
    )["candidate_node_ids"]


def test_mmdocrag_deduplicates_question_local_quote_ids() -> None:
    row = {
        "q_id": 486,
        "doc_name": "document",
        "question": "Question?",
        "text_quotes": [
            {"quote_id": "text11", "text": "Shared", "page_id": 10, "layout_id": 32},
            {"quote_id": "text11", "text": "Shared", "page_id": 9, "layout_id": 32},
        ],
        "img_quotes": [],
        "gold_quotes": ["text11"],
    }

    graph, _ = _build_quote_graph([("development", row)], {})
    sample = _sample(row, "development")

    assert len(graph.nodes) == 1
    assert len(sample["candidate_node_ids"]) == 1
    assert sample["relevant_node_ids"] == sample["candidate_node_ids"]


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


def test_connected_split_keeps_shared_document_components_together() -> None:
    rows = [
        {"query_id": "q1", "paper_ids": ["a", "b"]},
        {"query_id": "q2", "paper_ids": ["b", "c"]},
        {"query_id": "q3", "paper_ids": ["d"]},
    ]

    split = connected_grouped_split(rows)
    allocation = {
        row["query_id"]: name for name, values in split.items() for row in values
    }

    assert allocation["q1"] == allocation["q2"]
    documents_by_split = {
        name: {paper for row in values for paper in row["paper_ids"]}
        for name, values in split.items()
    }
    assert all(
        not left_docs & right_docs
        for left, left_docs in documents_by_split.items()
        for right, right_docs in documents_by_split.items()
        if left < right
    )


def test_connected_split_uses_all_splits_when_possible() -> None:
    rows = [
        {"query_id": f"q{index}", "paper_ids": [f"p{index}"]}
        for index in range(12)
    ]

    split = connected_grouped_split(rows)

    assert all(split.values())
    assert sum(map(len, split.values())) == len(rows)


def test_mmdocrag_modality_metadata_accepts_scalar_or_list() -> None:
    assert _string_list("image") == ["image"]
    assert _string_list(["text", "image"]) == ["text", "image"]


def test_benchmark_cutoffs_are_dataset_specific() -> None:
    assert _ranking_cutoffs(None, "peerqa") == (1, 3, 5, 10)
    assert _ranking_cutoffs(None, "mmdocrag") == (1, 3, 5, 10, 15, 20)
    assert _ranking_cutoffs([20, 10, 20], "mmdocrag") == (10, 20)
    assert _ranking_cutoffs(None, "m3docvqa") == (1, 3, 5, 10)
    assert _ranking_cutoffs(None, "mmlongbench_doc") == (1, 3, 5, 10)
    assert _ranking_cutoffs(None, "multimodalqa") == (1, 3, 5, 10)
    assert _ranking_cutoffs(None, "spiqa") == (1, 3, 5, 10)


def test_spiqa_builds_figure_table_caption_graph_and_audits_missing_reference(
    tmp_path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in (
        "paper-Figure1-1.png",
        "paper-Figure2-1.png",
        "paper-Table1-1.png",
    ):
        (image_root / name).write_bytes(b"\x89PNG\r\n\x1a\nvalid-test-stub")
    paper = {
        "paper_id": "paper",
        "all_figures": {
            "paper-Figure1-1.png": {
                "caption": "A result plot.",
                "content_type": "figure",
                "figure_type": "plot",
            },
            "paper-Table1-1.png": {
                "caption": "Main results.",
                "content_type": "table",
                "figure_type": "table",
            },
            "paper-Figure2-1.png": {
                "caption": "A comparison plot.",
                "content_type": "figure",
                "figure_type": "plot",
            },
        },
        "qa": [
            {
                "question": "Which plot is relevant?",
                "answer": "Figure 1",
                "reference_figure": "paper-Figure1-1.png",
            },
            {
                "question": "Which table is relevant?",
                "answer": "Table 1",
                "reference_figure": "paper-Table1-1.png",
            },
            {
                "question": "Which reference is absent?",
                "answer": "Missing",
                "reference_figure": "paper-Figure9-1.png",
            },
        ],
    }

    graph, samples, audit = _convert_splits(
        {"train": {"paper": paper}}, {"train": image_root}
    )

    assert len(samples["train"]) == 2
    assert {node.node_type for node in graph.nodes.values()} == {
        NodeType.FIGURE,
        NodeType.TABLE,
        NodeType.CAPTION,
    }
    assert sum(edge.relation is RelationType.CAPTION_OF for edge in graph.edges) == 3
    assert count_relation_triples(graph, {"paper"}) == 2
    assert _official_split("spiqa") == "test"
    assert samples["train"][0]["required_modalities"] == ["figure"]
    assert samples["train"][1]["required_modalities"] == ["table"]
    assert all(
        set(sample["relevant_node_ids"]).issubset(sample["candidate_node_ids"])
        for sample in samples["train"]
    )
    assert all("paper_ids" not in sample for sample in samples["train"])
    assert not audit["missing_images"]
    assert audit["missing_evidence"] == ["spiqa::train::paper::2:paper-Figure9-1.png"]


def test_multimodalqa_imports_text_table_image_components(tmp_path) -> None:
    documents = tmp_path / "parsed_documents" / "dev"
    images = tmp_path / "image_components" / "dev"
    documents.mkdir(parents=True)
    images.mkdir(parents=True)
    (images / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\nvalid-test-stub")
    (documents / "doc.json").write_text(
        __import__("json").dumps(
            {
                "title": "Paper A",
                "text": {"text_1": {"text": "Text evidence"}},
                "table": {
                    "table_1": {
                        "table": [
                            [{"text": "Model"}, {"text": "F1"}],
                            [
                                {"text": "Ours", "image": {"filename": "figure.png"}},
                                {"text": "90"},
                            ],
                        ]
                    }
                },
                "image": {
                    "image_1": {"filename": "figure.png", "caption": {"text": "A figure"}}
                },
            }
        ),
        encoding="utf-8",
    )

    graph, index, missing_images = _component_graph(documents.parent, images.parent)
    rows = [
        {
            "qid": "q1",
            "question": "Compare the evidence",
            "answers": [{"answer": "Ours"}],
            "evidences": [
                {
                    "mmqa_doc_modality": "table",
                    "gold_webpage_title": "Paper A",
                    "gold_component_id": "table_1",
                },
                {
                    "mmqa_doc_modality": "image",
                    "gold_webpage_title": "Paper A",
                    "gold_component_id": "image_1",
                },
            ],
        }
    ]
    samples, missing_evidence = _samples(rows, index)

    assert not missing_images
    assert not missing_evidence
    assert {node.node_type for node in graph.nodes.values()} >= {
        NodeType.SENTENCE,
        NodeType.TABLE,
        NodeType.FIGURE,
        NodeType.CAPTION,
    }
    assert samples[0]["required_modalities"] == ["table", "image"]
    assert len(samples[0]["relevant_node_ids"]) == 2
    assert samples[0]["split_group_ids"] == ["Paper A"]
    assert "paper_ids" not in samples[0]
    assert "candidate_node_ids" not in samples[0]
    assert next(
        node for node in graph.nodes.values() if node.node_type is NodeType.TABLE
    ).image_path.endswith("figure.png")


def test_multimodalqa_prepare_extracts_downloaded_component_archives(tmp_path) -> None:
    source = tmp_path / "snapshot"
    source.mkdir()
    rows = [
        {
            "qid": "q1",
            "question": "What is shown?",
            "answers": [{"answer": "A figure"}],
            "evidences": [
                {
                    "mmqa_doc_modality": "image",
                    "gold_webpage_title": "Paper A",
                    "gold_component_id": "image_1",
                }
            ],
        }
    ]
    (source / "QAs_dev_labeled.json").write_text(
        __import__("json").dumps(rows), encoding="utf-8"
    )
    with zipfile.ZipFile(source / "parsed_documents.zip", "w") as archive:
        archive.writestr(
            "dev/doc.json",
            __import__("json").dumps(
                {
                    "title": "Paper A",
                    "image": {
                        "image_1": {
                            "filename": "figure.png",
                            "caption": {"text": "A figure"},
                        }
                    },
                }
            ),
        )
    with zipfile.ZipFile(source / "image_components.zip", "w") as archive:
        archive.writestr("dev/figure.png", b"\x89PNG\r\n\x1a\nvalid-test-stub")

    report = prepare_multimodalqa(
        BenchmarkLayout.create("multimodalqa", tmp_path / "benchmarks"),
        source=source,
    )

    assert report["samples"] == 1
    assert (source / "parsed_documents" / "dev" / "doc.json").is_file()
    assert (source / "image_components" / "dev" / "figure.png").is_file()


def test_multimodalqa_prepare_reads_current_parquet_snapshot(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "snapshot"
    source.mkdir()
    for name in multimodalqa.PARQUET_FILES:
        (source / name).touch()
    png = b"\x89PNG\r\n\x1a\nvalid-test-stub"
    parquet_rows = {
        "text.parquet": [
            {
                "doc_title": "Paper_A",
                "text": {"po_1": {"text": "Text evidence"}},
            }
        ],
        "table.parquet": [
            {
                "doc_title": "Paper_A",
                "component_id": "t_1",
                "table": '[[{"text": "Model"}], [{"text": "Ours"}]]',
            }
        ],
        "image.parquet": [
            {
                "doc_title": "Paper_A",
                "image": {
                    "i_1": {
                        "image_name": "nested/figure.png",
                        "caption": "A figure",
                    }
                },
            }
        ],
        "image_dump.parquet": [
            {"image_name": "nested/figure.png", "image_bytes": png}
        ],
        "dev.parquet": [
            {
                "qid": "q1",
                "question": "Compare the evidence",
                "answer": "Ours",
                "evidence": [["Paper_A", "t_1"], ["Paper_A", "i_1"]],
            }
        ],
    }
    monkeypatch.setattr(
        multimodalqa,
        "_iter_parquet_rows",
        lambda path: iter(parquet_rows[path.name]),
    )

    layout = BenchmarkLayout.create("multimodalqa", tmp_path / "benchmarks")
    report = prepare_multimodalqa(layout, source=source)
    samples = read_jsonl(layout.samples("all"))

    assert report["graph_mode"] == "official_parquet_component_graph"
    assert report["samples"] == 1
    assert report["missing_images"] == []
    assert any((layout.processed / "images").iterdir())
    assert samples[0]["required_modalities"] == ["table", "image"]


def test_mmlongbench_uses_gold_evidence_pages() -> None:
    from paper_rag.domain import EvidenceNode
    from paper_rag.evidence_graph import EvidenceGraph

    graph = EvidenceGraph()
    for page in range(1, 4):
        graph.add_node(
            EvidenceNode(
                _page_node_id("mmlongbench_doc", "paper.pdf", page),
                "paper.pdf",
                NodeType.FIGURE,
                image_path=f"page-{page}.png",
            )
        )
    samples, invalid = _mmlong_samples(
        [
            {
                "doc_id": "paper.pdf",
                "question": "Which model wins?",
                "answer": "Ours",
                "answer_format": "Str",
                "evidence_pages": "[1, 2]",
                "evidence_sources": "['Table', 'Chart']",
            }
        ],
        graph,
    )

    assert not invalid
    assert samples[0]["required_modalities"] == ["table", "figure"]
    assert len(samples[0]["relevant_node_ids"]) == 2
    assert len(samples[0]["candidate_node_ids"]) == 3


def test_peerqa_redistributable_scope_allows_licensed_subset(tmp_path) -> None:
    layout = BenchmarkLayout.create("peerqa", tmp_path)
    write_json(
        layout.processed / "prepare_report.json",
        {
            "evaluation_scope": "official_redistributable_papers",
            "missing_papers": ["openreview/paper"],
            "download_errors": {"openreview/paper": "HTTP 403"},
            "parse_errors": {"openreview/paper": "MinerU failed"},
        },
    )

    _validate_preparation(layout)


def test_peerqa_complete_scope_rejects_missing_inputs(tmp_path) -> None:
    layout = BenchmarkLayout.create("peerqa", tmp_path)
    write_json(
        layout.processed / "prepare_report.json",
        {
            "evaluation_scope": "official_all_papers",
            "missing_papers": ["openreview/paper"],
        },
    )

    try:
        _validate_preparation(layout)
    except RuntimeError as error:
        assert "missing_papers" in str(error)
    else:
        raise AssertionError("Incomplete full PeerQA preparation was accepted")


def test_prepare_console_report_summarizes_details() -> None:
    summary = _report_summaries(
        {
            "peerqa": {
                "dataset": "peerqa",
                "nodes": 10,
                "missing_papers": ["a", "b"],
                "download_errors": {"a": "403"},
            }
        }
    )["peerqa"]

    assert summary == {
        "dataset": "peerqa",
        "nodes": 10,
        "missing_papers_count": 2,
        "missing_images_count": 0,
        "missing_evidence_count": 0,
        "download_errors_count": 1,
        "parse_errors_count": 0,
    }


def test_training_report_counts_splits_and_relation_triples(tmp_path) -> None:
    layout = BenchmarkLayout.create("dataset", tmp_path)
    graph = EvidenceGraph()
    graph.extend(
        [
            EvidenceNode("train:a", "train", NodeType.SENTENCE, text="a"),
            EvidenceNode("train:b", "train", NodeType.SENTENCE, text="b"),
            EvidenceNode("train:c", "train", NodeType.SENTENCE, text="c"),
            EvidenceNode("dev:a", "dev", NodeType.TABLE, text="table"),
            EvidenceNode("test:a", "test", NodeType.FIGURE, image_path="test.png"),
        ],
        [EvidenceEdge("train:a", "train:b", RelationType.NEXT_SENTENCE)],
    )
    save_graph(graph, layout.graph)
    for split, paper, candidates in (
        ("train", "train", ["train:a", "train:b", "train:c"]),
        ("dev", "dev", ["dev:a"]),
        ("test", "test", ["test:a"]),
    ):
        write_jsonl(
            layout.samples(split),
            [
                {
                    "query_id": split,
                    "paper_id": paper,
                    "query": split,
                    "relevant_node_ids": candidates[:1],
                    "candidate_node_ids": candidates,
                }
            ],
        )

    statistics = benchmark_split_statistics(layout)

    assert statistics["train"] == {
        "questions": 1,
        "papers": 1,
        "nodes": 3,
        "node_types": {"Sentence": 3},
        "relation_types": {"next_sentence": 1},
        "relation_triples": 1,
    }
    assert statistics["dev"]["node_types"] == {"Table": 1}
    assert statistics["test"]["node_types"] == {"Figure": 1}
    _validate_training_split(layout)


def test_training_rejects_test_paper_leakage(tmp_path) -> None:
    layout = BenchmarkLayout.create("dataset", tmp_path)
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("shared", "paper", NodeType.SENTENCE, text="shared"))
    save_graph(graph, layout.graph)
    row = {
        "paper_id": "paper",
        "query": "question",
        "relevant_node_ids": ["shared"],
        "candidate_node_ids": ["shared"],
    }
    write_jsonl(layout.samples("train"), [{**row, "query_id": "train"}])
    write_jsonl(layout.samples("test"), [{**row, "query_id": "test"}])

    with pytest.raises(ValueError, match="papers overlap"):
        _validate_training_split(layout)


def test_processed_schema_rejects_stale_artifacts(tmp_path) -> None:
    layout = BenchmarkLayout.create("dataset", tmp_path)
    write_json(layout.processed / "prepare_report.json", {"dataset": "dataset"})

    with pytest.raises(RuntimeError, match="Prepare dataset again"):
        _validate_processed_schema(layout)

    write_json(
        layout.processed / "prepare_report.json",
        {"dataset": "dataset", "schema_version": PROCESSED_SCHEMA_VERSION},
    )
    _validate_processed_schema(layout)


def test_training_rejects_empty_held_out_split(tmp_path) -> None:
    layout = BenchmarkLayout.create("dataset", tmp_path)
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:s", "p", NodeType.SENTENCE, text="answer"))
    save_graph(graph, layout.graph)
    write_jsonl(
        layout.samples("train"),
        [{"query_id": "q", "query": "question", "relevant_node_ids": ["p:s"]}],
    )
    write_jsonl(layout.samples("dev"), [])

    with pytest.raises(ValueError, match="dev split is empty"):
        _validate_training_split(layout)


def test_zip_validation_rejects_html_cache(tmp_path) -> None:
    archive = tmp_path / "dataset.zip"
    archive.write_text("<html>access denied</html>", encoding="utf-8")
    assert not _valid_download(archive)


def test_zip_validation_accepts_and_extracts_archive(tmp_path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("qa.jsonl", "{}\n")

    assert _valid_download(archive)
    output = extract_zip(archive, tmp_path / "output")
    assert (output / "qa.jsonl").read_text(encoding="utf-8") == "{}\n"


def test_download_validation_checks_pdf_and_jsonl_signatures(tmp_path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"<html>rate limited</html>")
    assert not _valid_download(pdf)
    pdf.write_bytes(b"%PDF-1.7\n")
    assert _valid_download(pdf)

    data = tmp_path / "qa.jsonl"
    data.write_text("<html>access denied</html>", encoding="utf-8")
    assert not _valid_download(data)
    data.write_text('{"question": "why?"}\n', encoding="utf-8")
    assert _valid_download(data)
