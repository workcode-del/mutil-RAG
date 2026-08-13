from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from paper_rag.benchmarking.base import (
    BenchmarkLayout,
    grouped_split,
    read_jsonl,
    write_json,
    write_jsonl,
)
from paper_rag.benchmarking.download import download_file, extract_zip
from paper_rag.domain import EvidenceNode, NodeType
from paper_rag.evidence_graph import EvidenceGraph, save_graph


logger = logging.getLogger(__name__)


HF_ROOT = "https://huggingface.co/datasets/MMDocIR/MMDocRAG/resolve/main"


def prepare_mmdocrag(
    layout: BenchmarkLayout,
    *,
    setting: int = 20,
    force: bool = False,
    download_pdfs: bool = False,
) -> dict[str, Any]:
    logger.info("Preparing MMDocRAG: root=%s setting=%d", layout.root, setting)
    if setting not in {15, 20}:
        raise ValueError("MMDocRAG setting must be 15 or 20")
    dev_path = _download(layout, f"dev_{setting}.jsonl", force)
    test_path = _download(layout, f"evaluation_{setting}.jsonl", force)
    images = _download(layout, "images.zip", force)
    image_root = extract_zip(images, layout.raw / "images", force=force)
    if download_pdfs:
        pdfs = _download(layout, "doc_pdfs.zip", force)
        extract_zip(pdfs, layout.raw / "pdfs", force=force)

    development = read_jsonl(dev_path)
    evaluation = read_jsonl(test_path)
    image_lookup = _file_lookup(image_root, {".jpg", ".jpeg", ".png", ".webp"})
    graph, missing_images = _build_quote_graph(
        [("development", row) for row in development]
        + [("test", row) for row in evaluation],
        image_lookup,
    )
    save_graph(graph, layout.graph)
    development_samples = [_sample(row, "development") for row in development]
    test_samples = [_sample(row, "test") for row in evaluation]
    write_jsonl(layout.samples("development"), development_samples)
    write_jsonl(layout.samples("test"), test_samples)
    split = grouped_split(
        development_samples,
        group_key="paper_id",
        train_percent=85,
        dev_percent=14,
    )
    write_jsonl(layout.samples("train"), split["train"])
    write_jsonl(layout.samples("dev"), split["dev"] + split["test"])
    report = {
        "dataset": "mmdocrag",
        "graph_mode": "official_quote_candidates",
        "setting": setting,
        "development_samples": len(development_samples),
        "test_samples": len(test_samples),
        "nodes": len(graph.nodes),
        "papers": len({node.paper_id for node in graph.nodes.values()}),
        "missing_images": sorted(missing_images),
        "official_candidate_scope": True,
    }
    write_json(layout.processed / "prepare_report.json", report)
    logger.info(
        "MMDocRAG ready: development=%d test=%d nodes=%d missing_images=%d",
        len(development_samples),
        len(test_samples),
        len(graph.nodes),
        len(missing_images),
    )
    return report


def _download(layout: BenchmarkLayout, filename: str, force: bool) -> Path:
    return download_file(
        f"{HF_ROOT}/{filename}?download=true",
        layout.raw / filename,
        force=force,
    )


def _build_quote_graph(
    rows: list[tuple[str, dict[str, Any]]], image_lookup: dict[str, Path]
) -> tuple[EvidenceGraph, set[str]]:
    graph = EvidenceGraph()
    missing_images: set[str] = set()
    for split, row in rows:
        for quote in _candidate_quotes(row):
            if quote.get("img_path"):
                raw_path = str(quote["img_path"])
                resolved = image_lookup.get(Path(raw_path).as_posix()) or image_lookup.get(
                    Path(raw_path).name
                )
                if resolved is None:
                    missing_images.add(raw_path)
                node = _quote_node(
                    split,
                    row,
                    quote,
                    NodeType.FIGURE,
                    image_path=str(resolved or raw_path),
                    text_view=str(quote.get("img_description", "")),
                )
            else:
                node = _quote_node(
                    split,
                    row,
                    quote,
                    NodeType.SENTENCE,
                    text=str(quote.get("text", "")),
                )
            graph.add_node(node)
    return graph, missing_images


def _sample(row: dict[str, Any], split: str) -> dict[str, Any]:
    qid = str(row["q_id"])
    quotes = _candidate_quotes(row)
    quote_ids = {
        str(quote["quote_id"]): _quote_node_id(split, row, quote) for quote in quotes
    }
    modalities = row.get("evidence_modality_type", row.get("evidence_modality"))
    return {
        "query_id": f"mmdocrag::{split}::{qid}",
        "paper_id": str(row["doc_name"]),
        "query": str(row["question"]),
        "answer": str(row.get("answer_interleaved") or row.get("answer_short", "")),
        "relevant_node_ids": [quote_ids[str(value)] for value in row["gold_quotes"]],
        "candidate_node_ids": list(quote_ids.values()),
        "required_modalities": _string_list(modalities),
        "question_type": row.get("question_type"),
    }


def _candidate_quotes(row: dict[str, Any]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for quote in [*row.get("text_quotes", []), *row.get("img_quotes", [])]:
        unique.setdefault(str(quote["quote_id"]), quote)
    return list(unique.values())


def _quote_node_id(split: str, row: dict[str, Any], quote: dict[str, Any]) -> str:
    modality = str(quote.get("type") or ("image" if quote.get("img_path") else "text"))
    return (
        f"mmdocrag::{split}::{row['doc_name']}::{modality}:"
        f"q{row['q_id']}:{quote['quote_id']}"
    )


def _file_lookup(root: Path, suffixes: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            result[path.name] = path.resolve()
            result[path.relative_to(root).as_posix()] = path.resolve()
    return result


def _quote_attributes(quote: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "mmdocrag",
        "layout_id": quote.get("layout_id"),
    }


def _quote_node(
    split: str,
    row: dict[str, Any],
    quote: dict[str, Any],
    node_type: NodeType,
    **content: Any,
) -> EvidenceNode:
    text_view = content.pop("text_view", None)
    attributes = _quote_attributes(quote)
    if text_view is not None:
        attributes["text_view"] = text_view
    return EvidenceNode(
        _quote_node_id(split, row, quote),
        str(row["doc_name"]),
        node_type,
        page=_optional_int(quote.get("page_id")),
        attributes=attributes,
        **content,
    )


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]
