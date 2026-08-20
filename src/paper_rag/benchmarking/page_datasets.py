from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

from paper_rag.benchmarking.base import BenchmarkLayout, grouped_split, safe_name
from paper_rag.benchmarking.download import download_file, valid_image_file
from paper_rag.domain import EvidenceNode, NodeType
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.io import write_json, write_jsonl


logger = logging.getLogger(__name__)
MMLONG_ROOT = "https://raw.githubusercontent.com/mayubo2333/MMLongBench-Doc/main/data"


def prepare_mmlongbench_doc(
    layout: BenchmarkLayout,
    *,
    source: str | Path | None = None,
    force: bool = False,
    max_documents: int | None = None,
) -> dict[str, Any]:
    root = Path(source) if source else layout.raw
    samples_path = _find_file(root, "samples.json")
    if samples_path is None:
        if source:
            raise FileNotFoundError("MMLongBench-Doc source must contain samples.json")
        samples_path = download_file(
            f"{MMLONG_ROOT}/samples.json", root / "samples.json", force=force
        )
    rows = json.loads(samples_path.read_text(encoding="utf-8"))
    answerable = [row for row in rows if _literal_list(row.get("evidence_pages"))]
    doc_ids = list(dict.fromkeys(str(row["doc_id"]) for row in answerable))
    if max_documents is not None:
        doc_ids = doc_ids[:max_documents]
    selected = [row for row in answerable if str(row["doc_id"]) in set(doc_ids)]

    documents = _find_dir(root, "documents") or root / "documents"
    pages_root = layout.raw / "page_images_1based"
    graph = EvidenceGraph()
    missing_papers: list[str] = []
    for position, doc_id in enumerate(doc_ids, 1):
        pdf = documents / doc_id
        if not pdf.exists():
            try:
                pdf = download_file(
                    f"{MMLONG_ROOT}/documents/{doc_id}",
                    layout.raw / "documents" / doc_id,
                    force=force,
                )
            except Exception as exc:  # network failures are summarized, not hidden
                logger.warning("MMLongBench-Doc PDF failed: %s (%s)", doc_id, exc)
                missing_papers.append(doc_id)
                continue
        for page_number, image_path in enumerate(
            _render_pdf(pdf, pages_root / Path(doc_id).stem), start=1
        ):
            graph.add_node(
                EvidenceNode(
                    _page_node_id("mmlongbench_doc", doc_id, page_number),
                    doc_id,
                    NodeType.FIGURE,
                    image_path=str(image_path.resolve()),
                    page=page_number,
                    provenance={"dataset": "MMLongBench-Doc", "page_level": True},
                )
            )
        logger.info("MMLongBench-Doc pages: %d/%d", position, len(doc_ids))

    samples, invalid = _mmlong_samples(selected, graph)
    _save_splits(layout, samples, group_key="paper_id")
    save_graph(graph, layout.graph)
    report = {
        "dataset": "mmlongbench_doc",
        "graph_mode": "official_page_images",
        "evaluation_scope": "official_all_papers" if max_documents is None else "partial_documents",
        "samples": len(samples),
        "nodes": len(graph.nodes),
        "papers": len({node.paper_id for node in graph.nodes.values()}),
        "missing_papers": missing_papers,
        "missing_evidence": invalid,
        "dropped_without_gold_pages": len(rows) - len(answerable),
    }
    write_json(layout.processed / "prepare_report.json", report)
    return report


def prepare_m3docvqa(
    layout: BenchmarkLayout,
    *,
    source: str | Path | None,
) -> dict[str, Any]:
    if source is None:
        raise ValueError(
            "M3DocVQA has no immutable official archive. Provide --dataset-source "
            "m3docvqa=<LILaC-compatible snapshot> containing M3DocVQA_dev_labeled.json "
            "and pdf_pages/dev."
        )
    root = Path(source)
    qa_path = _find_file(root, "M3DocVQA_dev_labeled.json")
    page_root = _find_dir(root, "pdf_pages")
    if qa_path is None or page_root is None:
        raise FileNotFoundError("M3DocVQA source must contain labeled QA and pdf_pages")
    rows = json.loads(qa_path.read_text(encoding="utf-8"))
    page_root = page_root / "dev" if (page_root / "dev").is_dir() else page_root
    graph, by_name, invalid_images = _page_image_graph("m3docvqa", page_root)
    samples: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        gold: list[str] = []
        modalities: list[str] = []
        for evidence in row.get("evidences", []):
            raw_page = str(evidence.get("gold_page", ""))
            node_id = by_name.get(Path(raw_page).name) or by_name.get(Path(raw_page).stem)
            if node_id is None:
                missing.append(f"{row.get('qid')}:{raw_page}")
                continue
            gold.append(node_id)
            modalities.append(str(evidence.get("mmqa_doc_modality", "image")))
        if not gold or len(gold) != len(row.get("evidences", [])):
            continue
        samples.append(
            {
                "query_id": f"m3docvqa::{row['qid']}",
                "query": str(row["question"]),
                "answer": _first_answer(row),
                "relevant_node_ids": list(dict.fromkeys(gold)),
                "required_modalities": list(dict.fromkeys(modalities)),
            }
        )
    _save_splits(layout, samples, group_key="query_id")
    save_graph(graph, layout.graph)
    report = {
        "dataset": "m3docvqa",
        "graph_mode": "official_open_domain_pages",
        "evaluation_scope": "official_all_papers",
        "samples": len(samples),
        "nodes": len(graph.nodes),
        "papers": len({node.paper_id for node in graph.nodes.values()}),
        "missing_images": invalid_images,
        "missing_evidence": missing,
    }
    write_json(layout.processed / "prepare_report.json", report)
    return report


def _mmlong_samples(
    rows: list[dict[str, Any]], graph: EvidenceGraph
) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    invalid: list[str] = []
    for index, row in enumerate(rows):
        doc_id = str(row["doc_id"])
        gold = [
            _page_node_id("mmlongbench_doc", doc_id, int(page))
            for page in _literal_list(row["evidence_pages"])
        ]
        unknown = [node_id for node_id in gold if node_id not in graph.nodes]
        if unknown:
            invalid.append(f"{doc_id}:{index}:{unknown}")
            continue
        candidates = [node_id for node_id, node in graph.nodes.items() if node.paper_id == doc_id]
        samples.append(
            {
                "query_id": f"mmlongbench_doc::{index}",
                "paper_id": doc_id,
                "query": str(row["question"]),
                "answer": str(row.get("answer", "")),
                "answer_type": str(row.get("answer_format", "free_text")),
                "relevant_node_ids": gold,
                "candidate_node_ids": candidates,
                "required_modalities": _modalities(row.get("evidence_sources")),
            }
        )
    return samples, invalid


def _page_image_graph(
    dataset: str, root: Path
) -> tuple[EvidenceGraph, dict[str, str], list[str]]:
    graph = EvidenceGraph()
    lookup: dict[str, str] = {}
    invalid: list[str] = []
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    for path in paths:
        if not valid_image_file(path):
            invalid.append(path.relative_to(root).as_posix())
            continue
        relative = path.relative_to(root).as_posix()
        paper_id = path.parent.name if path.parent != root else path.stem.rsplit("_", 1)[0]
        node_id = f"{dataset}::{relative}"
        graph.add_node(
            EvidenceNode(
                node_id,
                paper_id,
                NodeType.FIGURE,
                image_path=str(path.resolve()),
                provenance={"dataset": dataset, "page_level": True},
            )
        )
        lookup[path.name] = node_id
        lookup[path.stem] = node_id
    return graph, lookup, invalid


def _render_pdf(pdf: Path, output: Path) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise RuntimeError("Install the unified dependencies to render benchmark PDFs") from exc
    output.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf)
    paths: list[Path] = []
    try:
        for index, page in enumerate(document):
            target = output / f"page-{index}.png"
            if not target.exists():
                page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(target)
            paths.append(target)
    finally:
        document.close()
    return paths


def _save_splits(layout: BenchmarkLayout, rows: list[dict[str, Any]], *, group_key: str) -> None:
    write_jsonl(layout.samples("all"), rows)
    split = grouped_split(rows, group_key=group_key)
    for name, values in split.items():
        write_jsonl(layout.samples(name), values)


def _page_node_id(dataset: str, doc_id: str, page: int) -> str:
    return f"{dataset}::{safe_name(doc_id)}::page:{page}"


def _literal_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    parsed = ast.literal_eval(str(value or "[]"))
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def _modalities(value: Any) -> list[str]:
    mapping = {
        "table": "table",
        "chart": "figure",
        "figure": "figure",
        "pure-text (plain-text)": "text",
        "generalized-text (layout)": "text",
    }
    return list(
        dict.fromkeys(
            mapping.get(str(item).casefold(), str(item).casefold())
            for item in _literal_list(value)
        )
    )


def _first_answer(row: dict[str, Any]) -> str:
    answers = row.get("answers", [])
    if answers and isinstance(answers[0], dict):
        return str(answers[0].get("answer", ""))
    return str(answers[0] if answers else "")


def _find_file(root: Path, name: str) -> Path | None:
    direct = root / name
    return direct if direct.exists() else next(root.rglob(name), None)


def _find_dir(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_dir():
        return direct
    return next((path for path in root.rglob(name) if path.is_dir()), None)
