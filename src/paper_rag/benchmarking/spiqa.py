from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from paper_rag.benchmarking.base import PROCESSED_SCHEMA_VERSION, BenchmarkLayout, safe_name
from paper_rag.benchmarking.download import extract_zip, valid_image_file
from paper_rag.domain import EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.io import write_json, write_jsonl


logger = logging.getLogger(__name__)
HF_DATASET = "google/spiqa"
SPLIT_FILES = {
    "train": "train_val/SPIQA_train.json",
    "dev": "train_val/SPIQA_val.json",
    "test": "test-A/SPIQA_testA.json",
}
IMAGE_ARCHIVES = {
    "train": "train_val/SPIQA_train_val_Images.zip",
    "dev": "train_val/SPIQA_train_val_Images.zip",
    "test": "test-A/SPIQA_testA_Images_224px.zip",
}


def prepare_spiqa(
    layout: BenchmarkLayout,
    *,
    source: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(source) if source else _download_snapshot(layout.raw, force)
    metadata = {
        split: _load_metadata(root, relative_path)
        for split, relative_path in SPLIT_FILES.items()
    }
    image_roots: dict[str, Path] = {}
    extracted_archives: dict[str, Path] = {}
    for split in SPLIT_FILES:
        archive_path = IMAGE_ARCHIVES[split]
        if archive_path not in extracted_archives:
            extracted_archives[archive_path] = _ensure_split_images(
                root,
                metadata[split],
                archive_path,
                layout.raw / "extracted" / Path(archive_path).stem,
                force=force,
            )
        image_roots[split] = extracted_archives[archive_path]
    graph, samples, audit = _convert_splits(metadata, image_roots)

    save_graph(graph, layout.graph)
    for split, rows in samples.items():
        write_jsonl(layout.samples(split), rows)
    write_jsonl(layout.samples("all"), [row for split in SPLIT_FILES for row in samples[split]])
    report = {
        "dataset": "spiqa",
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "graph_mode": "official_figure_table_caption_graph",
        "evaluation_scope": "official_all_papers",
        "official_test_split": "test-A",
        "samples": {split: len(rows) for split, rows in samples.items()},
        "nodes": len(graph.nodes),
        "papers": len({node.paper_id for node in graph.nodes.values()}),
        **audit,
    }
    write_json(layout.processed / "prepare_report.json", report)
    return report


def _convert_splits(
    metadata: dict[str, dict[str, Any]],
    image_roots: dict[str, Path],
) -> tuple[EvidenceGraph, dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    graph = EvidenceGraph()
    samples = {split: [] for split in metadata}
    missing_images: list[str] = []
    missing_evidence: list[str] = []

    for split, papers in metadata.items():
        image_lookup = _image_lookup(image_roots[split])
        for metadata_key, paper in papers.items():
            paper_id = str(paper.get("paper_id") or metadata_key)
            candidates: list[str] = []
            node_by_reference: dict[str, str] = {}
            for reference, figure in paper.get("all_figures", {}).items():
                reference = str(reference)
                image = image_lookup.get(reference) or image_lookup.get(Path(reference).name)
                if image is None:
                    missing_images.append(f"{split}:{paper_id}:{reference}")
                    continue
                content_type = str(figure.get("content_type", "figure")).casefold()
                node_type = NodeType.TABLE if content_type == "table" else NodeType.FIGURE
                caption = " ".join(str(figure.get("caption") or "").split())
                node_id = f"spiqa::{split}::{safe_name(paper_id)}::{safe_name(reference)}"
                graph.add_node(
                    EvidenceNode(
                        node_id,
                        paper_id,
                        node_type,
                        text=caption if node_type is NodeType.TABLE else None,
                        image_path=str(image.resolve()),
                        provenance={
                            "dataset": "SPIQA",
                            "split": split,
                            "reference": reference,
                        },
                        attributes={
                            "text_view": caption,
                            "figure_type": str(figure.get("figure_type") or ""),
                        },
                    )
                )
                candidates.append(node_id)
                node_by_reference[reference] = node_id
                if caption:
                    caption_id = f"{node_id}:caption"
                    graph.add_node(
                        EvidenceNode(
                            caption_id,
                            paper_id,
                            NodeType.CAPTION,
                            text=caption,
                            provenance={
                                "dataset": "SPIQA",
                                "split": split,
                                "reference": reference,
                            },
                        )
                    )
                    graph.add_edge(
                        EvidenceEdge(
                            caption_id,
                            node_id,
                            RelationType.CAPTION_OF,
                            mandatory_for_closure=True,
                        )
                    )

            for index, qa in enumerate(paper.get("qa", [])):
                reference = str(qa.get("reference") or "")
                relevant = node_by_reference.get(reference)
                query_id = f"spiqa::{split}::{paper_id}::{index}"
                if relevant is None:
                    missing_evidence.append(f"{query_id}:{reference}")
                    continue
                node_type = graph.nodes[relevant].node_type
                samples[split].append(
                    {
                        "query_id": query_id,
                        "paper_id": paper_id,
                        "paper_ids": [paper_id],
                        "query": str(qa.get("question") or ""),
                        "answer": str(qa.get("answer") or ""),
                        "relevant_node_ids": [relevant],
                        "candidate_node_ids": candidates,
                        "required_modalities": [
                            "table" if node_type is NodeType.TABLE else "figure"
                        ],
                    }
                )

    return graph, samples, {
        "missing_images": sorted(set(missing_images)),
        "missing_evidence": sorted(set(missing_evidence)),
    }


def _load_metadata(root: Path, relative_path: str) -> dict[str, Any]:
    direct = root / relative_path
    path = direct if direct.is_file() else next(root.rglob(Path(relative_path).name), None)
    if path is None:
        raise FileNotFoundError(f"SPIQA source is missing {relative_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"SPIQA metadata must be a paper mapping: {path}")
    return data


def _ensure_split_images(
    root: Path,
    metadata: dict[str, Any],
    archive_path: str,
    extraction_root: Path,
    *,
    force: bool,
) -> Path:
    references = {
        str(reference)
        for paper in metadata.values()
        for reference in paper.get("all_figures", {})
    }
    existing = _image_lookup(root)
    if references.issubset(existing):
        return root
    archive = root / archive_path
    if not archive.is_file():
        archive = next(root.rglob(Path(archive_path).name), None)
    if archive is None:
        raise FileNotFoundError(
            f"SPIQA images are not extracted and source is missing {archive_path}"
        )
    return extract_zip(archive, extraction_root, force=force)


def _image_lookup(root: Path) -> dict[str, Path]:
    return {
        key: path
        for path in root.rglob("*")
        if path.is_file() and valid_image_file(path)
        for key in (path.name, path.relative_to(root).as_posix())
    }


def _download_snapshot(target: Path, force: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise RuntimeError(
            "Install the unified dependencies or provide --dataset-source spiqa=PATH"
        ) from exc
    target.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            HF_DATASET,
            repo_type="dataset",
            local_dir=target,
            allow_patterns=[*SPLIT_FILES.values(), *set(IMAGE_ARCHIVES.values())],
            force_download=force,
        )
    )