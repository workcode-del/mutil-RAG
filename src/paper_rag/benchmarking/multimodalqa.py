from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
from typing import Any

from paper_rag.benchmarking.base import (
    PROCESSED_SCHEMA_VERSION,
    BenchmarkLayout,
    connected_grouped_split,
    safe_name,
)
from paper_rag.benchmarking.download import extract_zip, valid_image_file
from paper_rag.domain import EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.io import write_json, write_jsonl


logger = logging.getLogger(__name__)
HF_DATASET = "JoohyungYun/multimodalqa_doc"


def prepare_multimodalqa(
    layout: BenchmarkLayout,
    *,
    source: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(source) if source else _download_snapshot(layout.raw, force)
    qa_path = _find_file(root, "QAs_dev_labeled.json")
    documents = _resolve_component_dir(root, "parsed_documents", force=force)
    images = _resolve_component_dir(root, "image_components", force=force)
    if qa_path is None or documents is None or images is None:
        available = sorted(
            path.name for path in root.iterdir() if path.name != ".cache"
        )
        raise FileNotFoundError(
            "MultimodalQA source must contain QAs_dev_labeled.json, parsed_documents, "
            "and image_components (directories or matching ZIP archives). "
            f"Found at {root}: {available}"
        )
    documents = documents / "dev" if (documents / "dev").is_dir() else documents
    images = images / "dev" if (images / "dev").is_dir() else images
    graph, evidence_index, missing_images = _component_graph(documents, images)
    rows = json.loads(qa_path.read_text(encoding="utf-8"))
    samples, missing_evidence = _samples(rows, evidence_index)
    save_graph(graph, layout.graph)
    write_jsonl(layout.samples("all"), samples)
    split = connected_grouped_split(samples, members_key="split_group_ids")
    for name, values in split.items():
        write_jsonl(layout.samples(name), values)
    report = {
        "dataset": "multimodalqa",
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "graph_mode": "official_component_graph",
        "evaluation_scope": "official_all_papers",
        "samples": len(samples),
        "nodes": len(graph.nodes),
        "papers": len({node.paper_id for node in graph.nodes.values()}),
        "missing_images": missing_images,
        "missing_evidence": missing_evidence,
    }
    write_json(layout.processed / "prepare_report.json", report)
    return report


def _component_graph(
    document_root: Path, image_root: Path
) -> tuple[EvidenceGraph, dict[tuple[str, str], str], list[str]]:
    graph = EvidenceGraph()
    index: dict[tuple[str, str], str] = {}
    missing_images: list[str] = []
    image_lookup = {
        key: path.resolve()
        for path in image_root.rglob("*")
        if path.is_file() and valid_image_file(path)
        for key in (path.name, path.relative_to(image_root).as_posix())
    }
    for path in sorted(document_root.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        title = str(raw.get("title") or path.stem)
        paper_id = title
        for kind, node_type in (
            ("text", NodeType.SENTENCE),
            ("table", NodeType.TABLE),
            ("image", NodeType.FIGURE),
        ):
            for component_id, component in raw.get(kind, {}).items():
                node_id = f"multimodalqa::{safe_name(title)}::{component_id}"
                if node_type is NodeType.FIGURE:
                    filename = str(component.get("filename") or "")
                    image = image_lookup.get(filename) or image_lookup.get(Path(filename).name)
                    if image is None:
                        missing_images.append(f"{title}:{component_id}:{filename}")
                        continue
                    node = EvidenceNode(
                        node_id,
                        paper_id,
                        node_type,
                        image_path=str(image),
                        provenance={"dataset": "MultimodalQA", "component_id": component_id},
                        attributes={"text_view": _caption_text(component)},
                    )
                else:
                    text = (
                        str(component.get("text", ""))
                        if node_type is NodeType.SENTENCE
                        else _table_text(component)
                    )
                    if not text:
                        continue
                    table_image = None
                    if node_type is NodeType.TABLE:
                        table_image, absent = _table_image(component, image_lookup)
                        missing_images.extend(
                            f"{title}:{component_id}:{filename}" for filename in absent
                        )
                    node = EvidenceNode(
                        node_id,
                        paper_id,
                        node_type,
                        text=text,
                        image_path=str(table_image) if table_image else None,
                        provenance={"dataset": "MultimodalQA", "component_id": component_id},
                    )
                graph.add_node(node)
                index[(title, str(component_id))] = node_id
                if node_type is NodeType.FIGURE and (caption := _caption_text(component)):
                    caption_id = f"{node_id}:caption"
                    graph.add_node(
                        EvidenceNode(
                            caption_id,
                            paper_id,
                            NodeType.CAPTION,
                            text=caption,
                            provenance={"dataset": "MultimodalQA", "derived": "caption"},
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
    return graph, index, missing_images


def _samples(
    rows: list[dict[str, Any]], index: dict[tuple[str, str], str]
) -> tuple[list[dict[str, Any]], list[str]]:
    by_component: dict[str, list[str]] = defaultdict(list)
    paper_by_node: dict[str, str] = {}
    for (title, component_id), node_id in index.items():
        by_component[component_id].append(node_id)
        paper_by_node[node_id] = title
    samples: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        gold: list[str] = []
        modalities: list[str] = []
        paper_ids: set[str] = set()
        complete = True
        for evidence in row.get("evidences", []):
            component_id = str(evidence.get("gold_component_id", ""))
            title = str(evidence.get("gold_webpage_title", ""))
            node_id = index.get((title, component_id))
            if node_id is None and len(by_component[component_id]) == 1:
                node_id = by_component[component_id][0]
            if node_id is None:
                missing.append(f"{row.get('qid')}:{title}:{component_id}")
                complete = False
                continue
            gold.append(node_id)
            paper_ids.add(paper_by_node[node_id])
            modalities.append(str(evidence.get("mmqa_doc_modality", "text")))
        if not complete or not gold:
            continue
        answers = row.get("answers", [])
        answer = answers[0].get("answer", "") if answers and isinstance(answers[0], dict) else ""
        samples.append(
            {
                "query_id": f"multimodalqa::{row['qid']}",
                "split_group_ids": sorted(paper_ids),
                "query": str(row["question"]),
                "answer": str(answer),
                "relevant_node_ids": list(dict.fromkeys(gold)),
                "required_modalities": list(dict.fromkeys(modalities)),
            }
        )
    return samples, missing


def _table_text(component: dict[str, Any]) -> str:
    if component.get("text"):
        return " ".join(str(component["text"]).split())
    refs = component.get("refs", {})
    rows: list[str] = []
    for row in component.get("table", []):
        cells = []
        for cell in row:
            if isinstance(cell, dict) and "ref" in cell:
                cell = refs.get(str(cell["ref"]), cell)
            if isinstance(cell, dict):
                value = cell.get("text") or cell.get("value") or ""
            else:
                value = cell
            cells.append(" ".join(str(value).split()))
        rows.append("\t".join(value for value in cells if value))
    return "\n".join(row for row in rows if row)


def _caption_text(component: dict[str, Any]) -> str:
    caption = component.get("caption", "")
    if isinstance(caption, dict):
        caption = caption.get("text", "")
    return " ".join(str(caption or "").split())


def _table_image(
    component: dict[str, Any], image_lookup: dict[str, Path]
) -> tuple[Path | None, list[str]]:
    refs = component.get("refs", {})
    filenames: list[str] = []
    for row in component.get("table", []):
        for cell in row:
            if isinstance(cell, dict) and "ref" in cell:
                cell = refs.get(str(cell["ref"]), cell)
            image = cell.get("image", {}) if isinstance(cell, dict) else {}
            if image.get("filename"):
                filenames.append(str(image["filename"]))
    paths = [
        image_lookup.get(filename) or image_lookup.get(Path(filename).name)
        for filename in filenames
    ]
    return next((path for path in paths if path), None), [
        filename for filename, path in zip(filenames, paths, strict=True) if path is None
    ]


def _download_snapshot(target: Path, force: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise RuntimeError(
            "Install the unified dependencies or provide --dataset-source multimodalqa=PATH"
        ) from exc
    target.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            HF_DATASET,
            repo_type="dataset",
            local_dir=target,
            force_download=force,
        )
    )


def _find_file(root: Path, name: str) -> Path | None:
    direct = root / name
    return direct if direct.exists() else next(root.rglob(name), None)


def _find_dir(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_dir():
        return direct
    return next((path for path in root.rglob(name) if path.is_dir()), None)


def _resolve_component_dir(root: Path, name: str, *, force: bool) -> Path | None:
    existing = _find_dir(root, name)
    if existing is not None:
        return existing
    archive = _find_component_archive(root, name)
    if archive is None:
        return None
    destination = root if _archive_contains_dir(archive, name) else root / name
    logger.info("Extracting MultimodalQA component: %s", archive)
    extract_zip(archive, destination, force=force)
    return _find_dir(root, name)


def _find_component_archive(root: Path, name: str) -> Path | None:
    expected = _normalized_name(name)
    archives = sorted(root.rglob("*.zip"))
    named = next(
        (
            path
            for path in archives
            if expected in _normalized_name(path.stem)
        ),
        None,
    )
    return named or next(
        (path for path in archives if _archive_contains_dir(path, name)),
        None,
    )


def _archive_contains_dir(archive: Path, name: str) -> bool:
    expected = name.casefold()
    with ZipFile(archive) as bundle:
        return any(
            expected in {part.casefold() for part in Path(member.filename).parts}
            for member in bundle.infolist()
        )


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
