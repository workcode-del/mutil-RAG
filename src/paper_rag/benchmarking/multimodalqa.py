from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from zipfile import ZipFile

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
PARQUET_FILES = (
    "dev.parquet",
    "text.parquet",
    "table.parquet",
    "image.parquet",
    "image_dump.parquet",
)


def prepare_multimodalqa(
    layout: BenchmarkLayout,
    *,
    source: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(source) if source else _download_snapshot(layout.raw, force)
    parquet = {name: _find_file(root, name) for name in PARQUET_FILES}
    present_parquet = {name for name, path in parquet.items() if path is not None}
    if len(present_parquet) == len(PARQUET_FILES):
        graph, evidence_index, missing_images = _parquet_component_graph(
            parquet,
            layout.processed / "images",
            force=force,
        )
        rows = list(_iter_parquet_rows(_required_path(parquet, "dev.parquet")))
        graph_mode = "official_parquet_component_graph"
    else:
        qa_path = _find_file(root, "QAs_dev_labeled.json")
        documents = _resolve_component_dir(root, "parsed_documents", force=force)
        images = _resolve_component_dir(root, "image_components", force=force)
        if qa_path is None or documents is None or images is None:
            available = sorted(
                path.name for path in root.iterdir() if path.name != ".cache"
            )
            parquet_hint = ""
            if present_parquet:
                missing = sorted(set(PARQUET_FILES) - present_parquet)
                parquet_hint = f" Incomplete Parquet snapshot; missing: {missing}."
            raise FileNotFoundError(
                "MultimodalQA source must contain either the five Parquet files "
                f"{list(PARQUET_FILES)}, or QAs_dev_labeled.json plus parsed_documents "
                "and image_components (directories or matching ZIP archives)."
                f"{parquet_hint} Found at {root}: {available}"
            )
        documents = documents / "dev" if (documents / "dev").is_dir() else documents
        images = images / "dev" if (images / "dev").is_dir() else images
        graph, evidence_index, missing_images = _component_graph(documents, images)
        rows = json.loads(qa_path.read_text(encoding="utf-8"))
        graph_mode = "official_component_graph"
    samples, missing_evidence = _samples(rows, evidence_index)
    _validate_prepared(graph, rows, samples, missing_evidence, graph_mode)
    save_graph(graph, layout.graph)
    write_jsonl(layout.samples("all"), samples)
    split = connected_grouped_split(samples, members_key="split_group_ids")
    for name, values in split.items():
        write_jsonl(layout.samples(name), values)
    report = {
        "dataset": "multimodalqa",
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "graph_mode": graph_mode,
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
                _add_component(
                    graph,
                    index,
                    missing_images,
                    title=paper_id,
                    component_id=str(component_id),
                    node_type=node_type,
                    component=component,
                    image_lookup=image_lookup,
                )
    return graph, index, missing_images


def _parquet_component_graph(
    paths: dict[str, Path | None],
    image_root: Path,
    *,
    force: bool,
) -> tuple[EvidenceGraph, dict[tuple[str, str], str], list[str]]:
    image_lookup = _restore_parquet_images(
        _required_path(paths, "image_dump.parquet"), image_root, force=force
    )
    graph = EvidenceGraph()
    index: dict[tuple[str, str], str] = {}
    missing_images: list[str] = []
    for filename, kind, node_type in (
        ("text.parquet", "text", NodeType.SENTENCE),
        ("table.parquet", "table", NodeType.TABLE),
        ("image.parquet", "image", NodeType.FIGURE),
    ):
        for row in _iter_parquet_rows(_required_path(paths, filename)):
            title, component_id, component = _official_parquet_component(row, kind)
            _add_component(
                graph,
                index,
                missing_images,
                title=title,
                component_id=component_id,
                node_type=node_type,
                component=component,
                image_lookup=image_lookup,
            )
    return graph, index, missing_images


def _add_component(
    graph: EvidenceGraph,
    index: dict[tuple[str, str], str],
    missing_images: list[str],
    *,
    title: str,
    component_id: str,
    node_type: NodeType,
    component: dict[str, Any],
    image_lookup: dict[str, Path],
) -> None:
    node_id = f"multimodalqa::{safe_name(title)}::{component_id}"
    attributes = {
        key: component[key]
        for key in ("heading_path", "hyperlinks", "label_id")
        if key in component
    }
    if node_type is NodeType.FIGURE:
        filename = str(component.get("image_name") or component.get("filename") or "")
        image = image_lookup.get(filename) or image_lookup.get(Path(filename).name)
        if image is None:
            missing_images.append(f"{title}:{component_id}:{filename}")
            return
        node = EvidenceNode(
            node_id,
            title,
            node_type,
            image_path=str(image),
            provenance={"dataset": "MultimodalQA", "component_id": component_id},
            attributes={**attributes, "text_view": _caption_text(component)},
        )
    else:
        text = (
            str(component.get("text") or "")
            if node_type is NodeType.SENTENCE
            else _table_text(component)
        )
        if not text:
            return
        table_image = None
        if node_type is NodeType.TABLE:
            table_image, absent = _table_image(component, image_lookup)
            missing_images.extend(
                f"{title}:{component_id}:{filename}" for filename in absent
            )
        node = EvidenceNode(
            node_id,
            title,
            node_type,
            text=text,
            image_path=str(table_image) if table_image else None,
            provenance={"dataset": "MultimodalQA", "component_id": component_id},
            attributes=attributes,
        )
    graph.add_node(node)
    index[(title, component_id)] = node_id
    if node_type is not NodeType.FIGURE or not (caption := _caption_text(component)):
        return
    caption_id = f"{node_id}:caption"
    graph.add_node(
        EvidenceNode(
            caption_id,
            title,
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


def _official_parquet_component(
    row: dict[str, Any], kind: str
) -> tuple[str, str, dict[str, Any]]:
    """Decode one row exactly as the dataset's official load.py does."""
    required = {
        "doc_title",
        "component_id",
        "heading_path",
        "hyperlinks",
        "component",
        "label_id",
    }
    missing = required - row.keys()
    if missing:
        raise RuntimeError(
            f"Unsupported MultimodalQA {kind}.parquet schema; "
            f"missing={sorted(missing)}, columns={sorted(row)}"
        )
    title = str(row["doc_title"] or "")
    component_id = str(row["component_id"] or "")
    if not title or not component_id:
        raise RuntimeError(
            f"Invalid MultimodalQA {kind}.parquet row: "
            "doc_title and component_id must be non-empty"
        )

    if kind == "text":
        component: dict[str, Any] = {"text": str(row["component"] or "")}
    else:
        payload = _official_json(row["component"], "component", title, component_id)
        if kind == "table" and isinstance(payload, list):
            component = {"table": payload}
        elif kind == "image" and isinstance(payload, dict):
            component = dict(payload)
        else:
            raise RuntimeError(
                f"Invalid MultimodalQA {kind}.parquet component for "
                f"{title}:{component_id}; decoded type={type(payload).__name__}"
            )
    heading_path = _official_json(
        row["heading_path"], "heading_path", title, component_id
    )
    hyperlinks = _official_json(
        row["hyperlinks"], "hyperlinks", title, component_id
    )
    if not isinstance(heading_path, list) or not isinstance(hyperlinks, list):
        raise RuntimeError(
            f"Invalid MultimodalQA metadata for {title}:{component_id}; "
            "heading_path and hyperlinks must decode to lists"
        )
    component.update(
        {
            "heading_path": heading_path,
            "hyperlinks": hyperlinks,
            "label_id": row["label_id"],
        }
    )
    return title, component_id, component


def _official_json(value: Any, field: str, title: str, component_id: str) -> Any:
    if not isinstance(value, str):
        raise RuntimeError(
            f"Invalid MultimodalQA {field} for {title}:{component_id}; "
            f"expected JSON string, got {type(value).__name__}"
        )
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid MultimodalQA {field} JSON for {title}:{component_id}"
        ) from exc


def _decode_nested(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _restore_parquet_images(
    path: Path, image_root: Path, *, force: bool
) -> dict[str, Path]:
    image_root.mkdir(parents=True, exist_ok=True)
    lookup: dict[str, Path] = {}
    invalid_rows = 0
    for row in _iter_parquet_rows(path):
        missing = {"image_name", "byte_data"} - row.keys()
        if missing:
            raise RuntimeError(
                "Unsupported MultimodalQA image_dump.parquet schema; "
                f"missing={sorted(missing)}, columns={sorted(row)}"
            )
        name = _image_name(row)
        payload = _image_payload(row)
        if not name or payload is None:
            invalid_rows += 1
            continue
        target = image_root / _restored_image_name(name, payload)
        valid = valid_image_file(target)
        if force or not valid:
            target.write_bytes(payload)
            valid = valid_image_file(target)
        if not valid:
            invalid_rows += 1
            continue
        resolved = target.resolve()
        lookup[name] = resolved
        lookup[Path(name).name] = resolved
    if invalid_rows:
        logger.warning(
            "Skipped %d invalid MultimodalQA image_dump rows", invalid_rows
        )
    if not lookup:
        raise RuntimeError(
            "MultimodalQA image_dump.parquet produced no valid images; expected "
            "official columns image_name and byte_data"
        )
    return lookup


def _validate_prepared(
    graph: EvidenceGraph,
    rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    missing_evidence: list[str],
    graph_mode: str,
) -> None:
    if not graph.nodes:
        raise RuntimeError("MultimodalQA preparation produced an empty graph")
    if graph_mode == "official_parquet_component_graph":
        present = {node.node_type for node in graph.nodes.values()}
        required = {NodeType.SENTENCE, NodeType.TABLE, NodeType.FIGURE}
        if absent := required - present:
            raise RuntimeError(
                "MultimodalQA Parquet conversion lost required modalities: "
                f"{sorted(node_type.value for node_type in absent)}"
            )
    if rows and not samples:
        examples = ", ".join(missing_evidence[:5]) or "no gold evidence found"
        raise RuntimeError(
            "MultimodalQA preparation produced zero usable samples from "
            f"{len(rows)} questions. First evidence errors: {examples}"
        )


def _image_name(row: dict[str, Any]) -> str:
    value = row.get("image_name")
    return value if isinstance(value, str) else ""


def _image_payload(row: dict[str, Any]) -> bytes | None:
    payload = row.get("byte_data")
    return (
        bytes(payload)
        if isinstance(payload, (bytes, bytearray, memoryview))
        else None
    )


def _restored_image_name(name: str, payload: bytes) -> str:
    suffix = Path(name).suffix.casefold()
    known = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    if suffix not in known:
        suffix = _image_suffix(payload)
    return f"{safe_name(name)}{suffix}"


def _image_suffix(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if payload.startswith(b"BM"):
        return ".bmp"
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    return ".png"


def _iter_parquet_rows(path: Path):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise RuntimeError(
            "Reading the current MultimodalQA snapshot requires PyArrow. "
            "Install the unified project dependencies."
        ) from exc
    source = parquet.ParquetFile(path)
    batch_size = 32 if path.name == "image_dump.parquet" else 2048
    for batch in source.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _required_path(paths: dict[str, Path | None], name: str) -> Path:
    path = paths[name]
    if path is None:  # pragma: no cover - guarded by snapshot validation
        raise FileNotFoundError(name)
    return path


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
        raw_evidence = row.get("evidences") or row.get("evidence") or []
        for evidence in raw_evidence:
            if isinstance(evidence, dict):
                component_id = str(
                    evidence.get("gold_component_id")
                    or evidence.get("component_id")
                    or evidence.get("1")
                    or ""
                )
                title = str(
                    evidence.get("gold_webpage_title")
                    or evidence.get("doc_title")
                    or evidence.get("0")
                    or ""
                )
                modality = str(
                    evidence.get("mmqa_doc_modality")
                    or _component_modality(component_id)
                )
            elif isinstance(evidence, (list, tuple)) and len(evidence) >= 2:
                title, component_id = str(evidence[0]), str(evidence[1])
                modality = _component_modality(component_id)
            else:
                missing.append(f"{row.get('qid')}:invalid-evidence:{evidence!r}")
                complete = False
                continue
            node_id = index.get((title, component_id))
            if node_id is None and len(by_component[component_id]) == 1:
                node_id = by_component[component_id][0]
            if node_id is None:
                missing.append(f"{row.get('qid')}:{title}:{component_id}")
                complete = False
                continue
            gold.append(node_id)
            paper_ids.add(paper_by_node[node_id])
            modalities.append(modality)
        if not complete or not gold:
            continue
        answers = row.get("answers", [])
        answer = row.get("answer", "")
        if not answer and answers and isinstance(answers[0], dict):
            answer = answers[0].get("answer", "")
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


def _component_modality(component_id: str) -> str:
    prefix = component_id.casefold().split("_", 1)[0]
    if prefix in {"i", "image"}:
        return "image"
    if prefix in {"t", "table"}:
        return "table"
    return "text"


def _table_text(component: dict[str, Any]) -> str:
    if component.get("text"):
        return " ".join(str(component["text"]).split())
    refs = _decode_nested(component.get("refs", {}))
    if not isinstance(refs, dict):
        refs = {}
    rows: list[str] = []
    table = _decode_nested(component.get("table", []))
    for row in table if isinstance(table, list) else []:
        if not isinstance(row, list):
            row = [row]
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
    refs = _decode_nested(component.get("refs", {}))
    if not isinstance(refs, dict):
        refs = {}
    filenames: list[str] = []
    table = _decode_nested(component.get("table", []))
    for row in table if isinstance(table, list) else []:
        if not isinstance(row, list):
            row = [row]
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
