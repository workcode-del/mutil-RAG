from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from paper_rag.bootstrap import build_embedder, build_vector_store
from paper_rag.config import load_yaml
from paper_rag.domain import NodeType
from paper_rag.evidence_graph import EvidenceGraph, load_graph, save_graph
from paper_rag.indexing import IndexingReport, compute_base_embeddings, upsert_base_embeddings
from paper_rag.parsing import MinerUAdapter


logger = logging.getLogger(__name__)


def _content_stem(path: Path) -> str:
    return path.stem.removesuffix("_content_list")


def ingest_pdfs(
    pdf_dir: str | Path,
    mineru_dir: str | Path,
    graph_path: str | Path,
    *,
    mineru_command: str = "mineru",
    force: bool = False,
) -> EvidenceGraph:
    source = Path(pdf_dir)
    parsed = Path(mineru_dir)
    content_lists = list(parsed.rglob("*_content_list.json")) if parsed.exists() else []
    parsed_names = {_content_stem(path) for path in content_lists}
    pdfs = sorted(source.rglob("*.pdf"))
    pending = pdfs if force else [pdf for pdf in pdfs if pdf.stem not in parsed_names]
    logger.info("Corpus parse: pdfs=%d pending=%d mineru=%s", len(pdfs), len(pending), parsed)
    for index, pdf in enumerate(pending, 1):
        logger.info("MinerU [%d/%d]: %s", index, len(pending), pdf.name)
        subprocess.run(
            [mineru_command, "-p", str(pdf), "-o", str(parsed), "-b", "pipeline"],
            check=True,
        )
    content_lists = list(parsed.rglob("*_content_list.json"))
    logger.info("Building evidence graph from %d MinerU outputs", len(content_lists))
    graph = EvidenceGraph()
    for content in sorted(content_lists):
        paper_id = _content_stem(content)
        paper = MinerUAdapter().from_json(content, paper_id)
        graph.extend(paper.nodes.values(), paper.edges)
    save_graph(graph, graph_path)
    logger.info(
        "Evidence graph ready: nodes=%d edges=%d output=%s",
        len(graph.nodes),
        len(graph.edges),
        graph_path,
    )
    return graph


def index_graph(
    graph_path: str | Path,
    config_path: str | Path,
    embedding_cache: str | Path,
    *,
    force_embeddings: bool = False,
    upsert_vector_store: bool = True,
) -> IndexingReport:
    config = load_yaml(config_path)
    graph = load_graph(graph_path)
    logger.info(
        "%s: nodes=%d graph=%s",
        "Vector index" if upsert_vector_store else "Embedding cache",
        len(graph.nodes),
        graph_path,
    )
    target = Path(embedding_cache)
    embeddings = (
        None
        if force_embeddings or not embedding_cache_is_current(graph_path, config, target)
        else _load_embedding_cache(target, graph, int(config["embedding"]["dimension"]))
    )
    if embeddings is None:
        embedding_config = config["embedding"]
        embeddings, report = compute_base_embeddings(
            graph,
            build_embedder(config),
            batch_size=int(embedding_config.get("text_batch_size", 16)),
            image_batch_size=int(embedding_config.get("image_batch_size", 8)),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving embedding cache: vectors=%d path=%s", len(embeddings), target)
        np.savez_compressed(target, **embeddings)
        _write_embedding_cache_metadata(graph_path, config, target)
        logger.info("Embedding cache ready: path=%s", target)
    else:
        figures = sum(node.node_type is NodeType.FIGURE for node in graph.nodes.values())
        tables = sum(node.node_type is NodeType.TABLE for node in graph.nodes.values())
        report = IndexingReport(
            len(graph.nodes) - figures - tables,
            figures,
            int(config["embedding"]["dimension"]),
            tables,
        )
        logger.info("Using embedding cache: vectors=%d path=%s", len(embeddings), target)
    if upsert_vector_store:
        store = build_vector_store(config)
        upsert_base_embeddings(store, graph, embeddings)
        if hasattr(store.client, "close"):
            store.client.close()
    logger.info(
        "Vector index ready: text=%d tables=%d figures=%d dimension=%d cache=%s",
        report.text_nodes,
        report.table_nodes,
        report.figure_nodes,
        report.dimension,
        target,
    )
    return report


def embedding_config_digest(config: dict[str, Any]) -> str:
    payload = {
        "embedding": config.get("embedding", {}),
        "model_download": config.get("model_download", {}),
        "runtime": config.get("runtime", {}),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def embedding_cache_is_current(
    graph_path: str | Path,
    config: dict[str, Any],
    embedding_cache: str | Path,
) -> bool:
    cache = Path(embedding_cache)
    metadata_path = _embedding_cache_metadata_path(cache)
    if not cache.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        metadata.get("graph_sha256") == hashlib.sha256(Path(graph_path).read_bytes()).hexdigest()
        and metadata.get("embedding_config_sha256") == embedding_config_digest(config)
    )


def _write_embedding_cache_metadata(
    graph_path: str | Path,
    config: dict[str, Any],
    embedding_cache: Path,
) -> None:
    metadata = {
        "version": 1,
        "graph_sha256": hashlib.sha256(Path(graph_path).read_bytes()).hexdigest(),
        "embedding_config_sha256": embedding_config_digest(config),
        "embedding_model": config["embedding"].get("model"),
        "dimension": int(config["embedding"]["dimension"]),
    }
    _embedding_cache_metadata_path(embedding_cache).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _embedding_cache_metadata_path(embedding_cache: Path) -> Path:
    return embedding_cache.with_suffix(f"{embedding_cache.suffix}.meta.json")


def _load_embedding_cache(
    path: Path, graph: EvidenceGraph, dimension: int
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as archive:
        if set(archive.files) != set(graph.nodes):
            return None
        embeddings = {node_id: archive[node_id] for node_id in archive.files}
    if any(vector.shape != (dimension,) for vector in embeddings.values()):
        return None
    return embeddings


def build_corpus(
    pdf_dir: str | Path,
    *,
    graph_path: str | Path,
    mineru_dir: str | Path,
    embedding_cache: str | Path,
    config_path: str | Path,
    mineru_command: str = "mineru",
    force: bool = False,
) -> dict[str, Any]:
    graph = ingest_pdfs(
        pdf_dir,
        mineru_dir,
        graph_path,
        mineru_command=mineru_command,
        force=force,
    )
    report = index_graph(graph_path, config_path, embedding_cache)
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "text_nodes": report.text_nodes,
        "figure_nodes": report.figure_nodes,
        "table_nodes": report.table_nodes,
        "dimension": report.dimension,
        "graph": str(Path(graph_path).resolve()),
        "embeddings": str(Path(embedding_cache).resolve()),
    }
