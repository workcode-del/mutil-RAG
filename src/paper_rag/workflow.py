from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from paper_rag.bootstrap import build_embedder, build_vector_store
from paper_rag.config import load_yaml
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
) -> IndexingReport:
    config = load_yaml(config_path)
    graph = load_graph(graph_path)
    logger.info("Vector index: nodes=%d graph=%s", len(graph.nodes), graph_path)
    store = build_vector_store(config)
    embeddings, report = compute_base_embeddings(graph, build_embedder(config))
    target = Path(embedding_cache)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving embedding cache: vectors=%d path=%s", len(embeddings), target)
    np.savez_compressed(target, **embeddings)
    logger.info("Embedding cache ready: path=%s", target)
    upsert_base_embeddings(store, graph, embeddings)
    if hasattr(store.client, "close"):
        store.client.close()
    logger.info(
        "Vector index ready: text=%d figures=%d dimension=%d cache=%s",
        report.text_nodes,
        report.figure_nodes,
        report.dimension,
        target,
    )
    return report


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
        "dimension": report.dimension,
        "graph": str(Path(graph_path).resolve()),
        "embeddings": str(Path(embedding_cache).resolve()),
    }
