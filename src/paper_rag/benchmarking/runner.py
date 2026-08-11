from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from paper_rag.benchmarking.base import BenchmarkLayout, write_json
from paper_rag.bootstrap import build_deployed_pipeline, build_retriever_config
from paper_rag.config import load_yaml
from paper_rag.evaluation import evaluate, load_samples, save_report
from paper_rag.evaluation.comparison import save_comparison
from paper_rag.indexing import compute_base_embeddings, upsert_base_embeddings
from paper_rag.models.cached_scorer import CachedHGTScorer
from paper_rag.retrieval import build_evidence_retriever


@dataclass(frozen=True, slots=True)
class BenchmarkSystem:
    candidate_backend: str
    retrieval_method: str
    reranker: bool = False
    hgt: bool = False


SYSTEMS = {
    "bm25": BenchmarkSystem("bm25", "top_k"),
    "dense": BenchmarkSystem("embedding", "top_k"),
    "dense_reranker": BenchmarkSystem("embedding", "top_k", reranker=True),
    "one_hop": BenchmarkSystem("embedding", "one_hop"),
    "ppr": BenchmarkSystem("embedding", "ppr"),
    "pcst": BenchmarkSystem("embedding", "pcst"),
    "pcst_closure": BenchmarkSystem("embedding", "pcst_closure"),
    "ec_bfr": BenchmarkSystem("embedding", "ec_bfr"),
    "ec_bfr_reranker": BenchmarkSystem("embedding", "ec_bfr", reranker=True),
    "full": BenchmarkSystem("embedding", "ec_bfr", reranker=True, hgt=True),
}

DEFAULT_SYSTEMS = tuple(name for name in SYSTEMS if name != "full")


def run_benchmark(
    layout: BenchmarkLayout,
    *,
    config_path: str | Path,
    split: str,
    systems: list[str] | tuple[str, ...] = DEFAULT_SYSTEMS,
    hgt_artifacts: str | Path | None = None,
    enable_generator: bool = False,
    reindex: bool = False,
    selection_top_k: int = 10,
    per_type_top_k: int | None = None,
    cutoffs: tuple[int, ...] = (1, 3, 5, 10),
    allow_partial: bool = False,
) -> dict[str, Any]:
    unknown = set(systems) - SYSTEMS.keys()
    if unknown:
        raise ValueError(f"Unknown benchmark systems: {sorted(unknown)}")
    if "full" in systems and not hgt_artifacts:
        raise ValueError("The full system requires --hgt-artifacts")
    sample_path = layout.samples(_official_split(layout.name) if split == "official" else split)
    if not layout.graph.exists() or not sample_path.exists():
        raise FileNotFoundError(f"Prepare {layout.name} before running its benchmark")
    if not allow_partial:
        _validate_preparation(layout)
    selected = [SYSTEMS[name] for name in systems]
    if any(system.candidate_backend == "embedding" for system in selected):
        ensure_dense_index(layout, config_path, force=reindex)

    samples = load_samples(sample_path)
    config = load_yaml(config_path)
    retriever_config = build_retriever_config(config)
    report_paths: list[Path] = []
    summaries: dict[str, Any] = {}
    groups = {(system.candidate_backend, system.reranker) for system in selected}
    for backend, reranker_enabled in sorted(groups):
        pipeline = build_deployed_pipeline(
            layout.graph,
            config_path,
            enable_reranker=reranker_enabled,
            enable_generator=enable_generator,
            candidate_backend=backend,
            retrieval_method="top_k",
            selection_top_k=selection_top_k,
        )
        try:
            for name, system in zip(systems, selected, strict=True):
                if (system.candidate_backend, system.reranker) != (backend, reranker_enabled):
                    continue
                pipeline.forest_retriever = build_evidence_retriever(
                    system.retrieval_method,
                    pipeline.graph,
                    retriever_config,
                    selection_top_k=selection_top_k,
                )
                pipeline.graph_scorer = (
                    CachedHGTScorer(hgt_artifacts) if system.hgt else None
                )
                metadata = {
                    "dataset": layout.name,
                    "split": split,
                    "system": name,
                    "candidate_backend": backend,
                    "retrieval_method": system.retrieval_method,
                    "reranker": reranker_enabled,
                    "hgt": system.hgt,
                    "generator": enable_generator,
                    "selection_top_k": selection_top_k,
                    "per_type_top_k": per_type_top_k,
                    "ranking_cutoffs": cutoffs,
                    "scope": "sample",
                }
                report = evaluate(
                    pipeline,
                    samples,
                    cutoffs=cutoffs,
                    per_type_top_k=per_type_top_k,
                    scope_to_sample_papers=True,
                    metadata=metadata,
                )
                target = layout.reports / f"{split}_{name}.json"
                save_report(report, target)
                report_paths.append(target)
                summaries[name] = report["summary"]
        finally:
            _close_pipeline(pipeline)

    comparison, table = save_comparison(
        report_paths,
        layout.reports / f"{split}_comparison.csv",
    )
    summary = {
        "dataset": layout.name,
        "split": split,
        "samples": len(samples),
        "reports": [str(path.resolve()) for path in report_paths],
        "comparison": str(comparison),
        "table": table,
        "summaries": summaries,
    }
    write_json(layout.reports / f"{split}_summary.json", summary)
    return summary


def ensure_dense_index(
    layout: BenchmarkLayout,
    config_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    graph_digest = hashlib.sha256(layout.graph.read_bytes()).hexdigest()
    marker = layout.processed / "dense_index.json"
    config = load_yaml(config_path)
    vector = config["vector_store"]
    local_index_exists = (
        vector.get("mode", "local") != "local" or Path(vector["path"]).exists()
    )
    if marker.exists() and local_index_exists and not force:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("graph_sha256") == graph_digest:
            return marker

    pipeline = build_deployed_pipeline(
        layout.graph,
        config_path,
        enable_reranker=False,
        candidate_backend="embedding",
        retrieval_method="top_k",
    )
    try:
        if pipeline.embedder is None:
            raise RuntimeError("Dense indexing requires an embedder")
        embeddings, report = compute_base_embeddings(pipeline.graph, pipeline.embedder)
        upsert_base_embeddings(pipeline.vector_store, pipeline.graph, embeddings)
        np.savez_compressed(layout.processed / "base_embeddings.npz", **embeddings)
    finally:
        _close_pipeline(pipeline)
    write_json(
        marker,
        {
            "graph_sha256": graph_digest,
            "text_nodes": report.text_nodes,
            "figure_nodes": report.figure_nodes,
            "dimension": report.dimension,
        },
    )
    return marker


def _official_split(dataset: str) -> str:
    return "all" if dataset == "peerqa" else "test"


def _validate_preparation(layout: BenchmarkLayout) -> None:
    report_path = layout.processed / "prepare_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    problems = {
        key: report.get(key)
        for key in ("missing_papers", "missing_images", "download_errors", "parse_errors")
        if report.get(key)
    }
    if problems:
        raise RuntimeError(
            f"Incomplete {layout.name} preparation: {problems}. "
            "Fix the reported items or pass --allow-partial for a diagnostic run."
        )


def _close_pipeline(pipeline: Any) -> None:
    client = getattr(getattr(pipeline, "vector_store", None), "client", None)
    if client and hasattr(client, "close"):
        client.close()
