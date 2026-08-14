from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from paper_rag.benchmarking.base import BenchmarkLayout, read_jsonl, write_json
from paper_rag.bootstrap import build_deployed_pipeline, build_retriever_config
from paper_rag.config import load_yaml
from paper_rag.evaluation import evaluate, load_samples, save_report
from paper_rag.evaluation.comparison import save_comparison
from paper_rag.models.cached_scorer import CachedHGTScorer
from paper_rag.retrieval import build_evidence_retriever
from paper_rag.training import build_query_pairs, embed_training_queries, train_hgt
from paper_rag.workflow import index_graph


logger = logging.getLogger(__name__)
MIN_BATCH_QUERY_COSINE = 0.999


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
    query_batch_size: int = 64,
) -> dict[str, Any]:
    if "full" in systems and not hgt_artifacts:
        raise ValueError("The full system requires --hgt-artifacts")
    sample_path = layout.samples(_official_split(layout.name) if split == "official" else split)
    if not layout.graph.exists() or not sample_path.exists():
        raise FileNotFoundError(f"Prepare {layout.name} before running its benchmark")
    if not allow_partial:
        _validate_preparation(layout)
    if "full" in systems:
        _validate_hgt_artifacts(layout, sample_path, Path(hgt_artifacts))
    selected = [SYSTEMS[name] for name in systems]
    if any(system.candidate_backend == "embedding" for system in selected):
        ensure_dense_index(layout, config_path, force=reindex)

    samples = load_samples(sample_path)
    logger.info(
        "Benchmark start: dataset=%s split=%s samples=%d systems=%d",
        layout.name,
        split,
        len(samples),
        len(systems),
    )
    config = load_yaml(config_path)
    retriever_config = build_retriever_config(config)
    report_paths: list[Path] = []
    summaries: dict[str, Any] = {}
    query_vectors: dict[str, np.ndarray] | None = None
    query_embedding_ms = 0.0
    query_embedding_min_cosine: float | None = None
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
            if pipeline.embedder and query_vectors is None:
                query_vectors, query_embedding_ms, query_embedding_min_cosine = _embed_queries(
                    pipeline.embedder, samples, query_batch_size
                )
            for name, system in zip(systems, selected, strict=True):
                if (system.candidate_backend, system.reranker) != (backend, reranker_enabled):
                    continue
                pipeline.forest_retriever = build_evidence_retriever(
                    system.retrieval_method,
                    pipeline.graph,
                    retriever_config,
                    selection_top_k=selection_top_k,
                )
                logger.info(
                    "Benchmark system: dataset=%s system=%s backend=%s retrieval=%s reranker=%s hgt=%s",
                    layout.name,
                    name,
                    backend,
                    system.retrieval_method,
                    reranker_enabled,
                    system.hgt,
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
                    "query_batch_size": query_batch_size if pipeline.embedder else None,
                    "query_embedding_min_single_batch_cosine": (
                        query_embedding_min_cosine if pipeline.embedder else None
                    ),
                    "latency_mode": (
                        "batch_amortized_end_to_end" if pipeline.embedder else "online_end_to_end"
                    ),
                    "scope": "sample",
                }
                report = evaluate(
                    pipeline,
                    samples,
                    cutoffs=cutoffs,
                    per_type_top_k=per_type_top_k,
                    scope_to_sample_papers=True,
                    metadata=metadata,
                    query_vectors=query_vectors if pipeline.embedder else None,
                    query_embedding_ms=query_embedding_ms if pipeline.embedder else 0.0,
                )
                target = layout.reports / f"{split}_{name}.json"
                save_report(report, target)
                report_paths.append(target)
                summaries[name] = report["summary"]
                logger.info("Benchmark system complete: %s report=%s", name, target)
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
    logger.info("Benchmark complete: dataset=%s comparison=%s", layout.name, comparison)
    return summary


def _embed_queries(
    embedder, samples, batch_size: int
) -> tuple[dict[str, np.ndarray], float, float]:
    if batch_size < 1:
        raise ValueError("--query-batch-size must be positive")
    started = perf_counter()
    vectors: dict[str, np.ndarray] = {}
    logger.info(
        "Embedding benchmark queries: samples=%d batch_size=%d", len(samples), batch_size
    )
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        encoded = embedder.embed_queries([sample.query.query for sample in batch])
        vectors.update(
            {sample.query_id: vector for sample, vector in zip(batch, encoded, strict=True)}
        )
        logger.info(
            "Query embedding progress: %d/%d",
            min(start + batch_size, len(samples)),
            len(samples),
        )
    if len(vectors) != len(samples):
        raise ValueError("Benchmark query IDs must be unique")
    elapsed_ms = (perf_counter() - started) * 1000.0 / len(samples)
    checks = [samples[0], samples[-1]] if len(samples) > 1 else samples
    similarities = []
    for sample in checks:
        single = embedder.embed_queries([sample.query.query])[0]
        batched = vectors[sample.query_id]
        similarities.append(
            float(single @ batched / (np.linalg.norm(single) * np.linalg.norm(batched)))
        )
    min_cosine = min(similarities)
    if min_cosine < MIN_BATCH_QUERY_COSINE:
        raise RuntimeError(f"Batched query embedding mismatch: cosine={min_cosine:.6f}")
    logger.info("Query embeddings verified: min_cosine=%.6f", min_cosine)
    return vectors, elapsed_ms, min_cosine


def train_benchmark_index(
    layout: BenchmarkLayout,
    *,
    config_path: str | Path,
    output: str | Path,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    relation_weight: float = 0.2,
    seed: int = 42,
    device: str = "cuda",
    reindex: bool = False,
) -> dict[str, Any]:
    logger.info("Benchmark HGT training start: dataset=%s", layout.name)
    ensure_dense_index(layout, config_path, force=reindex)
    work = layout.processed / "training"
    pairs = build_query_pairs(
        layout.graph,
        layout.samples("train"),
        work / "query_pairs.jsonl",
        embeddings_path=layout.processed / "base_embeddings.npz",
        seed=seed,
    )
    queries = embed_training_queries(
        pairs,
        work / "query_embeddings.npz",
        config_path,
        batch_size=batch_size,
    )
    graph_config = load_yaml(config_path).get("graph_index", {})
    artifacts = train_hgt(
        layout.graph,
        layout.processed / "base_embeddings.npz",
        pairs,
        queries,
        output,
        epochs=epochs,
        learning_rate=learning_rate,
        relation_weight=relation_weight,
        seed=seed,
        device=device,
        hidden_dimension=int(graph_config.get("hidden_dimension", 256)),
        layers=int(graph_config.get("layers", 2)),
        heads=int(graph_config.get("heads", 4)),
    )
    metadata = json.loads((artifacts / "training.json").read_text(encoding="utf-8"))
    logger.info("Benchmark HGT training complete: dataset=%s output=%s", layout.name, artifacts)
    return {"dataset": layout.name, "artifacts": str(artifacts.resolve()), **metadata}


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
            logger.info("Using cached dense index: dataset=%s", layout.name)
            return marker

    logger.info("Building dense index: dataset=%s", layout.name)
    report = index_graph(
        layout.graph,
        config_path,
        layout.processed / "base_embeddings.npz",
    )
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
    keys = ["missing_images"]
    if report.get("evaluation_scope") == "official_all_papers":
        keys.extend(("download_errors", "parse_errors", "missing_papers"))
    problems = {key: report.get(key) for key in keys if report.get(key)}
    if problems:
        raise RuntimeError(
            f"Incomplete {layout.name} preparation: {problems}. "
            "Fix the reported items or pass --allow-partial for a diagnostic run."
        )


def _validate_hgt_artifacts(
    layout: BenchmarkLayout, samples: Path, artifacts: Path
) -> None:
    metadata_path = artifacts / "training.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    graph_digest = hashlib.sha256(layout.graph.read_bytes()).hexdigest()
    if metadata.get("graph_sha256") != graph_digest:
        raise ValueError(f"HGT artifacts do not match the {layout.name} graph")
    train_ids = set(metadata.get("train_query_ids", ()))
    evaluation_ids = {row["query_id"] for row in read_jsonl(samples)}
    if train_ids & evaluation_ids:
        raise ValueError("Training and evaluation queries overlap; use a held-out split")


def _close_pipeline(pipeline: Any) -> None:
    client = getattr(getattr(pipeline, "vector_store", None), "client", None)
    if client and hasattr(client, "close"):
        client.close()
