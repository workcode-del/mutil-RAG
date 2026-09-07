from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from paper_rag.benchmarking.base import (
    PROCESSED_SCHEMA_VERSION,
    BenchmarkLayout,
    read_jsonl,
    write_json,
)
from paper_rag.bootstrap import build_deployed_pipeline, build_retriever_config
from paper_rag.config import load_yaml
from paper_rag.evaluation import evaluate, load_samples, save_report
from paper_rag.evaluation.comparison import save_comparison
from paper_rag.embedding import ExactEmbeddingStore
from paper_rag.evidence_graph import EvidenceGraph, load_graph
from paper_rag.models.cached_scorer import CachedGraphScorer
from paper_rag.retrieval import build_evidence_retriever
from paper_rag.training import (
    build_query_pairs,
    count_relation_triples,
    embed_training_queries,
    train_graph_index,
)
from paper_rag.workflow import (
    embedding_cache_is_current,
    embedding_config_digest,
    index_graph,
)


logger = logging.getLogger(__name__)
MIN_BATCH_QUERY_COSINE = 0.999
OPEN_DOMAIN_DATASETS = {"m3docvqa", "multimodalqa"}


@dataclass(frozen=True, slots=True)
class BenchmarkSystem:
    candidate_backend: str
    retrieval_method: str
    reranker: bool = False
    graph_model: str | None = None


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
    "rgcn": BenchmarkSystem("embedding", "ec_bfr", reranker=True, graph_model="rgcn"),
    "full": BenchmarkSystem("embedding", "ec_bfr", reranker=True, graph_model="hgt"),
}

DEFAULT_SYSTEMS = tuple(name for name, system in SYSTEMS.items() if system.graph_model is None)


def run_benchmark(
    layout: BenchmarkLayout,
    *,
    config_path: str | Path,
    split: str,
    systems: list[str] | tuple[str, ...] = DEFAULT_SYSTEMS,
    hgt_artifacts: str | Path | None = None,
    rgcn_artifacts: str | Path | None = None,
    enable_generator: bool = False,
    reindex: bool = False,
    selection_top_k: int = 10,
    per_type_top_k: int | None = None,
    cutoffs: tuple[int, ...] = (1, 3, 5, 10),
    allow_partial: bool = False,
    query_batch_size: int = 64,
) -> dict[str, Any]:
    selected = [SYSTEMS[name] for name in systems]
    artifact_paths = {"hgt": hgt_artifacts, "rgcn": rgcn_artifacts}
    for graph_model in {system.graph_model for system in selected if system.graph_model}:
        if not artifact_paths[graph_model]:
            raise ValueError(f"The {graph_model} system requires --{graph_model}-artifacts")
    sample_path = layout.samples(_official_split(layout.name) if split == "official" else split)
    if not layout.graph.exists() or not sample_path.exists():
        raise FileNotFoundError(f"Prepare {layout.name} before running its benchmark")
    _validate_processed_schema(layout)
    if not allow_partial:
        _validate_preparation(layout)
    for graph_model, artifact_path in artifact_paths.items():
        if graph_model in {system.graph_model for system in selected} and artifact_path:
            _validate_graph_artifacts(
                layout,
                sample_path,
                Path(artifact_path),
                expected_model_type=graph_model,
            )
    if any(system.candidate_backend == "embedding" for system in selected):
        ensure_dense_index(layout, config_path, force=reindex)
        logger.info("Loading exact benchmark embedding store: dataset=%s", layout.name)
        dense_store = ExactEmbeddingStore.from_npz(
            load_graph(layout.graph), layout.processed / "base_embeddings.npz"
        )
        logger.info("Exact benchmark embedding store ready: nodes=%d", len(dense_store.nodes))
    else:
        dense_store = None

    config = load_yaml(config_path)
    retriever_config = build_retriever_config(config)
    samples = load_samples(sample_path)
    benchmark_nodes = (
        dense_store.nodes
        if dense_store is not None
        else tuple(load_graph(layout.graph).nodes.values())
    )
    structured_query_fraction = sum(
        len(sample.query.required_slots) > 1 for sample in samples
    ) / len(samples)
    entity_annotated_node_fraction = sum(
        bool(node.attributes.get("entities")) for node in benchmark_nodes
    ) / max(len(benchmark_nodes), 1)
    if retriever_config.slot_weight > 0 and structured_query_fraction == 0:
        logger.warning("Slot utility is inactive: benchmark queries contain no structured slots")
    if retriever_config.entity_weight > 0 and entity_annotated_node_fraction == 0:
        logger.warning("Entity novelty is inactive: benchmark nodes contain no entity annotations")
    open_domain = layout.name in OPEN_DOMAIN_DATASETS
    logger.info(
        "Benchmark start: dataset=%s split=%s samples=%d systems=%d",
        layout.name,
        split,
        len(samples),
        len(systems),
    )
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
            candidate_store=dense_store if backend == "embedding" else None,
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
                    "Benchmark system: dataset=%s system=%s backend=%s "
                    "retrieval=%s reranker=%s graph_model=%s",
                    layout.name,
                    name,
                    backend,
                    system.retrieval_method,
                    reranker_enabled,
                    system.graph_model,
                )
                pipeline.graph_scorer = (
                    CachedGraphScorer(artifact_paths[system.graph_model])
                    if system.graph_model
                    else None
                )
                metadata = {
                    "dataset": layout.name,
                    "split": split,
                    "system": name,
                    "candidate_backend": backend,
                    "dense_search_backend": "numpy_exact" if backend == "embedding" else None,
                    "retrieval_method": system.retrieval_method,
                    "reranker": reranker_enabled,
                    "graph_model": system.graph_model,
                    "hgt": system.graph_model == "hgt",
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
                    "scope": "corpus" if open_domain else "sample",
                    "structured_query_fraction": structured_query_fraction,
                    "entity_annotated_node_fraction": entity_annotated_node_fraction,
                }
                report = evaluate(
                    pipeline,
                    samples,
                    cutoffs=cutoffs,
                    per_type_top_k=per_type_top_k,
                    scope_to_sample_papers=not open_domain,
                    metadata=metadata,
                    query_vectors=query_vectors if pipeline.embedder else None,
                    query_embedding_ms=query_embedding_ms if pipeline.embedder else 0.0,
                )
                if (
                    system.retrieval_method in {"pcst", "pcst_closure", "ec_bfr"}
                    and report["summary"].get("macro_pcst_fallback", 0.0) > 0
                    and not allow_partial
                ):
                    raise RuntimeError(
                        "pcst_fast is unavailable; refusing to publish fallback results. "
                        "Install the graph dependencies or rerun with --allow-partial "
                        "for smoke tests."
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
    model_type: str = "hgt",
    allow_partial: bool = False,
) -> dict[str, Any]:
    logger.info("Benchmark %s training start: dataset=%s", model_type.upper(), layout.name)
    _validate_processed_schema(layout)
    if not allow_partial:
        _validate_preparation(layout)
    split_statistics = benchmark_split_statistics(layout)
    _validate_training_split(layout)
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
    artifacts = train_graph_index(
        layout.graph,
        layout.processed / "base_embeddings.npz",
        pairs,
        queries,
        output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        relation_weight=relation_weight,
        seed=seed,
        device=device,
        hidden_dimension=int(graph_config.get("hidden_dimension", 256)),
        layers=int(graph_config.get("layers", 2)),
        heads=int(graph_config.get("heads", 4)),
        model_type=model_type,
        rgcn_bases=int(graph_config.get("rgcn_bases", 8)),
    )
    metadata = json.loads((artifacts / "training.json").read_text(encoding="utf-8"))
    metadata["dataset"] = layout.name
    metadata["source_splits"] = {layout.name: "train"}
    metadata["split_statistics"] = split_statistics
    write_json(artifacts / "training.json", metadata)
    logger.info(
        "Benchmark %s training complete: dataset=%s output=%s",
        model_type.upper(),
        layout.name,
        artifacts,
    )
    return {"dataset": layout.name, "artifacts": str(artifacts.resolve()), **metadata}


def benchmark_split_statistics(layout: BenchmarkLayout) -> dict[str, Any]:
    graph = load_graph(layout.graph)
    result: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        path = layout.samples(split)
        if not path.exists():
            continue
        rows = read_jsonl(path)
        paper_ids: set[str] = set()
        node_ids: set[str] = set()
        for row in rows:
            paper_ids.update(str(value) for value in row.get("paper_ids", []))
            if row.get("paper_id") is not None:
                paper_ids.add(str(row["paper_id"]))
            row_nodes = {
                str(value)
                for value in row.get("candidate_node_ids", row["relevant_node_ids"])
                if str(value) in graph.nodes
            }
            node_ids.update(row_nodes)
            paper_ids.update(graph.nodes[node_id].paper_id for node_id in row_nodes)
        if paper_ids:
            node_ids.update(
                node_id for node_id, node in graph.nodes.items() if node.paper_id in paper_ids
            )
        relation_counts = Counter(
            edge.relation.value
            for edge in graph.edges
            if edge.src in node_ids and edge.dst in node_ids
        )
        node_counts = Counter(graph.nodes[node_id].node_type.value for node_id in node_ids)
        result[split] = {
            "questions": len(rows),
            "papers": len(paper_ids),
            "nodes": len(node_ids),
            "node_types": dict(sorted(node_counts.items())),
            "relation_types": dict(sorted(relation_counts.items())),
            "relation_triples": count_relation_triples(graph, paper_ids),
        }
    return result


def _validate_training_split(layout: BenchmarkLayout) -> None:
    train_path = layout.samples("train")
    if not layout.graph.exists() or not train_path.exists():
        raise FileNotFoundError(f"Prepare {layout.name} before training")
    graph = load_graph(layout.graph)
    train_rows = read_jsonl(train_path)
    if not train_rows:
        raise ValueError(f"{layout.name} train split is empty")
    train_queries = {str(row["query_id"]) for row in train_rows}
    train_papers = _sample_papers(graph, train_rows)
    for split in ("dev", "test"):
        path = layout.samples(split)
        if not path.exists():
            continue
        held_out = read_jsonl(path)
        if not held_out:
            raise ValueError(f"{layout.name} {split} split is empty")
        held_out_queries = {str(row["query_id"]) for row in held_out}
        if train_queries & held_out_queries:
            raise ValueError(f"{layout.name} train and {split} query IDs overlap")
        if train_papers & _sample_papers(graph, held_out):
            raise ValueError(f"{layout.name} train and {split} papers overlap")


def ensure_dense_index(
    layout: BenchmarkLayout,
    config_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    graph_digest = hashlib.sha256(layout.graph.read_bytes()).hexdigest()
    marker = layout.processed / "dense_index.json"
    embedding_cache = layout.processed / "base_embeddings.npz"
    config = load_yaml(config_path)
    embedding_digest = embedding_config_digest(config)
    state = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    cache_matches = (
        embedding_cache_is_current(layout.graph, config, embedding_cache)
        and state.get("graph_sha256") == graph_digest
        and state.get("embedding_config_sha256") == embedding_digest
    )
    if cache_matches and not force:
        logger.info("Using cached benchmark embeddings: dataset=%s", layout.name)
        return marker

    logger.info("Building benchmark embedding cache: dataset=%s", layout.name)
    report = index_graph(
        layout.graph,
        config_path,
        embedding_cache,
        force_embeddings=force or not cache_matches,
        upsert_vector_store=False,
    )
    write_json(
        marker,
        {
            "graph_sha256": graph_digest,
            "embedding_config_sha256": embedding_digest,
            "embedding_model": config["embedding"].get("model"),
            "text_nodes": report.text_nodes,
            "figure_nodes": report.figure_nodes,
            "table_nodes": report.table_nodes,
            "dimension": report.dimension,
        },
    )
    return marker


def _official_split(dataset: str) -> str:
    return "test" if dataset in {"mmdocrag", "spiqa"} else "all"


def _validate_preparation(layout: BenchmarkLayout) -> None:
    report_path = layout.processed / "prepare_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    keys = ["missing_images", "missing_evidence"]
    if report.get("evaluation_scope") == "official_all_papers":
        keys.extend(("download_errors", "parse_errors", "missing_papers"))
    problems = {key: report.get(key) for key in keys if report.get(key)}
    if report.get("official_benchmark") is False:
        problems["evaluation_scope"] = report.get("evaluation_scope", "non_official")
    elif layout.name in {"m3docvqa", "multimodalqa"}:
        problems["dataset_provenance"] = "legacy_or_unverified_derived_snapshot"
    if str(report.get("evaluation_scope", "")).startswith("partial"):
        problems["evaluation_scope"] = report["evaluation_scope"]
    if problems:
        raise RuntimeError(
            f"Incomplete {layout.name} preparation: {problems}. "
            "Fix the reported items or pass --allow-partial for a diagnostic run."
        )


def _validate_processed_schema(layout: BenchmarkLayout) -> None:
    report_path = layout.processed / "prepare_report.json"
    if not report_path.exists():
        raise RuntimeError(f"Prepare {layout.name} again: prepare_report.json is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("dataset") != layout.name
        or report.get("schema_version") != PROCESSED_SCHEMA_VERSION
    ):
        raise RuntimeError(
            f"Prepare {layout.name} again: expected processed schema "
            f"{PROCESSED_SCHEMA_VERSION}"
        )


def _sample_papers(graph: EvidenceGraph, rows: list[dict[str, Any]]) -> set[str]:
    papers: set[str] = set()
    for row in rows:
        papers.update(str(value) for value in row.get("paper_ids", []))
        if row.get("paper_id") is not None:
            papers.add(str(row["paper_id"]))
        papers.update(
            graph.nodes[str(node_id)].paper_id
            for node_id in row.get("candidate_node_ids", row.get("relevant_node_ids", []))
            if str(node_id) in graph.nodes
        )
    return papers


def _validate_graph_artifacts(
    layout: BenchmarkLayout,
    samples: Path,
    artifacts: Path,
    *,
    expected_model_type: str,
) -> None:
    metadata_path = artifacts / "training.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing graph artifact metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_type", "hgt") != expected_model_type:
        raise ValueError(
            f"Expected {expected_model_type} artifacts, got {metadata.get('model_type')}"
        )
    graph_digest = hashlib.sha256(layout.graph.read_bytes()).hexdigest()
    if metadata.get("graph_sha256") != graph_digest:
        raise ValueError(f"Graph artifacts do not match the {layout.name} graph")
    train_ids = set(metadata.get("train_query_ids", ()))
    evaluation_ids = {row["query_id"] for row in read_jsonl(samples)}
    if train_ids & evaluation_ids:
        raise ValueError("Training and evaluation queries overlap; use a held-out split")


def _close_pipeline(pipeline: Any) -> None:
    client = getattr(getattr(pipeline, "vector_store", None), "client", None)
    if client and hasattr(client, "close"):
        client.close()
