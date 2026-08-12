from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from paper_rag.config import load_yaml
from paper_rag.domain import RelationType
from paper_rag.embedding import BM25EvidenceStore, HTTPEmbedder, Qwen3VLEmbedder
from paper_rag.embedding.qdrant_store import QdrantEvidenceStore
from paper_rag.evidence_graph import load_graph
from paper_rag.generation import OpenAICompatibleGenerator
from paper_rag.models.cached_scorer import CachedHGTScorer
from paper_rag.pipeline import ScientificRAGPipeline
from paper_rag.reranking import HTTPReranker, Qwen3VLReranker
from paper_rag.retrieval import build_evidence_retriever
from paper_rag.retrieval.ec_bfr import ECBFRConfig


logger = logging.getLogger(__name__)


def build_deployed_pipeline(
    graph_path: str | Path,
    config_path: str | Path = "configs/default.yaml",
    hgt_artifact_dir: str | Path | None = None,
    enable_reranker: bool = True,
    enable_generator: bool = False,
    candidate_backend: str = "embedding",
    retrieval_method: str = "ec_bfr",
    selection_top_k: int = 10,
) -> ScientificRAGPipeline:
    logger.info(
        "Loading pipeline: graph=%s candidate=%s retrieval=%s hgt=%s reranker=%s generator=%s",
        graph_path,
        candidate_backend,
        retrieval_method,
        bool(hgt_artifact_dir),
        enable_reranker,
        enable_generator,
    )
    config = load_yaml(config_path)
    graph = load_graph(graph_path)
    normalized_candidate_backend = candidate_backend.strip().lower()
    if normalized_candidate_backend == "bm25":
        if hgt_artifact_dir:
            raise ValueError("HGT scoring requires candidate_backend=embedding")
        embedder = None
        store = BM25EvidenceStore(graph)
    elif normalized_candidate_backend == "embedding":
        embedder = build_embedder(config)
        store = build_vector_store(config)
    else:
        raise ValueError("candidate_backend must be one of: embedding, bm25")
    retriever_config = build_retriever_config(config)
    forest_retriever = build_evidence_retriever(
        retrieval_method,
        graph,
        retriever_config,
        selection_top_k=selection_top_k,
    )
    graph_scorer = CachedHGTScorer(hgt_artifact_dir) if hgt_artifact_dir else None
    reranker = build_reranker(config) if enable_reranker else None
    generator = build_generator(config) if enable_generator else None
    vector_config = config["vector_store"]
    pipeline = ScientificRAGPipeline(
        graph,
        embedder,
        store,
        forest_retriever,
        graph_scorer=graph_scorer,
        reranker=reranker,
        generator=generator,
        default_per_type_top_k=int(vector_config.get("per_type_top_k", 25)),
    )
    logger.info("Pipeline ready: nodes=%d edges=%d", len(graph.nodes), len(graph.edges))
    return pipeline


def build_embedder(config: dict[str, Any]):
    embedding = config["embedding"]
    backend = str(embedding.get("backend", "qwen3_vl")).lower()
    logger.info("Loading embedding module: backend=%s model=%s", backend, embedding.get("model"))
    if backend == "http":
        return HTTPEmbedder(embedding["service_url"], int(embedding["dimension"]))
    if backend not in {"qwen3_vl", "local"}:
        raise ValueError(f"Unsupported embedding backend: {backend}")
    runtime = config.get("runtime", {})
    download = config.get("model_download", {})
    return Qwen3VLEmbedder(
        model_name=embedding.get("model", "Qwen/Qwen3-VL-Embedding-2B"),
        official_repo=runtime.get("qwen3_vl_retrieval_repo")
        or os.getenv("QWEN3_VL_RETRIEVAL_REPO"),
        dimension=int(embedding["dimension"]),
        query_instruction=embedding.get(
            "query_instruction", "Retrieve scientific evidence that answers the question."
        ),
        device=str(embedding.get("device", runtime.get("device", "cuda"))),
        model_source=str(embedding.get("model_source", download.get("source", "modelscope"))),
        local_path=embedding.get("local_path"),
        modelscope_id=embedding.get("modelscope_id"),
        model_cache_dir=download.get("cache_dir", "data/models"),
    )


def build_vector_store(config: dict[str, Any]) -> QdrantEvidenceStore:
    vector = config["vector_store"]
    embedding = config["embedding"]
    server = vector.get("mode", "local") == "server"
    logger.info(
        "Opening vector store: mode=%s collection=%s",
        vector.get("mode", "local"),
        vector["collection"],
    )
    return QdrantEvidenceStore(
        collection=vector["collection"],
        dimension=int(embedding["dimension"]),
        path=None if server else vector["path"],
        url=vector.get("url", "http://127.0.0.1:6333") if server else None,
    )


def build_reranker(config: dict[str, Any]):
    reranker = config["reranker"]
    if not reranker.get("enabled", True):
        return None
    backend = str(reranker.get("backend", "qwen3_vl")).lower()
    logger.info("Loading reranker module: backend=%s model=%s", backend, reranker.get("model"))
    if backend == "http":
        return HTTPReranker(reranker["service_url"])
    if backend not in {"qwen3_vl", "local"}:
        raise ValueError(f"Unsupported reranker backend: {backend}")
    runtime = config.get("runtime", {})
    download = config.get("model_download", {})
    return Qwen3VLReranker(
        model_name=reranker.get("model", "Qwen/Qwen3-VL-Reranker-2B"),
        official_repo=runtime.get("qwen3_vl_retrieval_repo")
        or os.getenv("QWEN3_VL_RETRIEVAL_REPO"),
        device=str(reranker.get("device", runtime.get("device", "cuda"))),
        model_source=str(reranker.get("model_source", download.get("source", "modelscope"))),
        local_path=reranker.get("local_path"),
        modelscope_id=reranker.get("modelscope_id"),
        model_cache_dir=download.get("cache_dir", "data/models"),
    )


def build_generator(config: dict[str, Any]) -> OpenAICompatibleGenerator:
    generation = config["generation"]
    logger.info(
        "Loading generation module: model=%s base_url=%s",
        generation["model"],
        generation["base_url"],
    )
    return OpenAICompatibleGenerator(
        generation["base_url"],
        generation["model"],
        api_key_env=str(generation.get("api_key_env", "PAPER_RAG_API_KEY")),
        timeout=float(generation.get("timeout", 120)),
    )


def build_retriever_config(config: dict) -> ECBFRConfig:
    retrieve_config = config["retrieval"]
    relation_costs = {
        RelationType(name): float(value)
        for name, value in retrieve_config.get("relation_costs", {}).items()
    }
    return ECBFRConfig(
        budget=int(retrieve_config["budget"]["text_tokens"]),
        image_unit=int(retrieve_config["budget"]["image_unit"]),
        candidate_hops=int(retrieve_config["candidate_hops"]),
        lambda_values=tuple(float(x) for x in retrieve_config["lambda_values"]),
        relation_costs=relation_costs or None,
    )
