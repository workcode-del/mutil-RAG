from __future__ import annotations

import os
from pathlib import Path

from paper_rag.config import load_yaml
from paper_rag.domain import RelationType
from paper_rag.embedding import HTTPEmbedder, Qwen3VLEmbedder
from paper_rag.embedding.qdrant_store import QdrantEvidenceStore
from paper_rag.evidence_graph import load_graph
from paper_rag.generation import OpenAICompatibleGenerator
from paper_rag.models.cached_scorer import CachedHGTScorer
from paper_rag.pipeline import ScientificRAGPipeline
from paper_rag.reranking import HTTPReranker, Qwen3VLReranker
from paper_rag.retrieval.ec_bfr import ECBFRConfig, EvidenceClosureBudgetedForestRetriever


def build_deployed_pipeline(
    graph_path: str | Path,
    config_path: str | Path = "configs/default.yaml",
    hgt_artifact_dir: str | Path | None = None,
    enable_reranker: bool = True,
    enable_generator: bool = False,
) -> ScientificRAGPipeline:
    config = load_yaml(config_path)
    graph = load_graph(graph_path)
    runtime_config = config.get("runtime", {})
    embed_config = config["embedding"]
    vector_config = config["vector_store"]
    retrieve_config = config["retrieval"]
    runtime_device = str(runtime_config.get("device", "cuda"))
    official_repo = runtime_config.get("qwen3_vl_retrieval_repo") or os.getenv(
        "QWEN3_VL_RETRIEVAL_REPO"
    )
    embedding_backend = str(embed_config.get("backend", "qwen3_vl")).lower()
    if embedding_backend in {"qwen3_vl", "local"}:
        embedder = Qwen3VLEmbedder(
            model_name=embed_config.get("model", "Qwen/Qwen3-VL-Embedding-2B"),
            official_repo=official_repo,
            dimension=int(embed_config["dimension"]),
            query_instruction=embed_config.get(
                "query_instruction",
                "Retrieve scientific evidence that answers the question.",
            ),
            device=str(embed_config.get("device", runtime_device)),
        )
    elif embedding_backend == "http":
        embedder = HTTPEmbedder(
            embed_config["service_url"], dimension=int(embed_config["dimension"])
        )
    else:
        raise ValueError(f"Unsupported embedding backend: {embedding_backend}")

    mode = vector_config.get("mode", "local")
    if mode == "server":
        store = QdrantEvidenceStore(
            collection=vector_config["collection"],
            dimension=embed_config["dimension"],
            path=None,
            url=vector_config.get("url", "http://127.0.0.1:6333"),
        )
    else:
        store = QdrantEvidenceStore(
            collection=vector_config["collection"],
            dimension=embed_config["dimension"],
            path=vector_config["path"],
        )
    relation_costs = {
        RelationType(name): float(value)
        for name, value in retrieve_config.get("relation_costs", {}).items()
    }
    forest_retriever = EvidenceClosureBudgetedForestRetriever(
        graph,
        ECBFRConfig(
            budget=int(retrieve_config["budget"]["text_tokens"]),
            image_unit=int(retrieve_config["budget"]["image_unit"]),
            candidate_hops=int(retrieve_config["candidate_hops"]),
            lambda_values=tuple(float(x) for x in retrieve_config["lambda_values"]),
            relation_costs=relation_costs or None,
        ),
    )
    graph_scorer = CachedHGTScorer(hgt_artifact_dir) if hgt_artifact_dir else None
    reranker_config = config["reranker"]
    reranker = None
    if enable_reranker and reranker_config.get("enabled", True):
        reranker_backend = str(reranker_config.get("backend", "qwen3_vl")).lower()
        if reranker_backend in {"qwen3_vl", "local"}:
            reranker = Qwen3VLReranker(
                model_name=reranker_config.get("model", "Qwen/Qwen3-VL-Reranker-2B"),
                official_repo=official_repo,
                device=str(reranker_config.get("device", runtime_device)),
            )
        elif reranker_backend == "http":
            reranker = HTTPReranker(reranker_config["service_url"])
        else:
            raise ValueError(f"Unsupported reranker backend: {reranker_backend}")
    generation = config["generation"]
    generator = (
        OpenAICompatibleGenerator(generation["base_url"], generation["model"])
        if enable_generator
        else None
    )
    return ScientificRAGPipeline(
        graph,
        embedder,
        store,
        forest_retriever,
        graph_scorer=graph_scorer,
        reranker=reranker,
        generator=generator,
        default_per_type_top_k=int(vector_config.get("per_type_top_k", 25)),
    )
