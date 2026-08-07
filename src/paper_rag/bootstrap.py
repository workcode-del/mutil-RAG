from __future__ import annotations

from pathlib import Path

from paper_rag.config import load_yaml
from paper_rag.embedding import HTTPEmbedder
from paper_rag.embedding.qdrant_store import QdrantEvidenceStore
from paper_rag.evidence_graph import load_graph
from paper_rag.generation import OpenAICompatibleGenerator
from paper_rag.models.cached_scorer import CachedHGTScorer
from paper_rag.pipeline import ScientificRAGPipeline
from paper_rag.reranking import HTTPReranker
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
    embed_config = config["embedding"]
    vector_config = config["vector_store"]
    retrieve_config = config["retrieval"]
    embedder = HTTPEmbedder(embed_config["service_url"], dimension=int(embed_config["dimension"]))

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
    forest_retriever = EvidenceClosureBudgetedForestRetriever(
        graph,
        ECBFRConfig(
            budget=int(retrieve_config["budget"]["text_tokens"]),
            image_unit=int(retrieve_config["budget"]["image_unit"]),
            candidate_hops=int(retrieve_config["candidate_hops"]),
            lambda_values=tuple(float(x) for x in retrieve_config["lambda_values"]),
        ),
    )
    graph_scorer = CachedHGTScorer(hgt_artifact_dir) if hgt_artifact_dir else None
    reranker = (
        HTTPReranker(config["reranker"]["service_url"])
        if enable_reranker and config["reranker"].get("enabled", True)
        else None
    )
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
    )
