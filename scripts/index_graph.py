from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from paper_rag.config import load_yaml
from paper_rag.embedding import HTTPEmbedder, Qwen3VLEmbedder
from paper_rag.embedding.qdrant_store import QdrantEvidenceStore
from paper_rag.evidence_graph import load_graph
from paper_rag.indexing import compute_base_embeddings, upsert_base_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--embedding-cache", default="data/cache/base_embeddings.npz")
    args = parser.parse_args()
    config = load_yaml(args.config)
    graph = load_graph(args.graph)
    embedding = config["embedding"]
    backend = str(embedding.get("backend", "qwen3_vl")).lower()
    if backend in {"qwen3_vl", "local"}:
        runtime = config.get("runtime", {})
        download = config.get("model_download", {})
        embedder = Qwen3VLEmbedder(
            model_name=embedding.get("model", "Qwen/Qwen3-VL-Embedding-2B"),
            official_repo=runtime.get("qwen3_vl_retrieval_repo")
            or os.getenv("QWEN3_VL_RETRIEVAL_REPO"),
            dimension=int(embedding["dimension"]),
            query_instruction=embedding.get(
                "query_instruction",
                "Retrieve scientific evidence that answers the question.",
            ),
            device=str(embedding.get("device", runtime.get("device", "cuda"))),
            model_source=str(
                embedding.get("model_source", download.get("source", "modelscope"))
            ),
            local_path=embedding.get("local_path"),
            modelscope_id=embedding.get("modelscope_id"),
            model_cache_dir=download.get("cache_dir", "data/models"),
        )
    elif backend == "http":
        embedder = HTTPEmbedder(
            embedding["service_url"],
            dimension=int(embedding["dimension"]),
        )
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")
    vector = config["vector_store"]
    store = QdrantEvidenceStore(
        collection=vector["collection"],
        dimension=embedding["dimension"],
        path=vector["path"] if vector.get("mode") == "local" else None,
        url=vector.get("url") if vector.get("mode") == "server" else None,
    )
    embeddings, report = compute_base_embeddings(graph, embedder)
    cache_path = Path(args.embedding_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **embeddings)
    upsert_base_embeddings(store, graph, embeddings)
    print(report, f"cache={cache_path.resolve()}")


if __name__ == "__main__":
    main()
