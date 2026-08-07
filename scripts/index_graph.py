from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paper_rag.config import load_yaml
from paper_rag.embedding import HTTPEmbedder
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
    embedder = HTTPEmbedder(
        config["embedding"]["service_url"],
        dimension=int(config["embedding"]["dimension"]),
    )
    vector = config["vector_store"]
    store = QdrantEvidenceStore(
        collection=vector["collection"],
        dimension=config["embedding"]["dimension"],
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
