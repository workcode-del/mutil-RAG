from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from paper_rag.config import load_yaml
from paper_rag.embedding import HTTPEmbedder, Qwen3VLEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed SRMG training queries from query-pair JSONL"
    )
    parser.add_argument("samples", help="JSONL containing query_id and query")
    parser.add_argument("output", help="Output NPZ keyed by query_id")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with Path(args.samples).open("r", encoding="utf-8") as stream:
        samples = [json.loads(line) for line in stream if line.strip()]
    if not samples:
        raise ValueError("Training query JSONL is empty")
    for sample in samples:
        missing = {"query_id", "query", "positive_node_id", "negative_node_id"} - sample.keys()
        if missing:
            raise ValueError(f"Training sample misses fields: {sorted(missing)}")

    config = load_yaml(args.config)
    embedding = config["embedding"]
    backend = str(embedding.get("backend", "qwen3_vl")).lower()
    if backend in {"qwen3_vl", "local"}:
        runtime = config.get("runtime", {})
        download = config.get("model_download", {})
        embedder = Qwen3VLEmbedder(
            model_name=embedding["model"],
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
            embedding["service_url"], dimension=int(embedding["dimension"])
        )
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")

    output: dict[str, np.ndarray] = {}
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        vectors = embedder.embed_queries([str(sample["query"]) for sample in batch])
        output.update(
            {
                str(sample["query_id"]): vector
                for sample, vector in zip(batch, vectors, strict=True)
            }
        )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **output)
    print(f"Embedded {len(output)} training queries: {target.resolve()}")


if __name__ == "__main__":
    main()
