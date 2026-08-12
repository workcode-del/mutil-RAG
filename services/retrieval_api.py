from __future__ import annotations

import os

from paper_rag.api import create_app
from paper_rag.bootstrap import build_deployed_pipeline
from paper_rag.log import configure_logging


configure_logging()

graph_path = os.getenv("PAPER_RAG_GRAPH")
pipeline = None
if graph_path:
    pipeline = build_deployed_pipeline(
        graph_path=graph_path,
        config_path=os.getenv("PAPER_RAG_CONFIG", "configs/default.yaml"),
        hgt_artifact_dir=os.getenv("PAPER_RAG_HGT_ARTIFACTS"),
        # Default config loads Embedding and Reranker in this process and environment.
        # configs/server.yaml keeps the optional HTTP-isolated deployment mode.
        enable_reranker=os.getenv("PAPER_RAG_ENABLE_RERANKER", "1") == "1",
        enable_generator=os.getenv("PAPER_RAG_ENABLE_GENERATOR", "0") == "1",
    )

app = create_app(pipeline)
