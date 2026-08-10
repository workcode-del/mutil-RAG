from __future__ import annotations

import argparse

from paper_rag.api import create_app
from paper_rag.bootstrap import build_deployed_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-rag-serve",
        description="Run the complete scientific RAG API from one Python environment.",
    )
    parser.add_argument("--graph", required=True, help="Merged evidence graph JSON")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--hgt-artifacts")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--enable-generator", action="store_true")
    return parser


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the unified dependency group before serving") from exc

    args = build_parser().parse_args()
    pipeline = build_deployed_pipeline(
        graph_path=args.graph,
        config_path=args.config,
        hgt_artifact_dir=args.hgt_artifacts,
        enable_reranker=not args.disable_reranker,
        enable_generator=args.enable_generator,
    )
    uvicorn.run(create_app(pipeline), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
