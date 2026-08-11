from __future__ import annotations

import argparse
import json

from paper_rag.bootstrap import build_deployed_pipeline
from paper_rag.evaluation import evaluate, load_samples, save_report
from paper_rag.retrieval import RETRIEVAL_METHODS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval, evidence, budget, and citations"
    )
    parser.add_argument("dataset", help="JSONL with query and relevant_node_ids")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--candidate-backend", choices=("embedding", "bm25"), default="embedding")
    parser.add_argument("--retrieval-method", choices=RETRIEVAL_METHODS, default="ec_bfr")
    parser.add_argument("--selection-top-k", type=int, default=10)
    parser.add_argument("--per-type-top-k", type=int)
    parser.add_argument("--ranking-k", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--scope", choices=("sample", "corpus"), default="sample")
    parser.add_argument("--hgt-artifacts")
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--enable-generator", action="store_true")
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()

    cutoffs = tuple(sorted(set(args.ranking_k)))
    if not cutoffs or cutoffs[0] < 1:
        parser.error("--ranking-k values must be positive")
    pipeline = build_deployed_pipeline(
        graph_path=args.graph,
        config_path=args.config,
        hgt_artifact_dir=args.hgt_artifacts,
        enable_reranker=not args.disable_reranker,
        enable_generator=args.enable_generator,
        candidate_backend=args.candidate_backend,
        retrieval_method=args.retrieval_method,
        selection_top_k=args.selection_top_k,
    )
    metadata = {
        "candidate_backend": args.candidate_backend,
        "retrieval_method": args.retrieval_method,
        "reranker": not args.disable_reranker,
        "hgt": bool(args.hgt_artifacts),
        "generator": args.enable_generator,
        "selection_top_k": args.selection_top_k,
        "per_type_top_k": args.per_type_top_k,
        "ranking_cutoffs": cutoffs,
        "scope": args.scope,
    }
    report = evaluate(
        pipeline,
        load_samples(args.dataset),
        cutoffs=cutoffs,
        per_type_top_k=args.per_type_top_k,
        scope_to_sample_papers=args.scope == "sample",
        metadata=metadata,
    )
    target = save_report(report, args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved evaluation report: {target}")


if __name__ == "__main__":
    main()
