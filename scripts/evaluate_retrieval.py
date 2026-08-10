from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_rag.bootstrap import build_deployed_pipeline
from paper_rag.domain import QuerySpec


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-grained evidence retrieval")
    parser.add_argument("dataset", help="JSONL with query and relevant_node_ids")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--hgt-artifacts")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--disable-reranker", action="store_true")
    args = parser.parse_args()

    with Path(args.dataset).open("r", encoding="utf-8") as stream:
        samples = [json.loads(line) for line in stream if line.strip()]
    if not samples:
        raise ValueError("Evaluation dataset is empty")
    pipeline = build_deployed_pipeline(
        graph_path=args.graph,
        config_path=args.config,
        hgt_artifact_dir=args.hgt_artifacts,
        enable_reranker=not args.disable_reranker,
        enable_generator=False,
    )

    details: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        gold = {str(value) for value in sample["relevant_node_ids"]}
        if not gold:
            raise ValueError(f"Sample {index} has no relevant_node_ids")
        result = pipeline.run(
            QuerySpec(
                query=str(sample["query"]),
                answer_type=str(sample.get("answer_type", "free_text")),
                entity_type=sample.get("entity_type"),
                metric=sample.get("metric"),
                operator=sample.get("operator"),
                value=sample.get("value"),
                unit=sample.get("unit"),
                conditions=[str(value) for value in sample.get("conditions", [])],
            )
        )
        predicted = result.forest.node_ids
        true_positive = len(predicted & gold)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(gold)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        details.append(
            {
                "query_id": sample.get("query_id", str(index)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cost": result.forest.total_cost,
                "budget": result.forest.budget,
                "budget_valid": result.forest.total_cost <= result.forest.budget,
                "predicted_node_ids": sorted(predicted),
            }
        )

    count = len(details)
    report = {
        "summary": {
            "queries": count,
            "macro_precision": sum(float(row["precision"]) for row in details) / count,
            "macro_recall": sum(float(row["recall"]) for row in details) / count,
            "macro_f1": sum(float(row["f1"]) for row in details) / count,
            "budget_violation_rate": sum(not bool(row["budget_valid"]) for row in details)
            / count,
        },
        "details": details,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved evaluation report: {target.resolve()}")


if __name__ == "__main__":
    main()
