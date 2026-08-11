from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_rag.evaluation.evidence_mapping import map_evidence
from paper_rag.evidence_graph import load_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Map PeerQA evidence to local graph node IDs")
    parser.add_argument("qa_jsonl", help="PeerQA qa.jsonl")
    parser.add_argument("--graph", required=True, help="Combined local evidence graph")
    parser.add_argument("--output", default="data/evaluation/peerqa.jsonl")
    parser.add_argument("--paper-id-map", help="JSON object: PeerQA paper ID -> local paper ID")
    parser.add_argument("--min-score", type=float, default=0.85)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--report", default="outputs/peerqa_mapping.json")
    args = parser.parse_args()

    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be in [0, 1]")
    graph = load_graph(args.graph)
    paper_id_map = _load_id_map(args.paper_id_map)
    rows = _read_jsonl(args.qa_jsonl)
    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    counts = {"questions": len(rows), "written": 0, "unanswerable": 0, "unmapped": 0}

    for row in rows:
        source_paper_id = str(row["paper_id"])
        paper_id = paper_id_map.get(source_paper_id, source_paper_id)
        evidence = [str(item) for item in row.get("answer_evidence_sent", []) if str(item).strip()]
        answerable = bool(row.get("answerable_mapped", row.get("answerable", False)))
        if not answerable or not evidence:
            counts["unanswerable"] += 1
            continue
        matches = map_evidence(graph, paper_id, evidence, min_score=args.min_score)
        matched_ids = {match.node_id for match in matches if match.node_id}
        complete = len(matched_ids) > 0 and all(match.node_id for match in matches)
        audit.append(
            {
                "question_id": str(row["question_id"]),
                "source_paper_id": source_paper_id,
                "paper_id": paper_id,
                "complete": complete,
                "matches": [match.to_dict() for match in matches],
            }
        )
        if not complete:
            counts["unmapped"] += 1
            if not args.allow_partial or not matched_ids:
                continue
        output.append(
            {
                "query_id": str(row["question_id"]),
                "paper_id": paper_id,
                "query": str(row["question"]),
                "answer": str(row.get("answer_free_form", "")),
                "relevant_node_ids": sorted(matched_ids),
            }
        )

    counts["written"] = len(output)
    _write_jsonl(args.output, output)
    _write_json(args.report, {"counts": counts, "min_score": args.min_score, "audit": audit})
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def _load_id_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--paper-id-map must contain a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: str, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
