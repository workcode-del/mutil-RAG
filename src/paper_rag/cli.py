from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_rag.config import load_yaml
from paper_rag.evidence_graph import EvidenceGraph, load_graph, save_graph
from paper_rag.model_source import resolve_model_reference
from paper_rag.parsing import MinerUAdapter


def _parse_mineru(args: argparse.Namespace) -> int:
    parsed = MinerUAdapter().from_json(args.input, args.paper_id)
    graph = EvidenceGraph()
    graph.extend(parsed.nodes.values(), parsed.edges)
    save_graph(graph, args.output)
    print(
        json.dumps(
            {
                "paper_id": parsed.paper_id,
                "nodes": len(parsed.nodes),
                "edges": len(parsed.edges),
                "warnings": parsed.warnings,
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _inspect_graph(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph)
    by_type: dict[str, int] = {}
    for node in graph.nodes.values():
        by_type[node.node_type.value] = by_type.get(node.node_type.value, 0) + 1
    print(json.dumps({"nodes": len(graph.nodes), "edges": len(graph.edges), "by_type": by_type}, indent=2))
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    dimension = config.get("embedding", {}).get("dimension")
    hidden = config.get("graph_index", {}).get("hidden_dimension")
    if dimension != 2048 or hidden != 256:
        raise ValueError("Reference architecture requires base dimension 2048 and graph dimension 256")
    print(f"Configuration is structurally valid: {Path(args.config).resolve()}")
    return 0


def _merge_graphs(args: argparse.Namespace) -> int:
    merged = EvidenceGraph()
    for path in args.inputs:
        graph = load_graph(path)
        merged.extend(graph.nodes.values(), graph.edges)
    save_graph(merged, args.output)
    print(f"Merged {len(args.inputs)} graphs: nodes={len(merged.nodes)} edges={len(merged.edges)}")
    return 0


def _download_models(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    download = config.get("model_download", {})
    resolved: dict[str, str] = {}
    for component in args.components:
        component_config = config.get(component, {})
        if component == "reranker" and not component_config.get("enabled", True):
            continue
        model_id = component_config.get("model")
        if not model_id:
            raise ValueError(f"No model is configured for component: {component}")
        resolved[component] = resolve_model_reference(
            str(model_id),
            local_path=component_config.get("local_path"),
            modelscope_id=component_config.get("modelscope_id"),
            source=str(
                component_config.get("model_source", download.get("source", "modelscope"))
            ),
            cache_dir=download.get("cache_dir", "data/models"),
        )
    print(json.dumps(resolved, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    parse = commands.add_parser("parse-mineru", help="Normalize a MinerU JSON output")
    parse.add_argument("input")
    parse.add_argument("output")
    parse.add_argument("--paper-id")
    parse.set_defaults(handler=_parse_mineru)
    inspect = commands.add_parser("inspect-graph", help="Show graph schema statistics")
    inspect.add_argument("graph")
    inspect.set_defaults(handler=_inspect_graph)
    validate = commands.add_parser("validate-config")
    validate.add_argument("config", nargs="?", default="configs/default.yaml")
    validate.set_defaults(handler=_validate_config)
    merge = commands.add_parser("merge-graphs", help="Merge per-paper graph JSON files")
    merge.add_argument("output")
    merge.add_argument("inputs", nargs="+")
    merge.set_defaults(handler=_merge_graphs)
    download = commands.add_parser(
        "download-models",
        help="Resolve local model directories or pre-download snapshots from ModelScope",
    )
    download.add_argument("--config", default="configs/default.yaml")
    download.add_argument(
        "--components",
        nargs="+",
        choices=("embedding", "reranker", "chart"),
        default=("embedding", "reranker", "chart"),
    )
    download.set_defaults(handler=_download_models)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
