from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from paper_rag.chart import OpenAICompatibleChartExtractor, SelfEnsemblingChartExtractor
from paper_rag.config import load_yaml
from paper_rag.domain import NodeType
from paper_rag.evidence_graph import (
    EvidenceGraph,
    attach_chart_data,
    build_figure_text_views,
    load_graph,
    save_graph,
)
from paper_rag.io import read_jsonl
from paper_rag.log import configure_logging
from paper_rag.model_source import resolve_model_reference
from paper_rag.parsing import MinerUAdapter, locate_sentence_batch
from paper_rag.training import build_query_pairs, embed_training_queries, train_hgt
from paper_rag.workflow import build_corpus, index_graph


logger = logging.getLogger(__name__)


def _parse_mineru(args: argparse.Namespace) -> int:
    logger.info("Parsing MinerU output: %s", args.input)
    parsed = MinerUAdapter().from_json(args.input, args.paper_id)
    located = {}
    if args.pdf:
        located = locate_sentence_batch(
            args.pdf,
            (
                (node.node_id, node.page, node.text or "")
                for node in parsed.nodes.values()
                if node.node_type is NodeType.SENTENCE and node.page is not None
            ),
        )
        for node_id, location in located.items():
            node = parsed.nodes[node_id]
            node.page = location.page
            node.bbox = location.bbox
            node.provenance["location_level"] = location.level
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
                "sentence_locations": len(located),
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
    report = {"nodes": len(graph.nodes), "edges": len(graph.edges), "by_type": by_type}
    print(json.dumps(report, indent=2))
    return 0


def _list_figures(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph)
    rows = [
        {
            "figure_id": node.node_id,
            "paper_id": node.paper_id,
            "page": node.page,
            "image_path": node.image_path,
        }
        for node in graph.nodes.values()
        if node.node_type is NodeType.FIGURE
    ]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} figure candidates: {target.resolve()}")
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    dimension = config.get("embedding", {}).get("dimension")
    hidden = config.get("graph_index", {}).get("hidden_dimension")
    if dimension != 2048 or hidden != 256:
        raise ValueError(
            "Reference architecture requires base dimension 2048 and graph dimension 256"
        )
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
        if component == "chart" and component_config.get("backend") == "openai_compatible":
            resolved[component] = f"external-api:{component_config.get('model', '')}"
            continue
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


def _enrich_charts(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph)
    entries = read_jsonl(args.manifest)
    logger.info("Chart enrichment: figures=%d graph=%s", len(entries), args.graph)
    config = load_yaml(args.config)
    chart_config = config.get("chart", {})
    extractor = None
    reports: list[dict[str, object]] = []

    for entry in entries:
        figure_id = str(entry["figure_id"])
        if "linearized_table" in entry:
            table = str(entry["linearized_table"])
            status = str(entry.get("parse_status", "provided"))
            confidence = float(entry.get("confidence", 1.0))
            uncertainty = entry.get("uncertainty")
            extractor_name = str(entry.get("extractor", "manifest"))
        else:
            if extractor is None:
                backend = str(chart_config.get("backend", "openai_compatible"))
                if backend != "openai_compatible":
                    raise ValueError(
                        "The unified environment supports chart backend=openai_compatible. "
                        "PP-Chart2Table requires Transformers 5.x and must be exposed as an "
                        "external service or used only in a separate baseline environment."
                    )
                base = OpenAICompatibleChartExtractor(
                    base_url=str(chart_config["base_url"]),
                    model=str(chart_config["model"]),
                    api_key_env=str(chart_config.get("api_key_env", "PAPER_RAG_API_KEY")),
                    timeout=float(chart_config.get("timeout", 120)),
                    temperature=float(chart_config.get("temperature", 0.2)),
                )
                extractor = SelfEnsemblingChartExtractor(
                    base.extract,
                    repeats=int(chart_config.get("self_ensemble_repeats", 3)),
                )
            figure = graph.nodes.get(figure_id)
            if figure is None:
                raise KeyError(f"Unknown figure_id in chart manifest: {figure_id}")
            result = extractor.extract(figure.image_path or "")
            table = result.linearized_table
            status = result.parse_status
            confidence = float(result.confidence or 0.0)
            uncertainty = result.uncertainty
            extractor_name = result.extractor

        chart_node_id = attach_chart_data(
            graph,
            figure_id,
            table,
            status,
            confidence=confidence,
            extractor=extractor_name,
            uncertainty=float(uncertainty) if uncertainty is not None else None,
        )
        reports.append(
            {
                "figure_id": figure_id,
                "chart_node_id": chart_node_id,
                "status": status,
                "confidence": confidence,
            }
        )

    build_figure_text_views(graph)
    save_graph(graph, args.output)
    logger.info("Chart enrichment complete: output=%s", args.output)
    summary = {"charts": reports, "output": str(Path(args.output).resolve())}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_corpus(args: argparse.Namespace) -> int:
    result = build_corpus(
        args.pdf_dir,
        graph_path=args.graph,
        mineru_dir=args.mineru_output,
        embedding_cache=args.embedding_cache,
        config_path=args.config,
        mineru_command=args.mineru_command,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _index_graph(args: argparse.Namespace) -> int:
    report = index_graph(args.graph, args.config, args.embedding_cache)
    print(
        json.dumps(
            {
                "text_nodes": report.text_nodes,
                "figure_nodes": report.figure_nodes,
                "dimension": report.dimension,
            },
            indent=2,
        )
    )
    return 0


def _train_index(args: argparse.Namespace) -> int:
    root = Path(args.work_dir)
    graph_config = load_yaml(args.config).get("graph_index", {})
    pairs = build_query_pairs(
        args.graph,
        args.samples,
        root / "query_pairs.jsonl",
        embeddings_path=args.base_embeddings,
        seed=args.seed,
    )
    queries = embed_training_queries(
        pairs,
        root / "query_embeddings.npz",
        args.config,
        batch_size=args.batch_size,
    )
    artifacts = train_hgt(
        args.graph,
        args.base_embeddings,
        pairs,
        queries,
        args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        relation_weight=args.relation_weight,
        seed=args.seed,
        device=args.device,
        hidden_dimension=int(graph_config.get("hidden_dimension", 256)),
        layers=int(graph_config.get("layers", 2)),
        heads=int(graph_config.get("heads", 4)),
    )
    print(json.dumps({"query_pairs": str(pairs), "artifacts": str(artifacts)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    parse = commands.add_parser("parse-mineru", help="Normalize a MinerU JSON output")
    parse.add_argument("input")
    parse.add_argument("output")
    parse.add_argument("--paper-id")
    parse.add_argument("--pdf", help="Original PDF used to refine sentence-level bbox")
    parse.set_defaults(handler=_parse_mineru)
    inspect = commands.add_parser("inspect-graph", help="Show graph schema statistics")
    inspect.add_argument("graph")
    inspect.set_defaults(handler=_inspect_graph)
    figures = commands.add_parser(
        "list-figures", help="Export figure IDs for manual line-chart filtering"
    )
    figures.add_argument("graph")
    figures.add_argument("output")
    figures.set_defaults(handler=_list_figures)
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
    enrich = commands.add_parser(
        "enrich-charts",
        help="Attach line-chart tables from a manifest or an external multimodal API",
    )
    enrich.add_argument("graph", help="Input evidence graph JSON")
    enrich.add_argument("manifest", help="JSONL with figure_id and optional linearized_table")
    enrich.add_argument("output", help="Output graph containing ChartData nodes")
    enrich.add_argument("--config", default="configs/default.yaml")
    enrich.set_defaults(handler=_enrich_charts)
    build = commands.add_parser(
        "build-corpus", help="Batch parse PDFs, merge the graph, and build the vector index"
    )
    build.add_argument("pdf_dir")
    build.add_argument("--graph", default="data/parsed/evidence_graph.json")
    build.add_argument("--mineru-output", default="data/mineru")
    build.add_argument("--embedding-cache", default="data/cache/base_embeddings.npz")
    build.add_argument("--config", default="configs/default.yaml")
    build.add_argument("--mineru-command", default="mineru")
    build.add_argument("--force", action="store_true")
    build.set_defaults(handler=_build_corpus)
    index = commands.add_parser("index", help="Embed a graph and update the vector store")
    index.add_argument("graph")
    index.add_argument("--embedding-cache", default="data/cache/base_embeddings.npz")
    index.add_argument("--config", default="configs/default.yaml")
    index.set_defaults(handler=_index_graph)
    train = commands.add_parser(
        "train-index", help="Build hard query pairs and train the relation-supervised HGT index"
    )
    train.add_argument("--graph", required=True)
    train.add_argument("--samples", required=True, help="Benchmark train JSONL")
    train.add_argument("--base-embeddings", required=True)
    train.add_argument("--output", default="outputs/srmg_index")
    train.add_argument("--work-dir", default="data/train/srmg")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--relation-weight", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.set_defaults(handler=_train_index)
    from paper_rag.benchmarking.cli import add_benchmark_parser

    add_benchmark_parser(commands)
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
