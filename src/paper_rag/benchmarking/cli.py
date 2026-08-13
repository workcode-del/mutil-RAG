from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_rag.benchmarking.base import BenchmarkLayout
from paper_rag.benchmarking.mmdocrag import prepare_mmdocrag
from paper_rag.benchmarking.peerqa import prepare_peerqa
from paper_rag.benchmarking.runner import (
    DEFAULT_SYSTEMS,
    SYSTEMS,
    run_benchmark,
    train_benchmark_index,
)


DATASETS = ("peerqa", "mmdocrag")


def add_benchmark_parser(commands: argparse._SubParsersAction) -> None:
    benchmark = commands.add_parser("benchmark", help="Prepare and evaluate public benchmarks")
    actions = benchmark.add_subparsers(dest="benchmark_action", required=True)

    prepare = actions.add_parser("prepare", help="Download and convert datasets in batch")
    _add_dataset_options(prepare)
    _add_prepare_options(prepare)
    prepare.set_defaults(handler=_handle_prepare)

    run = actions.add_parser("run", help="Run a controlled system comparison")
    _add_dataset_options(run)
    _add_run_options(run)
    run.set_defaults(handler=_handle_run)

    train = actions.add_parser("train", help="Train the graph index for prepared datasets")
    _add_dataset_options(train)
    _add_train_options(train)
    train.set_defaults(handler=_handle_train)

    all_in_one = actions.add_parser("all", help="Prepare, index, evaluate, and compare")
    _add_dataset_options(all_in_one)
    _add_prepare_options(all_in_one)
    _add_run_options(all_in_one)
    _add_train_options(all_in_one, optional=True)
    all_in_one.set_defaults(handler=_handle_all)


def _add_dataset_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--root", default="data/benchmarks")


def _add_prepare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--setting", type=int, choices=(15, 20), default=20)
    parser.add_argument(
        "--peerqa-download-pdfs",
        action="store_true",
        help="Extend PeerQA with OpenReview PDFs; the official subset needs no PDFs",
    )
    parser.add_argument(
        "--mmdocrag-download-pdfs",
        "--download-pdfs",
        dest="mmdocrag_download_pdfs",
        action="store_true",
        help="Download MMDocRAG PDFs for later full-document experiments",
    )
    parser.add_argument("--skip-mineru", action="store_true")
    parser.add_argument("--mineru-command", default="mineru")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="official")
    parser.add_argument("--systems", nargs="+", choices=tuple(SYSTEMS), default=DEFAULT_SYSTEMS)
    parser.add_argument("--hgt-artifacts")
    parser.add_argument("--enable-generator", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--selection-top-k", type=int, default=10)
    parser.add_argument("--per-type-top-k", type=int)
    parser.add_argument("--ranking-k", nargs="+", type=int, default=(1, 3, 5, 10))
    parser.add_argument("--allow-partial", action="store_true")


def _add_train_options(parser: argparse.ArgumentParser, *, optional: bool = False) -> None:
    if optional:
        parser.add_argument("--train-hgt", action="store_true")
    else:
        parser.add_argument("--config", default="configs/default.yaml")
        parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--hgt-output-root", default="outputs/benchmark_hgt")
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--train-learning-rate", type=float, default=1e-3)
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")


def _handle_prepare(args: argparse.Namespace) -> int:
    reports = _prepare(args)
    print(json.dumps(_report_summaries(reports), ensure_ascii=False, indent=2))
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    reports = _run(args)
    for report in reports.values():
        print(report["table"])
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    print(json.dumps(_train(args), ensure_ascii=False, indent=2))
    return 0


def _handle_all(args: argparse.Namespace) -> int:
    prepared = _prepare(args)
    print(json.dumps(_report_summaries(prepared), ensure_ascii=False, indent=2))
    if args.train_hgt:
        print(json.dumps(_train(args), ensure_ascii=False, indent=2))
        args.systems = list(dict.fromkeys([*args.systems, "full"]))
        args.hgt_artifacts = args.hgt_output_root
        if args.split == "official":
            args.split = "test"
    reports = _run(args)
    for report in reports.values():
        print(report["table"])
    return 0


def _prepare(args: argparse.Namespace) -> dict[str, dict]:
    reports = {}
    for dataset in args.datasets:
        layout = BenchmarkLayout.create(dataset, args.root)
        if dataset == "peerqa":
            reports[dataset] = prepare_peerqa(
                layout,
                force=args.force,
                download_pdfs=args.peerqa_download_pdfs,
                run_mineru=not args.skip_mineru,
                workers=args.workers,
                mineru_command=args.mineru_command,
            )
        else:
            reports[dataset] = prepare_mmdocrag(
                layout,
                setting=args.setting,
                force=args.force,
                download_pdfs=args.mmdocrag_download_pdfs,
            )
    return reports


def _run(args: argparse.Namespace) -> dict[str, dict]:
    cutoffs = tuple(sorted(set(args.ranking_k)))
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError("--ranking-k values must be positive")
    return {
        dataset: run_benchmark(
            BenchmarkLayout.create(dataset, args.root),
            config_path=args.config,
            split=args.split,
            systems=args.systems,
            hgt_artifacts=_dataset_artifacts(args.hgt_artifacts, dataset),
            enable_generator=args.enable_generator,
            reindex=args.reindex,
            selection_top_k=args.selection_top_k,
            per_type_top_k=args.per_type_top_k,
            cutoffs=cutoffs,
            allow_partial=args.allow_partial,
        )
        for dataset in args.datasets
    }


def _train(args: argparse.Namespace) -> dict[str, dict]:
    return {
        dataset: train_benchmark_index(
            BenchmarkLayout.create(dataset, args.root),
            config_path=args.config,
            output=_dataset_artifacts(args.hgt_output_root, dataset),
            epochs=args.train_epochs,
            batch_size=args.train_batch_size,
            learning_rate=args.train_learning_rate,
            relation_weight=args.relation_weight,
            seed=args.seed,
            device=args.device,
            reindex=args.reindex,
        )
        for dataset in args.datasets
    }


def _dataset_artifacts(root: str | None, dataset: str) -> str | None:
    if not root:
        return None
    path = Path(root)
    direct = (path / "query_projector.pt").exists() or path.name == dataset
    return str(path if direct else path / dataset)


def _report_summaries(reports: dict[str, dict]) -> dict[str, dict]:
    details = {"missing_papers", "missing_images", "download_errors", "parse_errors"}
    return {
        dataset: {
            **{key: value for key, value in report.items() if key not in details},
            **{f"{key}_count": len(report.get(key, ())) for key in details},
        }
        for dataset, report in reports.items()
    }
