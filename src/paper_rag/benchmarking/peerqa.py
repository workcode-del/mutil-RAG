from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from paper_rag.benchmarking.base import (
    BenchmarkLayout,
    grouped_split,
    read_jsonl,
    safe_name,
    write_json,
    write_jsonl,
)
from paper_rag.benchmarking.download import download_file, extract_zip
from paper_rag.domain import EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.evaluation.evidence_mapping import map_evidence
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.parsing import MinerUAdapter


PEERQA_ARCHIVE = (
    "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4467/"
    "peerqa-data-v1.0.zip?sequence=5&isAllowed=y"
)


def prepare_peerqa(
    layout: BenchmarkLayout,
    *,
    force: bool = False,
    download_pdfs: bool = True,
    run_mineru: bool = True,
    workers: int = 4,
    mineru_command: str = "mineru",
) -> dict[str, Any]:
    archive = download_file(PEERQA_ARCHIVE, layout.raw / "peerqa-data-v1.0.zip", force=force)
    data_root = extract_zip(archive, layout.raw / "dataset", force=force)
    qa_path = _find_one(data_root, "qa.jsonl")
    papers_path = _find_one(data_root, "papers.jsonl")
    qa_rows = read_jsonl(qa_path)
    graph = _build_official_graph(read_jsonl(papers_path))
    available_papers = {node.paper_id for node in graph.nodes.values()}
    required_papers = {str(row["paper_id"]) for row in qa_rows}
    missing_papers = sorted(required_papers - available_papers)
    pdf_root = layout.raw / "pdfs"
    pdf_manifest = {
        paper_id: pdf_root / f"{safe_name(paper_id)}.pdf"
        for paper_id in missing_papers
        if (pdf_root / f"{safe_name(paper_id)}.pdf").exists()
    }
    download_errors: dict[str, str] = {}

    if download_pdfs:
        downloaded, download_errors = _download_openreview_pdfs(
            missing_papers,
            pdf_root,
            force=force,
            workers=workers,
        )
        pdf_manifest.update(downloaded)
    parse_errors: dict[str, str] = {}
    if run_mineru and pdf_manifest:
        parsed, parse_errors = _parse_pdfs(
            pdf_manifest,
            layout.raw / "mineru",
            command=mineru_command,
            force=force,
        )
        graph.extend(parsed.nodes.values(), parsed.edges)

    save_graph(graph, layout.graph)
    samples, conversion = _convert_questions(graph, qa_rows)
    write_jsonl(layout.samples("all"), samples)
    for split, rows in grouped_split(samples, group_key="paper_id").items():
        write_jsonl(layout.samples(split), rows)
    report = {
        "dataset": "peerqa",
        "graph_mode": "official_sentences_plus_mineru_openreview",
        "questions": len(qa_rows),
        "evaluation_samples": len(samples),
        "nodes": len(graph.nodes),
        "papers": len({node.paper_id for node in graph.nodes.values()}),
        "missing_papers": sorted(
            required_papers - {node.paper_id for node in graph.nodes.values()}
        ),
        "downloaded_pdfs": len(pdf_manifest),
        "download_errors": download_errors,
        "parse_errors": parse_errors,
        "conversion": conversion,
    }
    write_json(layout.processed / "prepare_report.json", report)
    return report


def _build_official_graph(rows: list[dict[str, Any]]) -> EvidenceGraph:
    graph = EvidenceGraph()
    by_paper: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        paper_id = str(row["paper_id"])
        index = int(row["idx"])
        node_id = _official_node_id(paper_id, index)
        kind = str(row.get("type", "sentence")).lower()
        node_type = NodeType.CAPTION if "caption" in kind else NodeType.SENTENCE
        graph.add_node(
            EvidenceNode(
                node_id,
                paper_id,
                node_type,
                text=str(row.get("content", "")),
                attributes={
                    "peerqa_idx": index,
                    "paragraph_index": row.get("pidx"),
                    "sentence_index": row.get("sidx"),
                    "last_heading": row.get("last_heading"),
                },
            )
        )
        by_paper.setdefault(paper_id, []).append((index, node_id))
    for paper_nodes in by_paper.values():
        ordered = [node_id for _, node_id in sorted(paper_nodes)]
        for source, target in zip(ordered, ordered[1:]):
            graph.add_edge(EvidenceEdge(source, target, RelationType.NEXT_SENTENCE))
    return graph


def _download_openreview_pdfs(
    paper_ids: list[str],
    root: Path,
    *,
    force: bool,
    workers: int,
) -> tuple[dict[str, Path], dict[str, str]]:
    jobs: dict[Any, tuple[str, Path]] = {}
    result: dict[str, Path] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for paper_id in paper_ids:
            parts = paper_id.split("/")
            if len(parts) < 3 or parts[0] != "openreview":
                continue
            target = root / f"{safe_name(paper_id)}.pdf"
            future = pool.submit(
                download_file,
                f"https://openreview.net/pdf?id={parts[-1]}",
                target,
                force=force,
            )
            jobs[future] = (paper_id, target)
        for future in as_completed(jobs):
            paper_id, target = jobs[future]
            try:
                future.result()
                result[paper_id] = target
            except Exception as exc:  # continue other downloads and report this paper
                errors[paper_id] = str(exc)
    return result, errors


def _parse_pdfs(
    pdfs: dict[str, Path],
    output: Path,
    *,
    command: str,
    force: bool,
) -> tuple[EvidenceGraph, dict[str, str]]:
    graph = EvidenceGraph()
    errors: dict[str, str] = {}
    if shutil.which(command) is None and any(
        force or _find_content_list(output, pdf.stem) is None for pdf in pdfs.values()
    ):
        raise RuntimeError(
            f"MinerU command not found: {command}. Install MinerU or pass --skip-mineru."
        )
    for paper_id, pdf in pdfs.items():
        content = _find_content_list(output, pdf.stem)
        try:
            if force or content is None:
                print(f"MinerU: {paper_id}")
                subprocess.run(
                    [command, "-p", str(pdf), "-o", str(output), "-b", "pipeline"],
                    check=True,
                )
                content = _find_content_list(output, pdf.stem)
            if content is None:
                raise FileNotFoundError(f"MinerU produced no content list for {pdf}")
            parsed = MinerUAdapter().from_json(content, paper_id)
            graph.extend(parsed.nodes.values(), parsed.edges)
        except Exception as exc:  # keep the batch resumable and report per-paper failures
            errors[paper_id] = str(exc)
    return graph, errors


def _convert_questions(
    graph: EvidenceGraph, qa_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    counts = {"unanswerable": 0, "unmapped": 0}
    for row in qa_rows:
        if not bool(row.get("answerable_mapped", row.get("answerable", False))):
            counts["unanswerable"] += 1
            continue
        paper_id = str(row["paper_id"])
        gold: set[str] = set()
        mapped_groups = row.get("answer_evidence_mapped") or []
        complete = bool(mapped_groups)
        for mapped in mapped_groups:
            group_ids: set[str] = set()
            for index in mapped.get("idx") or []:
                node_id = _official_node_id(paper_id, int(index)) if index is not None else ""
                if node_id in graph.nodes:
                    group_ids.add(node_id)
            complete = complete and bool(group_ids)
            gold.update(group_ids)
        if not complete:
            evidence = [str(value) for value in row.get("answer_evidence_sent", [])]
            matches = map_evidence(graph, paper_id, evidence)
            complete = bool(matches) and all(match.node_id for match in matches)
            gold = {match.node_id for match in matches if match.node_id} if complete else set()
        if not gold:
            counts["unmapped"] += 1
            continue
        samples.append(
            {
                "query_id": str(row["question_id"]),
                "paper_id": paper_id,
                "query": str(row["question"]),
                "answer": str(row.get("answer_free_form", "")),
                "relevant_node_ids": sorted(gold),
            }
        )
    return samples, counts


def _official_node_id(paper_id: str, index: int) -> str:
    return f"peerqa::{paper_id}::{index}"


def _find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def _find_content_list(root: Path, stem: str) -> Path | None:
    matches = list(root.rglob(f"{stem}_content_list.json"))
    return matches[0] if matches else None
