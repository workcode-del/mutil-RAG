from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from paper_rag.bootstrap import build_embedder
from paper_rag.config import load_yaml
from paper_rag.domain import EvidenceEdge
from paper_rag.evidence_graph import EvidenceGraph, load_graph
from paper_rag.io import iter_jsonl, read_jsonl, write_jsonl
from paper_rag.models import (
    HGTConfig,
    RGCNConfig,
    build_heterotopology,
    create_hgt_model,
    create_rgcn_model,
)
from paper_rag.models.losses import query_evidence_margin_loss, relation_info_nce


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _EmbeddingTable:
    vectors: np.ndarray
    positions: dict[str, int]

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.positions

    def __getitem__(self, node_id: str) -> np.ndarray:
        return self.vectors[self.positions[node_id]]

    def close(self) -> None:
        memory_map = getattr(self.vectors, "_mmap", None)
        if memory_map is not None:
            memory_map.close()


@dataclass(slots=True)
class _TrainingBatch:
    samples: list[dict]
    node_indices: dict[str, Any]
    edge_indices: dict[tuple[str, str, str], Any]
    positions: dict[str, tuple[str, int]]
    relations: list[tuple[str, str, str]]
    query_vectors: Any


def build_query_pairs(
    graph_path: str | Path,
    samples_path: str | Path,
    output: str | Path,
    *,
    embeddings_path: str | Path | None = None,
    seed: int = 42,
) -> Path:
    graph = load_graph(graph_path)
    rng = random.Random(seed)
    embeddings = (
        _load_embedding_table(embeddings_path, list(graph.nodes), require_all=False)
        if embeddings_path
        else None
    )
    pair_count = 0

    def generate_pairs():
        nonlocal pair_count
        for sample in iter_jsonl(samples_path):
            positives = [
                node_id for node_id in sample["relevant_node_ids"] if node_id in graph.nodes
            ]
            paper_ids = {str(value) for value in sample.get("paper_ids", [])}
            if sample.get("paper_id") is not None:
                paper_ids.add(str(sample["paper_id"]))
            if not paper_ids:
                paper_ids.update(graph.nodes[node_id].paper_id for node_id in positives)
            raw_candidates = sample.get("candidate_node_ids") or [
                node_id
                for node_id, node in graph.nodes.items()
                if node.paper_id in paper_ids
            ]
            candidates = [node_id for node_id in raw_candidates if node_id in graph.nodes]
            negatives = [node_id for node_id in candidates if node_id not in positives]
            if not positives or not negatives:
                continue
            for positive in positives:
                same_type = [
                    node_id
                    for node_id in negatives
                    if graph.nodes[node_id].node_type is graph.nodes[positive].node_type
                ]
                pool = same_type or negatives
                negative = (
                    max(pool, key=lambda node_id: _similarity(embeddings, positive, node_id))
                    if embeddings is not None and positive in embeddings
                    else rng.choice(pool)
                )
                pair_count += 1
                yield {
                    "query_id": str(sample["query_id"]),
                    "query": str(sample["query"]),
                    "positive_node_id": positive,
                    "negative_node_id": negative,
                }

    try:
        target = write_jsonl(output, generate_pairs())
    finally:
        if embeddings is not None:
            embeddings.close()
    logger.info("Training pairs ready: pairs=%d output=%s", pair_count, target)
    return target


def embed_training_queries(
    samples_path: str | Path,
    output: str | Path,
    config_path: str | Path,
    *,
    batch_size: int = 64,
) -> Path:
    embedder = build_embedder(load_yaml(config_path))
    vectors: dict[str, np.ndarray] = {}
    queries = {
        str(sample["query_id"]): str(sample["query"])
        for sample in iter_jsonl(samples_path)
    }
    logger.info("Embedding training queries: unique_queries=%d", len(queries))
    items = list(queries.items())
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        encoded = embedder.embed_queries([query for _, query in batch])
        vectors.update(
            {query_id: vector for (query_id, _), vector in zip(batch, encoded, strict=True)}
        )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **vectors)
    logger.info("Query embeddings ready: output=%s", target)
    return target


def train_hgt(
    graph_path: str | Path,
    base_embeddings_path: str | Path,
    query_pairs_path: str | Path,
    query_embeddings_path: str | Path,
    output: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    relation_weight: float = 0.2,
    seed: int = 42,
    device: str = "cuda",
    hidden_dimension: int = 256,
    layers: int = 2,
    heads: int = 4,
    model_type: str = "hgt",
    rgcn_bases: int = 8,
    precision: str = "auto",
) -> Path:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    graph = load_graph(graph_path)
    _validate_paper_local_edges(graph)
    paper_index = _paper_graph_index(graph)
    samples = read_jsonl(query_pairs_path)
    if not samples:
        raise ValueError("No trainable query pairs were produced")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    setup_started = perf_counter()
    base_embeddings = _load_embedding_table(base_embeddings_path, list(graph.nodes))
    query_embeddings = _load_npz(query_embeddings_path)
    input_dimension = int(base_embeddings.vectors.shape[1])
    device = torch.device(device)
    resolved_precision, amp_dtype = _resolve_precision(torch, device, precision)
    input_dtype = amp_dtype or torch.float32
    base_tensor = torch.from_numpy(base_embeddings.vectors).to(
        device=device, dtype=input_dtype
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    normalized_model_type = model_type.strip().lower()
    if normalized_model_type not in {"hgt", "rgcn"}:
        raise ValueError("model_type must be hgt or rgcn")
    metadata_schema = _graph_metadata(graph)
    model = (
        create_hgt_model(
            metadata_schema,
            HGTConfig(input_dimension, hidden_dimension, layers, heads),
        )
        if normalized_model_type == "hgt"
        else create_rgcn_model(
            metadata_schema,
            RGCNConfig(input_dimension, hidden_dimension, layers, rgcn_bases),
        )
    ).to(device)
    logger.info(
        "Preparing paper-aware batched %s model: nodes=%d device=%s precision=%s",
        normalized_model_type.upper(),
        len(graph.nodes),
        device,
        resolved_precision,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=resolved_precision == "fp16")
    train_papers = {graph.nodes[row["positive_node_id"]].paper_id for row in samples}
    relations = _relation_triples(graph, train_papers, seed)
    batches = _prepare_training_batches(
        graph,
        samples,
        query_embeddings,
        base_embeddings.positions,
        paper_index,
        batch_size,
        seed,
        device,
        input_dtype,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_seconds = perf_counter() - setup_started
    logger.info(
        "%s training: nodes=%d query_pairs=%d batches=%d relation_triples=%d "
        "epochs=%d device=%s setup_seconds=%.1f",
        normalized_model_type.upper(),
        len(graph.nodes),
        len(samples),
        len(batches),
        len(relations),
        epochs,
        device,
        setup_seconds,
    )

    epoch_seconds = []
    for epoch in range(epochs):
        epoch_started = perf_counter()
        model.train()
        order = list(range(len(batches)))
        random.Random(seed + epoch).shuffle(order)
        epoch_loss = torch.zeros((), device=device)
        for index in order:
            batch = batches[index]
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, device, amp_dtype):
                features = {
                    node_type: base_tensor.index_select(0, indices)
                    for node_type, indices in batch.node_indices.items()
                }
                hidden = model.encode_graph(features, batch.edge_indices)
                query_hidden = model.encode_query(batch.query_vectors)
            positives = _sample_nodes(
                hidden, batch.positions, batch.samples, "positive_node_id"
            ).float()
            negatives = _sample_nodes(
                hidden, batch.positions, batch.samples, "negative_node_id"
            ).float()
            loss = query_evidence_margin_loss(query_hidden.float(), positives, negatives)
            if relation_weight > 0 and batch.relations:
                anchors = _triple_nodes(hidden, batch.positions, batch.relations, 0).float()
                related = _triple_nodes(hidden, batch.positions, batch.relations, 1).float()
                unrelated = _triple_nodes(hidden, batch.positions, batch.relations, 2).float()
                loss += relation_weight * relation_info_nce(
                    anchors, related, unrelated[:, None, :]
                )
            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.detach() * len(batch.samples)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - epoch_started
        epoch_seconds.append(elapsed)
        logger.info(
            "%s epoch %d/%d loss=%.6f seconds=%.1f",
            normalized_model_type.upper(),
            epoch + 1,
            epochs,
            float(epoch_loss.cpu()) / len(samples),
            elapsed,
        )

    metadata = {
        "graph_sha256": hashlib.sha256(Path(graph_path).read_bytes()).hexdigest(),
        "model_type": normalized_model_type,
        "query_pairs": len(samples),
        "train_query_ids": sorted({row["query_id"] for row in samples}),
        "relation_triples": len(relations),
        "input_dimension": input_dimension,
        "hidden_dimension": hidden_dimension,
        "layers": layers,
        "heads": heads if normalized_model_type == "hgt" else None,
        "rgcn_bases": rgcn_bases if normalized_model_type == "rgcn" else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "batching": "paper_aware_query_pairs",
        "precision": resolved_precision,
        "loss_dtype": "float32",
        "optimizer_dtype": "float32",
        "setup_seconds": setup_seconds,
        "epoch_seconds": epoch_seconds,
        "relation_weight": relation_weight,
        "seed": seed,
    }
    export_started = perf_counter()
    artifacts = _export_hgt(
        model,
        graph,
        base_tensor,
        base_embeddings.positions,
        output,
        metadata,
        paper_index=paper_index,
        paper_batch_size=batch_size,
        amp_dtype=amp_dtype,
    )
    metadata["export_seconds"] = perf_counter() - export_started
    metadata["peak_gpu_memory_gib"] = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    )
    (artifacts / "training.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base_embeddings.close()
    logger.info("%s artifacts ready: %s", normalized_model_type.upper(), artifacts)
    return artifacts


train_graph_index = train_hgt


def _load_embedding_table(
    path: str | Path, node_ids: list[str], *, require_all: bool = True
) -> _EmbeddingTable:
    source = Path(path)
    with np.load(source) as archive:
        available = [node_id for node_id in node_ids if node_id in archive]
    if require_all and len(available) != len(node_ids):
        available_ids = set(available)
        missing = next(node_id for node_id in node_ids if node_id not in available_ids)
        raise KeyError(f"Missing base embeddings; first={missing}")
    node_ids = available
    if not node_ids:
        return _EmbeddingTable(np.empty((0, 0), dtype=np.float32), {})
    matrix_path = source.with_suffix(".matrix.npy")
    metadata_path = source.with_suffix(".matrix.json")
    stat = source.stat()
    expected = {
        "version": 1,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "node_ids_sha256": hashlib.sha256("\0".join(node_ids).encode()).hexdigest(),
    }
    metadata = {}
    if matrix_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if any(metadata.get(key) != value for key, value in expected.items()):
        _write_embedding_matrix(source, matrix_path, metadata_path, node_ids, expected)
    vectors = np.load(matrix_path, mmap_mode="c")
    if vectors.ndim != 2 or vectors.shape[0] != len(node_ids):
        raise ValueError(f"Invalid embedding matrix cache: {matrix_path}")
    logger.info(
        "Embedding matrix ready: nodes=%d dimension=%d size_gib=%.2f path=%s",
        *vectors.shape,
        vectors.nbytes / 1024**3,
        matrix_path,
    )
    return _EmbeddingTable(vectors, {node_id: index for index, node_id in enumerate(node_ids)})


def _write_embedding_matrix(
    source: Path,
    target: Path,
    metadata_path: Path,
    node_ids: list[str],
    metadata: dict[str, Any],
) -> None:
    started = perf_counter()
    temporary = target.with_name(f"{target.name}.tmp")
    try:
        with np.load(source) as archive:
            dimension = int(np.asarray(archive[node_ids[0]]).shape[-1])
            matrix = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(len(node_ids), dimension),
            )
            checkpoint = max(1, len(node_ids) // 10)
            for index, node_id in enumerate(node_ids, 1):
                vector = np.asarray(archive[node_id], dtype=np.float32)
                if vector.shape != (dimension,):
                    raise ValueError(f"Invalid embedding shape for {node_id}: {vector.shape}")
                matrix[index - 1] = vector
                if index == len(node_ids) or index % checkpoint == 0:
                    logger.info("Embedding matrix conversion: %d/%d", index, len(node_ids))
            matrix.flush()
            del matrix
        temporary.replace(target)
        elapsed = perf_counter() - started
        metadata_path.write_text(
            json.dumps(
                {**metadata, "dimension": dimension, "conversion_seconds": elapsed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    logger.info("Embedding matrix converted once in %.1fs: %s", elapsed, target)


def _paper_aware_batches(
    graph: EvidenceGraph, samples: list[dict], batch_size: int, seed: int
) -> list[list[dict]]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[tuple(sorted(_pair_papers(graph, [sample])))].append(sample)
    rng = random.Random(seed)
    groups = list(grouped.values())
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    units = [
        group[start : start + batch_size]
        for group in groups
        for start in range(0, len(group), batch_size)
    ]
    units.sort(key=len, reverse=True)
    batches: list[list[dict]] = []
    for unit in units:
        target = next(
            (batch for batch in batches if len(batch) + len(unit) <= batch_size), None
        )
        if target is None:
            batches.append(list(unit))
        else:
            target.extend(unit)
    return batches


def _prepare_training_batches(
    graph: EvidenceGraph,
    samples: list[dict],
    query_embeddings: dict[str, np.ndarray],
    embedding_positions: dict[str, int],
    paper_index,
    batch_size: int,
    seed: int,
    device,
    input_dtype,
) -> list[_TrainingBatch]:
    prepared = []
    batches = _paper_aware_batches(graph, samples, batch_size, seed)
    for index, batch in enumerate(batches):
        papers = _pair_papers(graph, batch)
        batch_graph = _paper_subgraph(graph, papers, index=paper_index)
        node_indices, edge_indices, ids_by_type = build_heterotopology(
            batch_graph, embedding_positions
        )
        prepared.append(
            _TrainingBatch(
                batch,
                {key: value.to(device) for key, value in node_indices.items()},
                {key: value.to(device) for key, value in edge_indices.items()},
                _node_positions(ids_by_type),
                _relation_triples(batch_graph, papers, seed + index),
                _query_tensor(query_embeddings, batch).to(
                    device=device, dtype=input_dtype
                ),
            )
        )
    return prepared


def _query_tensor(query_embeddings: dict[str, np.ndarray], batch: list[dict]):
    import torch

    return torch.from_numpy(
        np.stack([query_embeddings[row["query_id"]] for row in batch]).astype(
            np.float32, copy=False
        )
    )


def _resolve_precision(torch, device, requested: str):
    value = requested.strip().lower()
    if value not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("precision must be auto, bf16, fp16, or fp32")
    if value == "auto":
        value = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if device.type == "cuda" else "fp32"
        )
    if value in {"bf16", "fp16"} and device.type != "cuda":
        raise ValueError(f"{value} graph training requires a CUDA device")
    if value == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support bf16")
    return value, {"bf16": torch.bfloat16, "fp16": torch.float16}.get(value)


def _autocast(torch, device, dtype):
    return (
        torch.autocast(device_type=device.type, dtype=dtype)
        if dtype is not None
        else nullcontext()
    )


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _similarity(
    embeddings: _EmbeddingTable, positive: str, negative: str
) -> float:
    if negative not in embeddings:
        return -1.0
    return float(embeddings[positive] @ embeddings[negative])


def _relation_triples(
    graph, train_papers: set[str], seed: int
) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    relations = {"caption_of", "refers_to", "derived_from", "next_sentence"}
    edges = [
        edge
        for edge in graph.edges
        if graph.nodes[edge.src].paper_id in train_papers
        and edge.relation.value in relations
        and edge.confidence >= 0.8
    ]
    neighbors: dict[str, set[str]] = defaultdict(set)
    pools: dict[tuple[str, object], list[str]] = defaultdict(list)
    for edge in edges:
        neighbors[edge.src].add(edge.dst)
    for node_id, node in graph.nodes.items():
        if node.paper_id in train_papers:
            pools[(node.paper_id, node.node_type)].append(node_id)
    triples = []
    for edge in edges:
        source = graph.nodes[edge.src]
        target = graph.nodes[edge.dst]
        candidates = [
            node_id
            for node_id in pools[(source.paper_id, target.node_type)]
            if node_id != edge.src and node_id not in neighbors[edge.src]
        ]
        if candidates:
            triples.append((edge.src, edge.dst, rng.choice(candidates)))
    return triples


def count_relation_triples(graph, paper_ids: set[str]) -> int:
    return len(_relation_triples(graph, paper_ids, seed=0))


def _sample_nodes(hidden, positions, samples, key):
    import torch

    return torch.stack(
        [hidden[positions[row[key]][0]][positions[row[key]][1]] for row in samples]
    )


def _triple_nodes(hidden, positions, triples, index):
    import torch

    return torch.stack(
        [hidden[positions[row[index]][0]][positions[row[index]][1]] for row in triples]
    )


def _graph_metadata(graph: EvidenceGraph) -> tuple[list[str], list[tuple[str, str, str]]]:
    node_types = sorted({node.node_type.value for node in graph.nodes.values()})
    edge_types: set[tuple[str, str, str]] = set()
    for edge in graph.edges:
        source = graph.nodes[edge.src].node_type.value
        target = graph.nodes[edge.dst].node_type.value
        edge_types.add((source, edge.relation.value, target))
        edge_types.add((target, f"rev_{edge.relation.value}", source))
    return node_types, sorted(edge_types)


def _validate_paper_local_edges(graph: EvidenceGraph) -> None:
    if any(
        graph.nodes[edge.src].paper_id != graph.nodes[edge.dst].paper_id
        for edge in graph.edges
    ):
        raise ValueError("Batched graph training requires all evidence edges within one paper")


def _pair_papers(graph: EvidenceGraph, samples: list[dict]) -> set[str]:
    return {
        graph.nodes[row[key]].paper_id
        for row in samples
        for key in ("positive_node_id", "negative_node_id")
    }


def _paper_graph_index(
    graph: EvidenceGraph,
) -> tuple[dict[str, list[str]], dict[str, list[EvidenceEdge]]]:
    nodes: dict[str, list[str]] = defaultdict(list)
    edges: dict[str, list[EvidenceEdge]] = defaultdict(list)
    for node_id, node in graph.nodes.items():
        nodes[node.paper_id].append(node_id)
    for edge in graph.edges:
        edges[graph.nodes[edge.src].paper_id].append(edge)
    return dict(nodes), dict(edges)


def _paper_subgraph(
    graph: EvidenceGraph,
    paper_ids: set[str],
    *,
    index: tuple[dict[str, list[str]], dict[str, list[EvidenceEdge]]] | None = None,
) -> EvidenceGraph:
    nodes_by_paper, edges_by_paper = index or _paper_graph_index(graph)
    result = EvidenceGraph()
    result.extend(
        (
            graph.nodes[node_id]
            for paper_id in paper_ids
            for node_id in nodes_by_paper.get(paper_id, ())
        ),
        (
            edge
            for paper_id in paper_ids
            for edge in edges_by_paper.get(paper_id, ())
        ),
    )
    return result


def _node_positions(ids_by_type):
    return {
        node_id: (node_type, index)
        for node_type, node_ids in ids_by_type.items()
        for index, node_id in enumerate(node_ids)
    }


def _export_hgt(
    model,
    graph: EvidenceGraph,
    base_tensor,
    embedding_positions: dict[str, int],
    output: str | Path,
    metadata: dict,
    *,
    paper_index: tuple[dict[str, list[str]], dict[str, list[EvidenceEdge]]],
    paper_batch_size: int,
    amp_dtype,
) -> Path:
    import torch

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    model.eval()
    papers = sorted({node.paper_id for node in graph.nodes.values()})
    ordered = []
    device = next(model.parameters()).device
    for start in range(0, len(papers), paper_batch_size):
        batch_graph = _paper_subgraph(
            graph,
            set(papers[start : start + paper_batch_size]),
            index=paper_index,
        )
        node_indices, edge_indices, ids_by_type = build_heterotopology(
            batch_graph, embedding_positions
        )
        node_indices = {key: value.to(device) for key, value in node_indices.items()}
        edge_indices = {key: value.to(device) for key, value in edge_indices.items()}
        with torch.no_grad(), _autocast(torch, device, amp_dtype):
            hidden = model.encode_graph(
                {
                    node_type: base_tensor.index_select(0, indices)
                    for node_type, indices in node_indices.items()
                },
                edge_indices,
            )
        for node_type in sorted(ids_by_type):
            vectors = hidden[node_type].float().cpu().numpy()
            ordered.extend(zip(ids_by_type[node_type], vectors, strict=True))
    np.save(root / "graph_embeddings.npy", np.stack([vector for _, vector in ordered]))
    (root / "node_ids.json").write_text(
        json.dumps([node_id for node_id, _ in ordered], ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "training.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.jit.trace(
        model.query_projection.cpu().eval(), torch.zeros(metadata["input_dimension"])
    ).save(
        str(root / "query_projector.pt")
    )
    return root
