from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper_rag.bootstrap import build_embedder
from paper_rag.config import load_yaml
from paper_rag.evidence_graph import load_graph
from paper_rag.io import read_jsonl, write_jsonl
from paper_rag.models import HGTConfig, build_heterodata, create_hgt_model
from paper_rag.models.losses import query_evidence_margin_loss, relation_info_nce


logger = logging.getLogger(__name__)


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
    embeddings = _load_npz(embeddings_path) if embeddings_path else {}
    rows = []
    for sample in read_jsonl(samples_path):
        positives = [node_id for node_id in sample["relevant_node_ids"] if node_id in graph.nodes]
        candidates = sample.get("candidate_node_ids") or [
            node_id
            for node_id, node in graph.nodes.items()
            if node.paper_id == str(sample["paper_id"])
        ]
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
                if positive in embeddings
                else rng.choice(pool)
            )
            rows.append(
                {
                    "query_id": str(sample["query_id"]),
                    "query": str(sample["query"]),
                    "positive_node_id": positive,
                    "negative_node_id": negative,
                }
            )
    target = write_jsonl(output, rows)
    logger.info("Training pairs ready: pairs=%d output=%s", len(rows), target)
    return target


def embed_training_queries(
    samples_path: str | Path,
    output: str | Path,
    config_path: str | Path,
    *,
    batch_size: int = 16,
) -> Path:
    samples = read_jsonl(samples_path)
    embedder = build_embedder(load_yaml(config_path))
    vectors: dict[str, np.ndarray] = {}
    queries = {str(sample["query_id"]): str(sample["query"]) for sample in samples}
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
    learning_rate: float = 1e-3,
    relation_weight: float = 0.2,
    seed: int = 42,
    device: str = "cuda",
    hidden_dimension: int = 256,
    layers: int = 2,
    heads: int = 4,
) -> Path:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    graph = load_graph(graph_path)
    base_embeddings = _load_npz(base_embeddings_path)
    query_embeddings = _load_npz(query_embeddings_path)
    samples = read_jsonl(query_pairs_path)
    if not samples:
        raise ValueError("No trainable query pairs were produced")
    input_dimension = int(next(iter(base_embeddings.values())).shape[-1])
    logger.info("Preparing HGT graph and model: nodes=%d device=%s", len(graph.nodes), device)
    data, ids_by_type = build_heterodata(graph, base_embeddings, add_reverse_edges=True)
    data = data.to(device)
    model = create_hgt_model(
        data.metadata(),
        HGTConfig(input_dimension, hidden_dimension, layers, heads),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    positions = {
        node_id: (node_type, index)
        for node_type, node_ids in ids_by_type.items()
        for index, node_id in enumerate(node_ids)
    }
    train_papers = {graph.nodes[row["positive_node_id"]].paper_id for row in samples}
    relations = _relation_triples(graph, train_papers, seed)
    logger.info(
        "HGT training: nodes=%d query_pairs=%d relation_triples=%d epochs=%d device=%s",
        len(graph.nodes),
        len(samples),
        len(relations),
        epochs,
        device,
    )
    query_tensor = torch.from_numpy(
        np.stack([query_embeddings[row["query_id"]] for row in samples]).astype(np.float32)
    ).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        hidden = model.encode_graph(data.x_dict, data.edge_index_dict)
        query_hidden = model.encode_query(query_tensor)
        positives = _sample_nodes(hidden, positions, samples, "positive_node_id")
        negatives = _sample_nodes(hidden, positions, samples, "negative_node_id")
        loss = query_evidence_margin_loss(query_hidden, positives, negatives)
        if relations:
            anchors = _triple_nodes(hidden, positions, relations, 0)
            related = _triple_nodes(hidden, positions, relations, 1)
            unrelated = _triple_nodes(hidden, positions, relations, 2)
            loss += relation_weight * relation_info_nce(anchors, related, unrelated[:, None, :])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logger.info("HGT epoch %d/%d loss=%.6f", epoch + 1, epochs, float(loss.detach().cpu()))

    metadata = {
        "graph_sha256": hashlib.sha256(Path(graph_path).read_bytes()).hexdigest(),
        "query_pairs": len(samples),
        "train_query_ids": sorted({row["query_id"] for row in samples}),
        "relation_triples": len(relations),
        "input_dimension": input_dimension,
        "hidden_dimension": hidden_dimension,
        "layers": layers,
        "heads": heads,
        "epochs": epochs,
        "seed": seed,
    }
    artifacts = _export_hgt(model, data, ids_by_type, output, metadata)
    logger.info("HGT artifacts ready: %s", artifacts)
    return artifacts


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {key: archive[key] for key in archive.files}


def _similarity(
    embeddings: dict[str, np.ndarray], positive: str, negative: str
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


def _export_hgt(model, data, ids_by_type, output: str | Path, metadata: dict) -> Path:
    import torch

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        hidden = model.encode_graph(data.x_dict, data.edge_index_dict)
    ordered = [
        (node_id, hidden[node_type][index].float().cpu().numpy())
        for node_type in sorted(ids_by_type)
        for index, node_id in enumerate(ids_by_type[node_type])
    ]
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
