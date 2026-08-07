from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from paper_rag.evidence_graph import load_graph
from paper_rag.models import HGTConfig, build_heterodata, create_hgt_model
from paper_rag.models.losses import query_evidence_margin_loss, relation_info_nce


def load_npz(path: str) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {key: archive[key] for key in archive.files}


def read_query_samples(path: str | None) -> list[dict]:
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="PoC trainer for the SRMG HGT adapter")
    parser.add_argument("graph")
    parser.add_argument("base_embeddings")
    parser.add_argument("--query-samples", help="JSONL: query_id/positive_node_id/negative_node_id")
    parser.add_argument("--query-embeddings", help="NPZ keyed by query_id; dimension must match base embeddings")
    parser.add_argument("--output", default="outputs/srmg_index")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    graph = load_graph(args.graph)
    base_embeddings = load_npz(args.base_embeddings)
    if not base_embeddings:
        raise ValueError("Base embedding archive is empty")
    input_dimension = int(next(iter(base_embeddings.values())).shape[-1])
    data, ids_by_type = build_heterodata(graph, base_embeddings, add_reverse_edges=True)
    data = data.to(args.device)
    model = create_hgt_model(
        data.metadata(), HGTConfig(input_dimension=input_dimension)
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    positions = {
        node_id: (node_type, index)
        for node_type, node_ids in ids_by_type.items()
        for index, node_id in enumerate(node_ids)
    }
    relation_triples = build_relation_triples(graph, ids_by_type, positions, args.seed)
    query_samples = read_query_samples(args.query_samples)
    query_embeddings = load_npz(args.query_embeddings) if args.query_embeddings else {}
    if not relation_triples and not query_samples:
        raise ValueError("No relation triples or query ranking samples are available")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        hidden = model.encode_graph(data.x_dict, data.edge_index_dict)
        losses = []
        if relation_triples:
            anchor = torch.stack([hidden[positions[a][0]][positions[a][1]] for a, _, _ in relation_triples])
            positive = torch.stack([hidden[positions[p][0]][positions[p][1]] for _, p, _ in relation_triples])
            negative = torch.stack([hidden[positions[n][0]][positions[n][1]] for _, _, n in relation_triples])
            losses.append(args.relation_weight * relation_info_nce(anchor, positive, negative[:, None, :]))
        if query_samples:
            query_tensor = torch.from_numpy(
                np.stack([query_embeddings[sample["query_id"]] for sample in query_samples]).astype(np.float32)
            ).to(args.device)
            query_hidden = model.encode_query(query_tensor)
            positives = torch.stack(
                [
                    hidden[positions[sample["positive_node_id"]][0]][positions[sample["positive_node_id"]][1]]
                    for sample in query_samples
                ]
            )
            negatives = torch.stack(
                [
                    hidden[positions[sample["negative_node_id"]][0]][positions[sample["negative_node_id"]][1]]
                    for sample in query_samples
                ]
            )
            losses.append(query_evidence_margin_loss(query_hidden, positives, negatives))
        loss = sum(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"epoch={epoch + 1} loss={float(loss.detach().cpu()):.6f}")

    export_artifacts(model, data, ids_by_type, args.output, args.device, input_dimension)


def build_relation_triples(graph, ids_by_type, positions, seed: int) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    triples: list[tuple[str, str, str]] = []
    for edge in graph.edges:
        # Explicit parser-derived links are the relation supervision; sequence edges are omitted.
        if edge.relation.value not in {"caption_of", "refers_to", "derived_from"}:
            continue
        positive_type = positions[edge.dst][0]
        candidates = [
            node_id
            for node_id in ids_by_type[positive_type]
            if node_id != edge.dst and graph.nodes[node_id].paper_id == graph.nodes[edge.src].paper_id
        ]
        if candidates:
            triples.append((edge.src, edge.dst, rng.choice(candidates)))
    return triples


def export_artifacts(model, data, ids_by_type, output: str, device: str, input_dimension: int) -> None:
    import torch

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        hidden = model.encode_graph(data.x_dict, data.edge_index_dict)
    ordered_ids: list[str] = []
    rows: list[np.ndarray] = []
    for node_type in sorted(ids_by_type):
        for index, node_id in enumerate(ids_by_type[node_type]):
            ordered_ids.append(node_id)
            rows.append(hidden[node_type][index].float().cpu().numpy())
    np.save(root / "graph_embeddings.npy", np.stack(rows).astype(np.float32))
    (root / "node_ids.json").write_text(
        json.dumps(ordered_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    projector = model.query_projection.to("cpu").eval()
    example = torch.zeros(input_dimension)
    traced = torch.jit.trace(projector, example)
    traced.save(str(root / "query_projector.pt"))
    print(f"Exported SRMG artifacts to {root.resolve()}")


if __name__ == "__main__":
    main()
