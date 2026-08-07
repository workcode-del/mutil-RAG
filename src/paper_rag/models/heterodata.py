from __future__ import annotations

from collections import defaultdict

import numpy as np

from paper_rag.evidence_graph import EvidenceGraph


def build_heterodata(
    graph: EvidenceGraph,
    embeddings: dict[str, np.ndarray],
    add_reverse_edges: bool = True,
):
    """Map the stable evidence graph schema to PyG HeteroData.

    Returns `(data, node_ids_by_type)`, which is the mapping needed to translate
    HGT rows back to evidence IDs.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install graph training dependencies") from exc

    node_ids_by_type: dict[str, list[str]] = defaultdict(list)
    for node_id, node in graph.nodes.items():
        node_ids_by_type[node.node_type.value].append(node_id)
    position: dict[str, tuple[str, int]] = {}
    data = HeteroData()
    for node_type, node_ids in node_ids_by_type.items():
        node_ids.sort()
        vectors = []
        for index, node_id in enumerate(node_ids):
            if node_id not in embeddings:
                raise KeyError(f"Missing base embedding for {node_id}")
            position[node_id] = (node_type, index)
            vectors.append(np.asarray(embeddings[node_id], dtype=np.float32))
        data[node_type].x = torch.from_numpy(np.stack(vectors))

    typed_edges: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for edge in graph.edges:
        src_type, src_index = position[edge.src]
        dst_type, dst_index = position[edge.dst]
        typed_edges[(src_type, edge.relation.value, dst_type)].append((src_index, dst_index))
        if add_reverse_edges:
            typed_edges[(dst_type, f"rev_{edge.relation.value}", src_type)].append(
                (dst_index, src_index)
            )
    for edge_type, pairs in typed_edges.items():
        data[edge_type].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return data, dict(node_ids_by_type)

