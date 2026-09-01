from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None
) -> dict[str, float]:
    """Fuse heterogeneous scorers without assuming comparable raw score scales."""
    weights = weights or {}
    result: dict[str, float] = defaultdict(float)
    for scorer, node_ids in rankings.items():
        weight = weights.get(scorer, 1.0)
        for rank, node_id in enumerate(node_ids, start=1):
            result[node_id] += weight / (k + rank)
    return dict(result)
