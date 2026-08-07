from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable

from paper_rag.domain import NodeType, SearchHit


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


@dataclass(slots=True)
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def standard_deviation(self) -> float:
        return sqrt(self.m2 / max(self.count - 1, 1))


@dataclass(slots=True)
class ScoreCalibrator:
    """Per-node-type, per-scorer z-score calibration fitted on validation data."""

    stats: dict[tuple[NodeType, str], RunningStats] = field(default_factory=dict)

    def fit(self, hits: Iterable[SearchHit]) -> None:
        for hit in hits:
            for scorer, value in hit.score_components.items():
                self.stats.setdefault((hit.node_type, scorer), RunningStats()).add(value)

    def transform(self, node_type: NodeType, scorer: str, value: float) -> float:
        stats = self.stats.get((node_type, scorer))
        if stats is None or stats.count < 2:
            return value
        deviation = max(stats.standard_deviation, 1e-6)
        return (value - stats.mean) / deviation

    def fuse(self, hit: SearchHit, weights: dict[str, float]) -> float:
        return sum(
            weights.get(scorer, 0.0) * self.transform(hit.node_type, scorer, value)
            for scorer, value in hit.score_components.items()
        )

