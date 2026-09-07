from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from paper_rag.io import read_jsonl, write_json, write_jsonl

if TYPE_CHECKING:
    from paper_rag.evidence_graph import EvidenceGraph


PROCESSED_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class BenchmarkLayout:
    name: str
    root: Path

    @classmethod
    def create(cls, name: str, root: str | Path) -> "BenchmarkLayout":
        layout = cls(name, Path(root) / name)
        layout.raw.mkdir(parents=True, exist_ok=True)
        layout.processed.mkdir(parents=True, exist_ok=True)
        layout.reports.mkdir(parents=True, exist_ok=True)
        return layout

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def graph(self) -> Path:
        return self.processed / "graph.json"

    def samples(self, split: str = "test") -> Path:
        return self.processed / f"{split}.jsonl"


def grouped_split(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    train_percent: int = 70,
    dev_percent: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    if train_percent < 1 or dev_percent < 1 or train_percent + dev_percent >= 100:
        raise ValueError("Invalid grouped split percentages")
    result = {"train": [], "dev": [], "test": []}
    for row in rows:
        digest = hashlib.sha1(str(row[group_key]).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % 100
        if bucket < train_percent:
            split = "train"
        elif bucket < train_percent + dev_percent:
            split = "dev"
        else:
            split = "test"
        result[split].append(row)
    return result


def connected_grouped_split(
    rows: list[dict[str, Any]],
    *,
    members_key: str = "paper_ids",
    train_percent: int = 70,
    dev_percent: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    """Keep rows connected through any shared document in the same split."""
    if train_percent < 1 or dev_percent < 1 or train_percent + dev_percent >= 100:
        raise ValueError("Invalid grouped split percentages")
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        members = {str(value) for value in row.get(members_key, [])}
        if not members and row.get("paper_id") is not None:
            members.add(str(row["paper_id"]))
        for member in members:
            if member in owner:
                union(index, owner[member])
            else:
                owner[member] = index

    component_members: dict[int, set[str]] = {}
    for member, index in owner.items():
        component_members.setdefault(find(index), set()).add(member)
    components: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        root = find(index)
        component = "\0".join(sorted(component_members.get(root, {f"row:{index}"})))
        components.setdefault(component, []).append(row)

    names = ("train", "dev", "test")
    percentages = (train_percent, dev_percent, 100 - train_percent - dev_percent)
    targets = {name: len(rows) * percent / 100 for name, percent in zip(names, percentages)}
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    ordered = sorted(
        components.items(),
        key=lambda item: (-len(item[1]), hashlib.sha1(item[0].encode()).hexdigest()),
    )
    require_nonempty = len(ordered) >= len(names)
    for position, (_, component_rows) in enumerate(ordered):
        remaining = len(ordered) - position - 1
        eligible = [
            name
            for name in names
            if not require_nonempty
            or sum(not result[other] for other in names if other != name) <= remaining
        ]
        chosen = min(
            eligible,
            key=lambda name: (
                (len(result[name]) + len(component_rows) - targets[name]) ** 2
                - (len(result[name]) - targets[name]) ** 2,
                names.index(name),
            ),
        )
        result[chosen].extend(component_rows)
    return result


def safe_name(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    readable = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return f"{readable[-60:]}-{digest}"


def validate_prepared_samples(
    dataset: str,
    graph: "EvidenceGraph",
    rows: list[dict[str, Any]],
) -> None:
    """Reject empty or internally inconsistent converted benchmark artifacts."""
    if not graph.nodes:
        raise RuntimeError(f"{dataset} preparation produced an empty graph")
    if not rows:
        raise RuntimeError(f"{dataset} preparation produced no usable samples")

    query_ids: set[str] = set()
    graph_ids = graph.nodes.keys()
    for index, row in enumerate(rows):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"{dataset} sample {index} has no query_id")
        if query_id in query_ids:
            raise ValueError(f"{dataset} has duplicate query_id: {query_id}")
        query_ids.add(query_id)
        if not str(row.get("query", "")).strip():
            raise ValueError(f"{dataset} sample {query_id} has an empty query")

        gold = {str(value) for value in row.get("relevant_node_ids", ())}
        if not gold:
            raise ValueError(f"{dataset} sample {query_id} has no gold evidence")
        candidates = {str(value) for value in row.get("candidate_node_ids", ())}
        invalid = gold - graph_ids
        if candidates:
            invalid.update(candidates - graph_ids)
            invalid.update(gold - candidates)
        if invalid:
            raise ValueError(
                f"{dataset} sample {query_id} has invalid evidence IDs: "
                f"{sorted(invalid)[:10]}"
            )
