from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_rag.io import read_jsonl, write_json, write_jsonl


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


def safe_name(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    readable = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return f"{readable[-60:]}-{digest}"
