from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Install PyYAML to load configuration") from exc
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level configuration must be a mapping")
    return data


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    root: Path
    data: Path
    parsed: Path
    figures: Path
    chart_tables: Path
    index: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "RuntimePaths":
        root_path = Path(root).resolve()
        data = root_path / "data"
        return cls(
            root=root_path,
            data=data,
            parsed=data / "parsed",
            figures=data / "figures",
            chart_tables=data / "chart_tables",
            index=data / "index",
        )

    def create(self) -> None:
        for path in (self.data, self.parsed, self.figures, self.chart_tables, self.index):
            path.mkdir(parents=True, exist_ok=True)
