from __future__ import annotations

import re

_FIGURE_REFERENCE = re.compile(r"(?i)\bfig(?:ure)?\.?\s*(\d+[a-z]?)")


def extract_figure_labels(text: str) -> set[str]:
    return {match.group(1).lower() for match in _FIGURE_REFERENCE.finditer(text)}


def normalize_figure_label(caption: str) -> str | None:
    match = _FIGURE_REFERENCE.search(caption)
    return match.group(1).lower() if match else None

