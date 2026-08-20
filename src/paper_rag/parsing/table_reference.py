from __future__ import annotations

import re


_TABLE_REFERENCE = re.compile(r"(?i)(?:\btab(?:le)?\.?\s*|表\s*)(\d+[a-z]?)")


def extract_table_labels(text: str) -> set[str]:
    return {match.group(1).lower() for match in _TABLE_REFERENCE.finditer(text)}


def normalize_table_label(caption: str) -> str | None:
    match = _TABLE_REFERENCE.search(caption)
    return match.group(1).lower() if match else None
