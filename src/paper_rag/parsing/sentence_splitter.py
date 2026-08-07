from __future__ import annotations

import re

_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z0-9\u4e00-\u9fff])")
_ABBREVIATIONS = ("Fig.", "Figs.", "Eq.", "Eqs.", "Ref.", "Refs.", "et al.", "i.e.", "e.g.")


def split_sentences(text: str, minimum_chars: int = 8) -> list[str]:
    """Conservative sentence splitter suitable for scientific paragraphs.

    It deliberately avoids heavyweight NLP dependencies. A production experiment may
    replace it with SciSpacy while keeping this function's output contract.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    protected = normalized
    replacements: dict[str, str] = {}
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        marker = f"__ABBR_{index}__"
        replacements[marker] = abbreviation
        protected = protected.replace(abbreviation, marker)
    parts = _BOUNDARY.split(protected)
    result: list[str] = []
    for part in parts:
        for marker, abbreviation in replacements.items():
            part = part.replace(marker, abbreviation)
        part = part.strip()
        if len(part) < minimum_chars and result:
            result[-1] = f"{result[-1]} {part}"
        elif part:
            result.append(part)
    return result
