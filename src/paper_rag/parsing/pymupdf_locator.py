from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from paper_rag.domain import BoundingBox


@dataclass(frozen=True, slots=True)
class LocatedSentence:
    page: int
    bbox: BoundingBox
    level: str  # exact, substring, paragraph_fallback


def locate_sentence(pdf_path: str | Path, page_number: int, sentence: str) -> LocatedSentence | None:
    """Locate a sentence with PyMuPDF; page_number is zero-based internally."""
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install PyMuPDF in the paper-rag Conda environment") from exc

    with fitz.open(str(pdf_path)) as document:
        if page_number < 0 or page_number >= document.page_count:
            return None
        page = document[page_number]
        variants = [(sentence, "exact")]
        words = sentence.split()
        if len(words) >= 12:
            variants.append((" ".join(words[3:12]), "substring"))
        for needle, level in variants:
            rectangles = page.search_for(needle)
            if rectangles:
                x0 = min(rect.x0 for rect in rectangles)
                y0 = min(rect.y0 for rect in rectangles)
                x1 = max(rect.x1 for rect in rectangles)
                y1 = max(rect.y1 for rect in rectangles)
                return LocatedSentence(page_number + 1, BoundingBox(x0, y0, x1, y1), level)
    return None


def locate_sentence_batch(
    pdf_path: str | Path,
    sentences: Iterable[tuple[str, int, str]],
) -> dict[str, LocatedSentence]:
    """Locate many sentences in one PDF open; input page numbers are one-based."""
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install PyMuPDF in the active Conda environment") from exc

    found: dict[str, LocatedSentence] = {}
    with fitz.open(str(pdf_path)) as document:
        for node_id, page_number, sentence in sentences:
            page_index = page_number - 1
            if page_index < 0 or page_index >= document.page_count:
                continue
            page = document[page_index]
            variants = [(sentence, "exact")]
            words = sentence.split()
            if len(words) >= 12:
                variants.append((" ".join(words[3:12]), "substring"))
            for needle, level in variants:
                rectangles = page.search_for(needle)
                if not rectangles:
                    continue
                x0 = min(rect.x0 for rect in rectangles)
                y0 = min(rect.y0 for rect in rectangles)
                x1 = max(rect.x1 for rect in rectangles)
                y1 = max(rect.y1 for rect in rectangles)
                found[node_id] = LocatedSentence(
                    page_number,
                    BoundingBox(x0, y0, x1, y1),
                    level,
                )
                break
    return found
