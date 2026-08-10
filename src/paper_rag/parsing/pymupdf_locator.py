from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
