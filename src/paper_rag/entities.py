from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import TYPE_CHECKING

from paper_rag.domain import NodeType

if TYPE_CHECKING:
    from paper_rag.evidence_graph import EvidenceGraph


ENTITY_EXTRACTION_VERSION = "scientific-rules-v1"

_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
}
_IGNORED_TOKENS = {
    "api", "auc", "cpu", "csv", "dpi", "f1", "fig", "figure", "gpu", "html", "http",
    "json", "jsonl", "mae", "map", "mape", "mpa", "gpa", "pdf", "ram", "rmse", "table",
    "these", "those", "ui", "url", "what", "which", "xml",
}
_PREFIXES = (
    "a ", "an ", "the ", "our ", "this ", "that ", "proposed ", "existing ", "new ",
    "using ", "based on ", "which ", "what ", "was ", "were ", "evaluated ", "on ",
    "accuracy ", "一种", "哪些", "什么", "本文采用", "本文", "使用", "采用", "基于", "研究",
    "提出的", "所研究的",
)


@dataclass(frozen=True, slots=True)
class ScientificEntity:
    text: str
    normalized: str
    entity_type: str
    start: int
    end: int
    confidence: float
    source: str = ENTITY_EXTRACTION_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ScientificEntityExtractor:
    """Deterministic scientific entity recognizer for retrieval-time diversity.

    It deliberately favors high-precision formulas, named artifacts and cue-bound
    noun phrases. Every mention is auditable and no external model is required.
    """

    _artifact = re.compile(
        r"\b(?:[A-Z]{2,}[A-Za-z0-9.+-]*|[A-Z][A-Za-z]+(?:[A-Z][A-Za-z]+)+|"
        r"[A-Z][A-Za-z]+-\d+|[A-Z][A-Za-z]+\d+[A-Za-z0-9.+-]*|"
        r"[A-Za-z]+\d+[A-Za-z0-9.+-]*-[A-Za-z0-9.+-]+)\b"
    )
    _formula = re.compile(r"\b(?:[A-Z][a-z]?\d*(?:\.\d+)?){2,}\b")
    _alloy = re.compile(r"\b[A-Z][a-z]?(?:-\d+(?:\.\d+)?[A-Z][a-z]?){1,}\b")
    _english_phrase = re.compile(
        r"\b(?P<name>[A-Za-z][A-Za-z0-9.+-]*(?:\s+[A-Za-z][A-Za-z0-9.+-]*){0,3})\s+"
        r"(?P<cue>model|method|algorithm|architecture|dataset|benchmark|alloy|steel|"
        r"polymer|composite|material)\b",
        re.IGNORECASE,
    )
    _chinese_phrase = re.compile(
        r"(?P<name>[\u4e00-\u9fffA-Za-z0-9+.-]{1,12})"
        r"(?P<cue>模型|方法|算法|架构|数据集|基准|合金|钢|聚合物|复合材料|材料)"
    )

    def extract(self, text: str) -> list[ScientificEntity]:
        candidates: list[ScientificEntity] = []
        occupied: set[tuple[int, int, str]] = set()

        def add(value: str, entity_type: str, start: int, end: int, confidence: float) -> None:
            cleaned = self._clean(value)
            relative_start = value.find(cleaned)
            if relative_start >= 0:
                start += relative_start
                end = start + len(cleaned)
            normalized = normalize_entity(cleaned)
            identity = (start, end, normalized)
            if len(normalized) < 2 or identity in occupied or normalized in _IGNORED_TOKENS:
                return
            occupied.add(identity)
            candidates.append(
                ScientificEntity(cleaned, normalized, entity_type, start, end, confidence)
            )

        for match in self._alloy.finditer(text):
            add(match.group(), "material", match.start(), match.end(), 0.98)
        for match in self._formula.finditer(text):
            if _is_chemical_formula(match.group()):
                add(match.group(), "material", match.start(), match.end(), 0.98)
        for match in self._english_phrase.finditer(text):
            cue = match.group("cue").casefold()
            add(
                match.group("name"),
                _cue_type(cue),
                match.start("name"),
                match.end("name"),
                0.88,
            )
        for match in self._chinese_phrase.finditer(text):
            cue = match.group("cue")
            add(
                match.group("name") + cue,
                _cue_type(cue),
                match.start(),
                match.end(),
                0.88,
            )
        for match in self._artifact.finditer(text):
            entity_type = _context_type(text, match.start(), match.end())
            add(match.group(), entity_type, match.start(), match.end(), 0.82)

        selected: list[ScientificEntity] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (-item.confidence, -(item.end - item.start), item.start),
        ):
            if any(
                candidate.start >= existing.start
                and candidate.end <= existing.end
                and candidate.normalized in existing.normalized
                for existing in selected
            ):
                continue
            selected.append(candidate)
        return sorted(selected, key=lambda item: (item.start, -item.confidence, item.normalized))

    @staticmethod
    def _clean(value: str) -> str:
        result = " ".join(value.strip(" \t\r\n,.;:()[]{}").split())
        lowered = result.casefold()
        changed = True
        while changed:
            changed = False
            for prefix in _PREFIXES:
                if lowered.startswith(prefix.casefold()) and len(result) > len(prefix):
                    result = result[len(prefix) :].strip()
                    lowered = result.casefold()
                    changed = True
                    break
        return result


def normalize_entity(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def enrich_graph_entities(
    graph: EvidenceGraph,
    extractor: ScientificEntityExtractor | None = None,
) -> int:
    """Attach normalized entities and auditable mentions to every evidence node."""
    extractor = extractor or ScientificEntityExtractor()
    enriched = 0
    for node in graph.nodes.values():
        if node.node_type is NodeType.PAPER:
            continue
        text_digest = hashlib.sha256(node.searchable_text.encode("utf-8")).hexdigest()
        if (
            node.attributes.get("entity_extraction_version") == ENTITY_EXTRACTION_VERSION
            and node.attributes.get("entity_text_sha256") == text_digest
        ):
            continue
        mentions = extractor.extract(node.searchable_text)
        previous_auto = {
            normalize_entity(str(value))
            for value in node.attributes.get("auto_entities", [])
        }
        existing = {
            normalize_entity(str(value))
            for value in node.attributes.get("entities", [])
        } - previous_auto
        automatic = {mention.normalized for mention in mentions}
        node.attributes["entities"] = sorted(value for value in existing | automatic if value)
        node.attributes["auto_entities"] = sorted(automatic)
        node.attributes["entity_mentions"] = [mention.to_dict() for mention in mentions]
        node.attributes["entity_extraction_version"] = ENTITY_EXTRACTION_VERSION
        node.attributes["entity_text_sha256"] = text_digest
        enriched += 1
    return enriched


def _is_chemical_formula(value: str) -> bool:
    parts = re.findall(r"([A-Z][a-z]?)(?:\d*(?:\.\d+)?)", value)
    return len(parts) >= 2 and all(part in _ELEMENTS for part in parts)


def _cue_type(cue: str) -> str:
    normalized = cue.casefold()
    if normalized in {"dataset", "benchmark", "数据集", "基准"}:
        return "dataset"
    if normalized in {
        "alloy",
        "steel",
        "polymer",
        "composite",
        "material",
        "合金",
        "钢",
        "聚合物",
        "复合材料",
        "材料",
    }:
        return "material"
    if normalized in {"model", "architecture", "模型", "架构"}:
        return "model"
    return "method"


def _context_type(text: str, start: int, end: int) -> str:
    context = text[max(0, start - 40) : min(len(text), end + 40)].casefold()
    for cue in ("dataset", "benchmark", "数据集", "基准"):
        if cue in context:
            return "dataset"
    for cue in ("alloy", "steel", "material", "polymer", "composite", "合金", "材料", "钢"):
        if cue in context:
            return "material"
    for cue in ("model", "architecture", "模型", "架构"):
        if cue in context:
            return "model"
    for cue in ("method", "algorithm", "方法", "算法"):
        if cue in context:
            return "method"
    return "scientific_term"
