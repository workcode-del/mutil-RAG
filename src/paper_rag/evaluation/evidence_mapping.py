from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from paper_rag.domain import NodeType
from paper_rag.evidence_graph import EvidenceGraph


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    evidence: str
    node_id: str | None
    score: float
    method: str

    def to_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


def normalize_evidence(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def map_evidence(
    graph: EvidenceGraph,
    paper_id: str,
    evidence: list[str],
    *,
    min_score: float = 0.85,
) -> list[EvidenceMatch]:
    candidates = [
        node
        for node in graph.nodes.values()
        if node.paper_id == paper_id
        and node.node_type in {NodeType.SENTENCE, NodeType.CAPTION, NodeType.CHART_DATA}
        and normalize_evidence(node.searchable_text)
    ]
    normalized = {node.node_id: normalize_evidence(node.searchable_text) for node in candidates}
    exact: dict[str, str] = {}
    for node in candidates:
        exact.setdefault(normalized[node.node_id], node.node_id)

    matches: list[EvidenceMatch] = []
    for source in evidence:
        query = normalize_evidence(source)
        if not query:
            matches.append(EvidenceMatch(source, None, 0.0, "empty"))
            continue
        if query in exact:
            matches.append(EvidenceMatch(source, exact[query], 1.0, "exact"))
            continue
        node_id, score = _best_match(query, normalized)
        matches.append(
            EvidenceMatch(source, node_id if score >= min_score else None, score, "fuzzy")
        )
    return matches


def _best_match(query: str, candidates: dict[str, str]) -> tuple[str | None, float]:
    best_id: str | None = None
    best_score = 0.0
    query_tokens = set(query.split())
    for node_id, candidate in candidates.items():
        candidate_tokens = set(candidate.split())
        union = query_tokens | candidate_tokens
        token_score = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
        sequence_score = SequenceMatcher(None, query, candidate, autojunk=False).ratio()
        score = max(token_score, sequence_score)
        if score > best_score:
            best_id, best_score = node_id, score
    return best_id, best_score
