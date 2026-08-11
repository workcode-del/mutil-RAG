from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

import numpy as np

from paper_rag.domain import NodeType, SearchHit
from paper_rag.evidence_graph import EvidenceGraph


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\W\s]", re.UNICODE)


def tokenize(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(value)]


class BM25EvidenceStore:
    """Dependency-free BM25 candidate baseline over evidence-node text."""

    def __init__(self, graph: EvidenceGraph, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.graph = graph
        self.k1 = k1
        self.b = b
        self.tokens = {
            node_id: tokenize(node.searchable_text)
            for node_id, node in graph.nodes.items()
            if node.searchable_text.strip()
        }
        self.term_frequencies = {
            node_id: Counter(tokens) for node_id, tokens in self.tokens.items()
        }

    def search(
        self,
        query: str,
        query_vector: np.ndarray | None,
        node_types: Iterable[NodeType],
        per_type_top_k: int = 25,
        paper_ids: set[str] | None = None,
        candidate_node_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        query_terms = Counter(tokenize(query))
        hits: list[SearchHit] = []
        for node_type in node_types:
            candidate_ids = [
                node_id
                for node_id in self.tokens
                if self.graph.nodes[node_id].node_type is node_type
                and (not paper_ids or self.graph.nodes[node_id].paper_id in paper_ids)
                and (not candidate_node_ids or node_id in candidate_node_ids)
            ]
            if not candidate_ids:
                continue
            average_length = sum(len(self.tokens[node_id]) for node_id in candidate_ids) / len(
                candidate_ids
            )
            document_frequency = {
                term: sum(term in self.term_frequencies[node_id] for node_id in candidate_ids)
                for term in query_terms
            }
            ranked: list[tuple[float, str]] = []
            for node_id in candidate_ids:
                length = len(self.tokens[node_id])
                frequencies = self.term_frequencies[node_id]
                score = 0.0
                for term, query_frequency in query_terms.items():
                    frequency = frequencies.get(term, 0)
                    if not frequency:
                        continue
                    count = len(candidate_ids)
                    frequency_in_corpus = document_frequency[term]
                    inverse_document_frequency = math.log(
                        1.0 + (count - frequency_in_corpus + 0.5) / (frequency_in_corpus + 0.5)
                    )
                    denominator = frequency + self.k1 * (
                        1.0 - self.b + self.b * length / max(average_length, 1.0)
                    )
                    score += query_frequency * inverse_document_frequency * (
                        frequency * (self.k1 + 1.0) / denominator
                    )
                if score > 0.0:
                    ranked.append((score, node_id))
            for score, node_id in sorted(ranked, reverse=True)[:per_type_top_k]:
                node = self.graph.nodes[node_id]
                hits.append(
                    SearchHit(
                        node_id=node_id,
                        paper_id=node.paper_id,
                        node_type=node.node_type,
                        score=score,
                        score_components={"bm25": score},
                    )
                )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)
