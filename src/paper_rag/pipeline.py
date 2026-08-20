from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Protocol

import numpy as np

from paper_rag.domain import EvidenceForest, NodeType, QuerySpec, SearchHit
from paper_rag.embedding.base import Embedder
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.generation.base import Answer, AnswerGenerator
from paper_rag.reranking.base import Reranker
from paper_rag.retrieval.base import EvidenceRetriever


logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    def search(
        self,
        query: str,
        query_vector: np.ndarray | None,
        node_types: list[NodeType],
        per_type_top_k: int,
        paper_ids: set[str] | None = None,
        candidate_node_ids: set[str] | None = None,
    ) -> list[SearchHit]: ...


GraphScoreFunction = Callable[[np.ndarray, list[SearchHit]], dict[str, float]]


@dataclass(slots=True)
class PipelineResult:
    query: QuerySpec
    hits: list[SearchHit]
    forest: EvidenceForest
    answer: Answer | None = None


class ScientificRAGPipeline:
    def __init__(
        self,
        graph: EvidenceGraph,
        embedder: Embedder | None,
        vector_store: VectorStore,
        forest_retriever: EvidenceRetriever,
        graph_scorer: GraphScoreFunction | None = None,
        reranker: Reranker | None = None,
        generator: AnswerGenerator | None = None,
        default_per_type_top_k: int = 25,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.vector_store = vector_store
        self.forest_retriever = forest_retriever
        self.graph_scorer = graph_scorer
        self.reranker = reranker
        self.generator = generator
        self.default_per_type_top_k = default_per_type_top_k

    def run(
        self,
        query: QuerySpec,
        per_type_top_k: int | None = None,
        paper_ids: set[str] | None = None,
        candidate_node_ids: set[str] | None = None,
        log_stages: bool = False,
        query_vector: np.ndarray | None = None,
    ) -> PipelineResult:
        started = perf_counter()
        effective_top_k = per_type_top_k or self.default_per_type_top_k
        if query_vector is None and self.embedder:
            query_vector = self.embedder.embed_queries([query.query])[0]
        if log_stages:
            logger.info("Query stage: candidate recall (%s)", type(self.vector_store).__name__)
        hits = self.vector_store.search(
            query.query,
            query_vector,
            [
                NodeType.SENTENCE,
                NodeType.FIGURE,
                NodeType.TABLE,
                NodeType.CAPTION,
                NodeType.CHART_DATA,
            ],
            effective_top_k,
            paper_ids,
            candidate_node_ids,
        )
        logger.debug("Candidate recall complete: hits=%d", len(hits))
        if self.graph_scorer:
            if log_stages:
                logger.info("Query stage: HGT candidate scoring")
            if query_vector is None:
                raise ValueError("HGT scoring requires a query embedder")
            graph_scores = self.graph_scorer(query_vector, hits)
            for hit in hits:
                hit.score_components["hgt"] = graph_scores.get(hit.node_id, 0.0)
        if self.reranker:
            if log_stages:
                logger.info("Query stage: multimodal reranking (%s)", type(self.reranker).__name__)
            documents: list[str | dict[str, object]] = []
            for hit in hits:
                node = self.graph.nodes[hit.node_id]
                if node.node_type in {NodeType.FIGURE, NodeType.TABLE} and node.image_path:
                    documents.append(
                        {"image": node.image_path, "text": node.searchable_text or None}
                    )
                else:
                    documents.append(node.searchable_text)
            rerank_scores = self.reranker.score(query.query, documents)
            for hit, score in zip(hits, rerank_scores, strict=True):
                hit.score_components["reranker"] = score

        # Safe default is RRF. Learned calibrated fusion can replace this block.
        scorer_names = sorted({name for hit in hits for name in hit.score_components})
        rank_positions: dict[str, dict[str, int]] = {}
        for scorer in scorer_names:
            ranked = sorted(
                hits,
                key=lambda hit: hit.score_components.get(scorer, -1e9),
                reverse=True,
            )
            rank_positions[scorer] = {hit.node_id: rank for rank, hit in enumerate(ranked, 1)}
        for hit in hits:
            hit.score = sum(
                1.0 / (60 + rank_positions[scorer][hit.node_id]) for scorer in scorer_names
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        hits = self.forest_retriever.rank_hits(query, hits)

        if log_stages:
            logger.info(
                "Query stage: evidence retrieval (%s)",
                type(self.forest_retriever).__name__,
            )
        forest = self.forest_retriever.retrieve(query, hits)
        if log_stages and self.generator:
            logger.info("Query stage: answer generation (%s)", type(self.generator).__name__)
        answer = self.generator.generate(query, forest, self.graph) if self.generator else None
        logger.debug(
            "Query complete: retrieval=%s hits=%d selected=%d cost=%d time_ms=%.1f",
            type(self.forest_retriever).__name__,
            len(hits),
            len(forest.node_ids),
            forest.total_cost,
            (perf_counter() - started) * 1000,
        )
        return PipelineResult(query, hits, forest, answer)
