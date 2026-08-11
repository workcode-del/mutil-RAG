from __future__ import annotations

from collections import defaultdict

from paper_rag.domain import EvidenceForest, EvidenceTree, QuerySpec, SearchHit
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.retrieval.closure import ClosurePolicy
from paper_rag.retrieval.cost import CostModel
from paper_rag.retrieval.ec_bfr import ECBFRConfig
from paper_rag.retrieval.pcst_candidates import build_pcst_candidates, forest_from_trees


class RankedEvidenceRetriever:
    """Top-k evidence baseline with optional graph expansion or PPR reranking."""

    def __init__(
        self,
        graph: EvidenceGraph,
        *,
        top_k: int,
        budget: int,
        image_unit: int,
        hops: int = 0,
        use_ppr: bool = False,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.graph = graph
        self.top_k = top_k
        self.budget = budget
        self.hops = hops
        self.use_ppr = use_ppr
        self.cost_model = CostModel(image_unit)

    def rank_hits(self, query: QuerySpec, hits: list[SearchHit]) -> list[SearchHit]:
        return _ppr_rank(self.graph, hits) if self.use_ppr else hits

    def retrieve(self, query: QuerySpec, hits: list[SearchHit]) -> EvidenceForest:
        selected = {hit.node_id for hit in hits[: self.top_k] if hit.node_id in self.graph.nodes}
        if self.hops:
            selected = self.graph.expand(selected, hops=self.hops)
        scores = {hit.node_id: hit.score for hit in hits}
        by_paper: dict[str, set[str]] = defaultdict(set)
        for node_id in selected:
            by_paper[self.graph.nodes[node_id].paper_id].add(node_id)
        method = "ppr" if self.use_ppr else "one_hop" if self.hops else "top_k"
        trees = [
            EvidenceTree(
                paper_id=paper_id,
                node_ids=node_ids,
                relevance=sum(scores.get(node_id, 0.0) for node_id in node_ids),
                cost=self.cost_model.set_cost(self.graph, node_ids),
                metadata={"retrieval_method": method},
            )
            for paper_id, node_ids in sorted(by_paper.items())
        ]
        return forest_from_trees(trees, self.budget)


class PCSTEvidenceRetriever:
    """Plain PCST baseline, optionally followed by evidence closure but no hard budget selection."""

    def __init__(
        self,
        graph: EvidenceGraph,
        config: ECBFRConfig,
        *,
        apply_closure: bool,
        closure_policy: ClosurePolicy = ClosurePolicy(),
    ) -> None:
        self.graph = graph
        self.config = config
        self.policy = closure_policy if apply_closure else None

    def rank_hits(self, query: QuerySpec, hits: list[SearchHit]) -> list[SearchHit]:
        return hits

    def retrieve(self, query: QuerySpec, hits: list[SearchHit]) -> EvidenceForest:
        candidates = build_pcst_candidates(
            self.graph,
            query,
            hits,
            self.config,
            closure_policy=self.policy,
        )
        if not candidates:
            return EvidenceForest([], 0, self.config.budget)
        best = max(candidates, key=lambda tree: (tree.relevance, -tree.cost))
        best.metadata["retrieval_method"] = "pcst_closure" if self.policy else "pcst"
        return forest_from_trees([best], self.config.budget)


def _ppr_rank(
    graph: EvidenceGraph,
    hits: list[SearchHit],
    *,
    damping: float = 0.85,
    iterations: int = 50,
    tolerance: float = 1e-10,
) -> list[SearchHit]:
    if not hits:
        return hits
    hit_by_id = {hit.node_id: hit for hit in hits}
    node_ids = set(hit_by_id)
    neighbors = {
        node_id: graph.neighbors(node_id) & node_ids
        for node_id in node_ids
        if node_id in graph.nodes
    }
    weights = {node_id: max(hit.score, 0.0) for node_id, hit in hit_by_id.items()}
    total = sum(weights.values())
    personalization = {
        node_id: (weights[node_id] / total if total else 1.0 / len(node_ids))
        for node_id in node_ids
    }
    scores = dict(personalization)
    for _ in range(iterations):
        dangling = sum(scores[node_id] for node_id in node_ids if not neighbors.get(node_id))
        updated = {
            node_id: (1.0 - damping) * personalization[node_id]
            + damping * dangling * personalization[node_id]
            for node_id in node_ids
        }
        for source, targets in neighbors.items():
            if not targets:
                continue
            share = damping * scores[source] / len(targets)
            for target in targets:
                updated[target] += share
        if sum(abs(updated[node_id] - scores[node_id]) for node_id in node_ids) < tolerance:
            scores = updated
            break
        scores = updated
    for node_id, hit in hit_by_id.items():
        hit.score_components["ppr"] = scores[node_id]
        hit.score = scores[node_id]
    return sorted(hits, key=lambda hit: hit.score, reverse=True)
