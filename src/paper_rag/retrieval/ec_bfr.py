from __future__ import annotations

from dataclasses import dataclass, field

from paper_rag.domain import EvidenceForest, EvidenceTree, QuerySpec, SearchHit
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.retrieval.closure import ClosurePolicy
from paper_rag.retrieval.cost import CostModel
from paper_rag.retrieval.pcst import DEFAULT_RELATION_COSTS
from paper_rag.retrieval.pcst_candidates import build_pcst_candidates


@dataclass(frozen=True, slots=True)
class ECBFRConfig:
    budget: int = 4096
    image_unit: int = 512
    candidate_hops: int = 2
    min_edge_confidence: float = 0.8
    lambda_values: tuple[float, ...] = (0.5, 1.0, 2.0)
    slot_weight: float = 0.4
    entity_weight: float = 0.3
    redundancy_weight: float = 0.2
    relation_costs: dict | None = field(default_factory=lambda: dict(DEFAULT_RELATION_COSTS))


class EvidenceClosureBudgetedForestRetriever:
    """EC-BFR: PCST skeletons followed by closure-safe global budget selection."""

    def __init__(
        self,
        graph: EvidenceGraph,
        config: ECBFRConfig = ECBFRConfig(),
        closure_policy: ClosurePolicy = ClosurePolicy(),
    ) -> None:
        self.graph = graph
        self.config = config
        self.policy = closure_policy
        self.cost_model = CostModel(config.image_unit)

    def rank_hits(self, query: QuerySpec, hits: list[SearchHit]) -> list[SearchHit]:
        return hits

    def retrieve(self, query: QuerySpec, hits: list[SearchHit]) -> EvidenceForest:
        candidates = build_pcst_candidates(
            self.graph,
            query,
            hits,
            self.config,
            closure_policy=self.policy,
            max_cost=self.config.budget,
        )
        return self._select_forest(query, candidates)

    def _select_forest(
        self, query: QuerySpec, candidates: list[EvidenceTree]
    ) -> EvidenceForest:
        selected: list[EvidenceTree] = []
        selected_nodes: set[str] = set()
        covered_slots: set[str] = set()
        entities: set[str] = set()
        total_cost = 0
        remaining = list(candidates)

        while remaining:
            best: tuple[float, EvidenceTree, int] | None = None
            for candidate in remaining:
                new_nodes = candidate.node_ids - selected_nodes
                marginal_cost = self.cost_model.set_cost(self.graph, new_nodes)
                if total_cost + marginal_cost > self.config.budget or marginal_cost <= 0:
                    continue
                new_slots = candidate.covered_slots - covered_slots
                new_entities = candidate.entities - entities
                overlap = len(candidate.node_ids & selected_nodes) / max(len(candidate.node_ids), 1)
                gain = (
                    candidate.relevance
                    + self.config.slot_weight * len(new_slots) / max(len(query.required_slots), 1)
                    + self.config.entity_weight * len(new_entities)
                    - self.config.redundancy_weight * overlap
                )
                utility = gain / marginal_cost
                if gain > 0 and (best is None or utility > best[0]):
                    best = (utility, candidate, marginal_cost)
            if best is None:
                break
            _, chosen, marginal_cost = best
            selected.append(chosen)
            selected_nodes.update(chosen.node_ids)
            covered_slots.update(chosen.covered_slots)
            entities.update(chosen.entities)
            total_cost += marginal_cost
            remaining.remove(chosen)

        forest = EvidenceForest(selected, total_cost, self.config.budget)
        forest.validate_budget()
        return forest
