from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from paper_rag.domain import EvidenceForest, EvidenceTree, QuerySpec, SearchHit
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.retrieval.closure import ClosurePolicy, evidence_closure
from paper_rag.retrieval.cost import CostModel
from paper_rag.retrieval.pcst import DEFAULT_RELATION_COSTS, solve_pcst


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

    def retrieve(self, query: QuerySpec, hits: list[SearchHit]) -> EvidenceForest:
        prizes = {hit.node_id: max(0.0, hit.score) for hit in hits}
        seeds_by_paper: dict[str, set[str]] = defaultdict(set)
        for hit in hits:
            if hit.node_id in self.graph.nodes:
                seeds_by_paper[hit.paper_id].add(hit.node_id)

        candidates: list[EvidenceTree] = []
        seen_closed_sets: set[frozenset[str]] = set()
        for paper_id, seed_ids in seeds_by_paper.items():
            expanded = self.graph.expand(
                seed_ids,
                hops=self.config.candidate_hops,
                min_confidence=self.config.min_edge_confidence,
            )
            paper_graph = self.graph.paper_subgraph(paper_id, expanded)
            paper_prizes = {node_id: prizes.get(node_id, 0.0) for node_id in paper_graph.nodes}
            for scale in self.config.lambda_values:
                skeleton = solve_pcst(
                    paper_graph,
                    paper_prizes,
                    self.config.relation_costs or DEFAULT_RELATION_COSTS,
                    cost_scale=scale,
                )
                if not skeleton.node_ids:
                    continue
                closed = evidence_closure(self.graph, skeleton.node_ids, self.policy)
                identity = frozenset(closed)
                if identity in seen_closed_sets:
                    continue
                seen_closed_sets.add(identity)
                cost = self.cost_model.set_cost(self.graph, closed)
                if cost > self.config.budget:
                    continue
                candidates.append(
                    EvidenceTree(
                        paper_id=paper_id,
                        node_ids=closed,
                        edge_ids=skeleton.edge_pairs,
                        relevance=sum(prizes.get(node_id, 0.0) for node_id in closed),
                        covered_slots=self._covered_slots(query, closed),
                        entities=self._entities(closed),
                        cost=cost,
                        metadata={"skeleton_backend": skeleton.backend, "lambda": scale},
                    )
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

    def _covered_slots(self, query: QuerySpec, node_ids: set[str]) -> set[str]:
        text = " ".join(self.graph.nodes[node_id].searchable_text.lower() for node_id in node_ids)
        covered = {"answer"} if text.strip() else set()
        for slot in query.required_slots - {"answer"}:
            value = getattr(query, slot, None)
            if value is not None and str(value).lower() in text:
                covered.add(slot)
        if query.conditions and any(condition.lower() in text for condition in query.conditions):
            covered.add("conditions")
        return covered

    def _entities(self, node_ids: set[str]) -> set[str]:
        entities: set[str] = set()
        for node_id in node_ids:
            values = self.graph.nodes[node_id].attributes.get("entities", [])
            entities.update(str(value) for value in values)
        return entities
