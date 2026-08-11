from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from paper_rag.domain import EvidenceForest, EvidenceTree, QuerySpec, SearchHit
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.retrieval.closure import ClosurePolicy, evidence_closure
from paper_rag.retrieval.cost import CostModel
from paper_rag.retrieval.pcst import DEFAULT_RELATION_COSTS, solve_pcst


class PCSTCandidateConfig(Protocol):
    budget: int
    image_unit: int
    candidate_hops: int
    min_edge_confidence: float
    lambda_values: tuple[float, ...]
    relation_costs: dict | None


def build_pcst_candidates(
    graph: EvidenceGraph,
    query: QuerySpec,
    hits: list[SearchHit],
    config: PCSTCandidateConfig,
    *,
    closure_policy: ClosurePolicy | None = None,
    max_cost: int | None = None,
) -> list[EvidenceTree]:
    """Build per-paper PCST candidates once for both baselines and EC-BFR."""
    prizes = {hit.node_id: max(0.0, hit.score) for hit in hits}
    seeds_by_paper: dict[str, set[str]] = defaultdict(set)
    for hit in hits:
        if hit.node_id in graph.nodes:
            seeds_by_paper[hit.paper_id].add(hit.node_id)

    cost_model = CostModel(config.image_unit)
    candidates: list[EvidenceTree] = []
    seen: set[frozenset[str]] = set()
    for paper_id, seed_ids in seeds_by_paper.items():
        expanded = graph.expand(
            seed_ids,
            hops=config.candidate_hops,
            min_confidence=config.min_edge_confidence,
        )
        paper_graph = graph.paper_subgraph(paper_id, expanded)
        paper_prizes = {node_id: prizes.get(node_id, 0.0) for node_id in paper_graph.nodes}
        for scale in config.lambda_values:
            skeleton = solve_pcst(
                paper_graph,
                paper_prizes,
                config.relation_costs or DEFAULT_RELATION_COSTS,
                cost_scale=scale,
            )
            if not skeleton.node_ids:
                continue
            selected = (
                evidence_closure(graph, skeleton.node_ids, closure_policy)
                if closure_policy
                else set(skeleton.node_ids)
            )
            identity = frozenset(selected)
            if identity in seen:
                continue
            seen.add(identity)
            cost = cost_model.set_cost(graph, selected)
            if max_cost is not None and cost > max_cost:
                continue
            candidates.append(
                EvidenceTree(
                    paper_id=paper_id,
                    node_ids=selected,
                    edge_ids=skeleton.edge_pairs,
                    relevance=sum(prizes.get(node_id, 0.0) for node_id in selected),
                    covered_slots=covered_slots(graph, query, selected),
                    entities=node_entities(graph, selected),
                    cost=cost,
                    metadata={"skeleton_backend": skeleton.backend, "lambda": scale},
                )
            )
    return candidates


def forest_from_trees(trees: list[EvidenceTree], budget: int) -> EvidenceForest:
    return EvidenceForest(trees, sum(tree.cost for tree in trees), budget)


def covered_slots(graph: EvidenceGraph, query: QuerySpec, node_ids: set[str]) -> set[str]:
    text = " ".join(graph.nodes[node_id].searchable_text.lower() for node_id in node_ids)
    covered = {"answer"} if text.strip() else set()
    for slot in query.required_slots - {"answer"}:
        value = getattr(query, slot, None)
        if value is not None and str(value).lower() in text:
            covered.add(slot)
    if query.conditions and any(condition.lower() in text for condition in query.conditions):
        covered.add("conditions")
    return covered


def node_entities(graph: EvidenceGraph, node_ids: set[str]) -> set[str]:
    entities: set[str] = set()
    for node_id in node_ids:
        entities.update(str(value) for value in graph.nodes[node_id].attributes.get("entities", []))
    return entities
