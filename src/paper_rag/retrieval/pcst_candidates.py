from __future__ import annotations

from collections import defaultdict
import re
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
    selection_score_source: str
    selection_threshold: float
    compact_trees: bool


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
    # Plain PCST baselines keep their historical rewards. Only the budgeted
    # selector uses thresholded raw scores, independent of ranking fusion.
    source = config.selection_score_source if max_cost is not None else "fusion"
    if source != "fusion" and not any(source in hit.score_components for hit in hits):
        source = "embedding" if any("embedding" in h.score_components for h in hits) else "fusion"
    prizes = {
        hit.node_id: max(0.0, hit.score) if source == "fusion" else max(
            0.0, hit.score_components.get(source, float("-inf")) - config.selection_threshold
        )
        for hit in hits
    }
    compact = config.compact_trees and max_cost is not None
    seeds_by_paper: dict[str, set[str]] = defaultdict(set)
    for hit in hits:
        if hit.node_id in graph.nodes and prizes[hit.node_id] > 0:
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
        paper_graph = graph.paper_subgraph(
            paper_id,
            expanded,
            min_edge_confidence=config.min_edge_confidence,
        )
        raw_prizes = {node_id: prizes.get(node_id, 0.0) for node_id in paper_graph.nodes}
        peak_prize = max(raw_prizes.values(), default=0.0)
        # RRF scores are around 1e-2 while relation costs are around 1e-1.  Normalize
        # within each paper so PCST optimizes the intended relevance/cost trade-off.
        paper_prizes = {
            node_id: prize / peak_prize if source == "fusion" and peak_prize > 0 else prize
            for node_id, prize in raw_prizes.items()
        }
        for scale in config.lambda_values:
            skeleton = solve_pcst(
                paper_graph,
                paper_prizes,
                config.relation_costs or DEFAULT_RELATION_COSTS,
                cost_scale=scale,
            )
            if not skeleton.node_ids:
                continue
            roots = {node_id for node_id in skeleton.node_ids if prizes.get(node_id, 0) > 0}
            selected = (
                evidence_closure(graph, skeleton.node_ids, closure_policy)
                if closure_policy
                else set(skeleton.node_ids)
            )
            if compact:
                selected, roots = compact_subtree(
                    graph, skeleton.node_ids, skeleton.edge_pairs, roots, prizes,
                    cost_model, max_cost, closure_policy,
                )
            if not selected:
                continue
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
                    edge_ids={
                        (src, dst) for src, dst in skeleton.edge_pairs
                        if src in selected and dst in selected
                    },
                    relevance=sum(
                        prizes.get(node_id, 0.0)
                        for node_id in (roots if compact else selected)
                    ),
                    covered_slots=covered_slots(graph, query, selected),
                    entities=node_entities(graph, selected, query.entity_type),
                    cost=cost,
                    metadata={
                        "skeleton_backend": skeleton.backend, "lambda": scale,
                        "selection_score_source": source,
                        "compact": compact,
                        "primary_node_ids": sorted(roots),
                        "dependency_node_ids": sorted(selected - skeleton.node_ids),
                    },
                )
            )
    return candidates


def compact_subtree(
    graph: EvidenceGraph,
    skeleton: set[str],
    edges: set[tuple[str, str]],
    roots: set[str],
    prizes: dict[str, float],
    cost_model: CostModel,
    budget: int,
    policy: ClosurePolicy | None,
) -> tuple[set[str], set[str]]:
    """Prune optional branches, then remove lowest-value bundles until feasible.

    Connector nodes between retained roots survive pruning. Dependencies
    are recomputed after each proposed deletion, never removed independently.
    """
    adjacency = {node_id: set() for node_id in skeleton}
    for src, dst in edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    def close(keep: set[str]) -> set[str]:
        if not keep:
            return set()
        # PCST returns a tree; BFS also handles cyclic/disconnected fallback graphs.
        parents: dict[str, str | None] = {}
        for root in sorted(keep):
            if root in parents:
                continue
            parents[root] = None
            queue = [root]
            for node_id in queue:
                for other in sorted(adjacency[node_id]):
                    if other not in parents:
                        parents[other] = node_id
                        queue.append(other)
        nodes: set[str] = set()
        for root in keep:
            current = root
            while current is not None and current not in nodes:
                nodes.add(current)
                current = parents[current]
        return evidence_closure(graph, nodes, policy) if policy else nodes

    roots = set(roots)
    selected = close(roots)
    cost = cost_model.set_cost(graph, selected)
    while cost > budget and roots:
        best = None
        for node_id in sorted(roots):
            trial = close(roots - {node_id})
            saved = cost - cost_model.set_cost(graph, trial)
            if saved <= 0:
                continue
            loss = sum(prizes.get(root, 0.0) for root in roots - trial)
            candidate = (loss / saved, node_id, trial, saved)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            # All remaining seeds require each other; no smaller closed bundle.
            return set(), set()
        _, removed, selected, saved = best
        roots.remove(removed)
        roots.intersection_update(selected)
        cost -= saved
    return selected, roots


def forest_from_trees(trees: list[EvidenceTree], budget: int) -> EvidenceForest:
    return EvidenceForest(trees, sum(tree.cost for tree in trees), budget)


def covered_slots(graph: EvidenceGraph, query: QuerySpec, node_ids: set[str]) -> set[str]:
    text = " ".join(graph.nodes[node_id].searchable_text.casefold() for node_id in node_ids)
    covered = {"answer"} if text.strip() else set()
    if query.metric and query.metric.casefold() in text:
        covered.add("metric")
    unit_values = _observed_values(text, query.unit) if query.unit else []
    if query.unit and unit_values:
        covered.add("unit")
    if query.entity_type and any(
        _entity_type_matches(query.entity_type, str(mention.get("entity_type", "")))
        for node_id in node_ids
        for mention in graph.nodes[node_id].attributes.get("entity_mentions", [])
        if isinstance(mention, dict)
    ):
        covered.add("entity_type")
    if query.value is not None:
        values = unit_values if query.unit else _observed_values(text, None)
        if any(_satisfies(value, query.value, query.operator) for value in values):
            covered.add("value")
            if query.operator:
                covered.add("operator")
    elif query.operator and _operator_is_explicit(query.operator, text):
        covered.add("operator")
    if query.conditions and any(
        _condition_matches(condition, text) for condition in query.conditions
    ):
        covered.add("conditions")
    return covered


def node_entities(
    graph: EvidenceGraph,
    node_ids: set[str],
    entity_type: str | None = None,
) -> set[str]:
    entities: set[str] = set()
    for node_id in node_ids:
        node = graph.nodes[node_id]
        mentions = node.attributes.get("entity_mentions", [])
        if entity_type and mentions:
            entities.update(
                str(mention["normalized"])
                for mention in mentions
                if isinstance(mention, dict)
                and mention.get("normalized")
                and _entity_type_matches(entity_type, str(mention.get("entity_type", "")))
            )
        elif not entity_type:
            entities.update(str(value) for value in node.attributes.get("entities", []))
    return entities


def _satisfies(observed: float, expected: float, operator: str | None) -> bool:
    tolerance = max(abs(expected) * 0.05, 1e-6)
    if operator == "gt":
        return observed > expected
    if operator == "ge":
        return observed >= expected
    if operator == "lt":
        return observed < expected
    if operator == "le":
        return observed <= expected
    return abs(observed - expected) <= tolerance


def _normalize_unit(value: str) -> str:
    return value.casefold().replace(" ", "").replace("℃", "°c")


def _observed_values(text: str, unit: str | None) -> list[float]:
    if unit:
        normalized_unit = _normalize_unit(unit)
        aliases = {normalized_unit}
        if normalized_unit == "°c":
            aliases.add("℃")
        pattern = re.compile(
            r"([+-]?\d+(?:\.\d+)?)\s*(?:"
            + "|".join(re.escape(value) for value in sorted(aliases, key=len, reverse=True))
            + r")",
            re.IGNORECASE,
        )
        return [float(match.group(1)) for match in pattern.finditer(_normalize_unit(text))]
    return [float(value) for value in re.findall(r"[+-]?\d+(?:\.\d+)?", text)]


def _operator_is_explicit(operator: str, text: str) -> bool:
    aliases = {
        "gt": (">", "above", "greater than", "超过", "高于"),
        "ge": (">=", "≥", "at least", "至少", "不低于"),
        "lt": ("<", "below", "less than", "低于"),
        "le": ("<=", "≤", "at most", "不超过"),
        "eq": ("=", "equal", "等于"),
        "approx": ("approximately", "about", "约"),
    }
    return any(alias in text for alias in aliases.get(operator, ()))


def _condition_matches(condition: str, text: str) -> bool:
    normalized = " ".join(condition.casefold().split())
    if normalized in text:
        return True
    tokens = {token for token in re.findall(r"[\w.+%-]+", normalized) if len(token) > 1}
    return bool(tokens) and len(tokens & set(re.findall(r"[\w.+%-]+", text))) / len(tokens) >= 0.6


def _entity_type_matches(required: str, observed: str) -> bool:
    if required == observed:
        return True
    return required == "method_or_model" and observed in {"method", "model"}
