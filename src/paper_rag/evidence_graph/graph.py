from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

from paper_rag.domain import EvidenceEdge, EvidenceNode, RelationType


@dataclass(slots=True)
class EvidenceGraph:
    nodes: dict[str, EvidenceNode] = field(default_factory=dict)
    edges: list[EvidenceEdge] = field(default_factory=list)
    _outgoing: dict[str, list[EvidenceEdge]] = field(default_factory=lambda: defaultdict(list))
    _incoming: dict[str, list[EvidenceEdge]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node: EvidenceNode) -> None:
        existing = self.nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"Conflicting node id: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: EvidenceEdge) -> None:
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            raise KeyError(f"Edge endpoints must exist: {edge.src} -> {edge.dst}")
        self.edges.append(edge)
        self._outgoing[edge.src].append(edge)
        self._incoming[edge.dst].append(edge)

    def extend(self, nodes: Iterable[EvidenceNode], edges: Iterable[EvidenceEdge]) -> None:
        for node in nodes:
            self.add_node(node)
        for edge in edges:
            self.add_edge(edge)

    def incident_edges(self, node_id: str) -> list[EvidenceEdge]:
        return [*self._outgoing.get(node_id, []), *self._incoming.get(node_id, [])]

    def neighbors(
        self, node_id: str, relations: set[RelationType] | None = None
    ) -> set[str]:
        found: set[str] = set()
        for edge in self.incident_edges(node_id):
            if relations is not None and edge.relation not in relations:
                continue
            found.add(edge.dst if edge.src == node_id else edge.src)
        return found

    def expand(self, seed_ids: set[str], hops: int, min_confidence: float = 0.8) -> set[str]:
        visited = set(seed_ids)
        queue = deque((node_id, 0) for node_id in seed_ids)
        while queue:
            current, depth = queue.popleft()
            if depth >= hops:
                continue
            for edge in self.incident_edges(current):
                if edge.confidence < min_confidence:
                    continue
                neighbor = edge.dst if edge.src == current else edge.src
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return visited

    def paper_subgraph(self, paper_id: str, node_ids: set[str] | None = None) -> "EvidenceGraph":
        selected = {
            node_id
            for node_id, node in self.nodes.items()
            if node.paper_id == paper_id and (node_ids is None or node_id in node_ids)
        }
        graph = EvidenceGraph()
        for node_id in selected:
            graph.add_node(self.nodes[node_id])
        for edge in self.edges:
            if edge.src in selected and edge.dst in selected:
                graph.add_edge(edge)
        return graph
