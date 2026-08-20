from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    PAPER = "Paper"
    SECTION = "Section"
    PARAGRAPH = "Paragraph"
    SENTENCE = "Sentence"
    FIGURE = "Figure"
    TABLE = "Table"
    CAPTION = "Caption"
    CHART_DATA = "ChartData"


class RelationType(StrEnum):
    CONTAINS = "contains"
    CAPTION_OF = "caption_of"
    REFERS_TO = "refers_to"
    DERIVED_FROM = "derived_from"
    NEXT_SENTENCE = "next_sentence"
    SEMANTICALLY_SIMILAR = "semantically_similar"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Bounding box must satisfy x0 <= x1 and y0 <= y1")

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(slots=True)
class EvidenceNode:
    node_id: str
    paper_id: str
    node_type: NodeType
    text: str | None = None
    image_path: str | None = None
    page: int | None = None
    bbox: BoundingBox | None = None
    parser_block_id: str | None = None
    confidence: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id or not self.paper_id:
            raise ValueError("node_id and paper_id are required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.node_type is NodeType.FIGURE and not self.image_path:
            raise ValueError("Figure node requires image_path")
        if self.node_type is NodeType.TABLE and not (self.text or self.image_path):
            raise ValueError("Table node requires text and/or image_path")
        if self.node_type is not NodeType.FIGURE and self.text is None:
            self.text = ""

    @property
    def searchable_text(self) -> str:
        return self.text or str(self.attributes.get("text_view", ""))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["node_type"] = self.node_type.value
        if self.bbox:
            data["bbox"] = self.bbox.as_list()
        return data


@dataclass(frozen=True, slots=True)
class EvidenceEdge:
    src: str
    dst: str
    relation: RelationType
    confidence: float = 1.0
    mandatory_for_closure: bool = False
    directional: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.src == self.dst and self.relation is not RelationType.NEXT_SENTENCE:
            raise ValueError("Unexpected self edge")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(slots=True)
class QuerySpec:
    query: str
    answer_type: str = "free_text"
    entity_type: str | None = None
    metric: str | None = None
    operator: str | None = None
    value: float | None = None
    unit: str | None = None
    conditions: list[str] = field(default_factory=list)
    required_modalities: list[str] = field(
        default_factory=lambda: ["text", "figure", "table"]
    )

    @property
    def required_slots(self) -> set[str]:
        slots = {"answer"}
        for name in ("entity_type", "metric", "operator", "value", "unit"):
            if getattr(self, name) is not None:
                slots.add(name)
        if self.conditions:
            slots.add("conditions")
        return slots


@dataclass(slots=True)
class SearchHit:
    node_id: str
    paper_id: str
    node_type: NodeType
    score: float
    score_components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceTree:
    paper_id: str
    node_ids: set[str]
    edge_ids: set[tuple[str, str]] = field(default_factory=set)
    relevance: float = 0.0
    covered_slots: set[str] = field(default_factory=set)
    entities: set[str] = field(default_factory=set)
    cost: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceForest:
    trees: list[EvidenceTree] = field(default_factory=list)
    total_cost: int = 0
    budget: int = 0

    @property
    def node_ids(self) -> set[str]:
        return set().union(*(tree.node_ids for tree in self.trees)) if self.trees else set()

    def validate_budget(self) -> None:
        if self.total_cost > self.budget:
            raise ValueError(f"Forest cost {self.total_cost} exceeds budget {self.budget}")

