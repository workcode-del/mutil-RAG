from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from paper_rag.domain import BoundingBox, EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.parsing.figure_reference import extract_figure_labels, normalize_figure_label
from paper_rag.parsing.sentence_splitter import split_sentences


@dataclass(slots=True)
class ParsedPaper:
    paper_id: str
    nodes: dict[str, EvidenceNode] = field(default_factory=dict)
    edges: list[EvidenceEdge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class MinerUAdapter:
    """Version-isolation layer for MinerU JSON-like content lists.

    MinerU releases may rename fields. Only this adapter contains tolerant field lookup;
    downstream modules consume stable EvidenceNode/EvidenceEdge objects.
    """

    TEXT_TYPES = {"text", "paragraph", "para", "title"}
    IMAGE_TYPES = {"image", "figure", "img"}
    CAPTION_TYPES = {"image_caption", "figure_caption", "caption"}

    def from_json(self, path: str | Path, paper_id: str | None = None) -> ParsedPaper:
        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        resolved_id = paper_id or source.stem.replace("_content_list", "")
        blocks = payload if isinstance(payload, list) else self._find_blocks(payload)
        return self.from_blocks(blocks, resolved_id, source.parent)

    def from_blocks(
        self, blocks: Iterable[dict[str, Any]], paper_id: str, base_dir: str | Path = "."
    ) -> ParsedPaper:
        result = ParsedPaper(paper_id=paper_id)
        figure_by_label: dict[str, str] = {}
        caption_by_label: dict[str, str] = {}
        sentence_nodes: list[EvidenceNode] = []
        base = Path(base_dir)

        for index, block in enumerate(blocks):
            block_type = str(block.get("type") or block.get("category") or "").lower()
            page = self._page(block)
            bbox = self._bbox(block)
            block_id = str(block.get("id") or f"block:{index}")
            text = str(block.get("text") or block.get("content") or "").strip()

            if block_type in self.TEXT_TYPES and text:
                previous: EvidenceNode | None = sentence_nodes[-1] if sentence_nodes else None
                for sentence_index, sentence in enumerate(split_sentences(text)):
                    node_id = f"{paper_id}:sentence:{index}:{sentence_index}"
                    node = EvidenceNode(
                        node_id=node_id,
                        paper_id=paper_id,
                        node_type=NodeType.SENTENCE,
                        text=sentence,
                        page=page,
                        bbox=bbox,
                        parser_block_id=block_id,
                        provenance={"parser": "mineru", "location_level": "block"},
                        attributes={"source_index": index},
                    )
                    result.nodes[node_id] = node
                    sentence_nodes.append(node)
                    if previous is not None:
                        result.edges.append(
                            EvidenceEdge(previous.node_id, node_id, RelationType.NEXT_SENTENCE)
                        )
                    previous = node
            elif block_type in self.IMAGE_TYPES:
                raw_path = block.get("img_path") or block.get("image_path") or block.get("path")
                if not raw_path:
                    result.warnings.append(f"Image block {block_id} has no path")
                    continue
                image_path = Path(str(raw_path))
                if not image_path.is_absolute():
                    image_path = base / image_path
                node_id = f"{paper_id}:figure:{index}"
                node = EvidenceNode(
                    node_id=node_id,
                    paper_id=paper_id,
                    node_type=NodeType.FIGURE,
                    image_path=str(image_path),
                    page=page,
                    bbox=bbox,
                    parser_block_id=block_id,
                    provenance={"parser": "mineru"},
                    attributes={"source_index": index},
                )
                result.nodes[node_id] = node
                embedded_caption = block.get("img_caption") or block.get("caption")
                if isinstance(embedded_caption, list):
                    embedded_caption = " ".join(str(value) for value in embedded_caption)
                embedded_caption = str(embedded_caption or "").strip()
                label_source = embedded_caption or text
                label = normalize_figure_label(label_source) if label_source else None
                if label:
                    figure_by_label[label] = node_id
                if embedded_caption:
                    caption_id = f"{paper_id}:caption:{index}:embedded"
                    result.nodes[caption_id] = EvidenceNode(
                        node_id=caption_id,
                        paper_id=paper_id,
                        node_type=NodeType.CAPTION,
                        text=embedded_caption,
                        page=page,
                        bbox=bbox,
                        parser_block_id=block_id,
                        provenance={"parser": "mineru", "embedded_in_image": True},
                        attributes={"source_index": index},
                    )
                    result.edges.append(
                        EvidenceEdge(
                            caption_id,
                            node_id,
                            RelationType.CAPTION_OF,
                            mandatory_for_closure=True,
                        )
                    )
                    node.attributes["caption_id"] = caption_id
            elif block_type in self.CAPTION_TYPES and text:
                node_id = f"{paper_id}:caption:{index}"
                node = EvidenceNode(
                    node_id=node_id,
                    paper_id=paper_id,
                    node_type=NodeType.CAPTION,
                    text=text,
                    page=page,
                    bbox=bbox,
                    parser_block_id=block_id,
                    provenance={"parser": "mineru"},
                    attributes={"source_index": index},
                )
                result.nodes[node_id] = node
                label = normalize_figure_label(text)
                if label:
                    caption_by_label[label] = node_id

        for label, caption_id in caption_by_label.items():
            figure_id = figure_by_label.get(label)
            if figure_id:
                result.edges.append(
                    EvidenceEdge(
                        caption_id,
                        figure_id,
                        RelationType.CAPTION_OF,
                        mandatory_for_closure=True,
                    )
                )
                result.nodes[figure_id].attributes["caption_id"] = caption_id
                result.nodes[figure_id].attributes["figure_label"] = label

        # Some MinerU backends keep image and caption as adjacent blocks but do not put a
        # Figure label in the image block. Pair remaining items by page and reading order.
        linked_figures = {
            edge.dst for edge in result.edges if edge.relation is RelationType.CAPTION_OF
        }
        linked_captions = {
            edge.src for edge in result.edges if edge.relation is RelationType.CAPTION_OF
        }
        orphan_captions = [
            node
            for node in result.nodes.values()
            if node.node_type is NodeType.CAPTION and node.node_id not in linked_captions
        ]
        for caption in orphan_captions:
            candidates = [
                node
                for node in result.nodes.values()
                if node.node_type is NodeType.FIGURE
                and node.node_id not in linked_figures
                and node.page == caption.page
            ]
            if not candidates:
                continue
            source_index = int(caption.attributes.get("source_index", 0))
            figure = min(
                candidates,
                key=lambda node: abs(int(node.attributes.get("source_index", 0)) - source_index),
            )
            result.edges.append(
                EvidenceEdge(
                    caption.node_id,
                    figure.node_id,
                    RelationType.CAPTION_OF,
                    confidence=0.9,
                    mandatory_for_closure=True,
                    attributes={"alignment": "same_page_nearest_block"},
                )
            )
            linked_figures.add(figure.node_id)
            figure.attributes["caption_id"] = caption.node_id

        for sentence in sentence_nodes:
            for label in extract_figure_labels(sentence.text or ""):
                figure_id = figure_by_label.get(label)
                if figure_id:
                    result.edges.append(
                        EvidenceEdge(
                            sentence.node_id,
                            figure_id,
                            RelationType.REFERS_TO,
                            mandatory_for_closure=True,
                        )
                    )
        return result

    @staticmethod
    def _find_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("content_list", "blocks", "items", "content"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        raise ValueError("Could not find a MinerU content block list")

    @staticmethod
    def _page(block: dict[str, Any]) -> int | None:
        raw = block.get("page_idx", block.get("page", block.get("page_number")))
        if raw is None:
            return None
        number = int(raw)
        return number + 1 if "page_idx" in block else number

    @staticmethod
    def _bbox(block: dict[str, Any]) -> BoundingBox | None:
        raw = block.get("bbox") or block.get("box")
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            return BoundingBox(*(float(value) for value in raw))
        return None
