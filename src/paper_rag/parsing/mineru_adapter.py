from __future__ import annotations

import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from paper_rag.domain import BoundingBox, EvidenceEdge, EvidenceNode, NodeType, RelationType
from paper_rag.parsing.figure_reference import extract_figure_labels, normalize_figure_label
from paper_rag.parsing.sentence_splitter import split_sentences
from paper_rag.parsing.table_reference import extract_table_labels, normalize_table_label


class _TableHTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"tr", "br"}:
            self.parts.append("\n")
        elif tag in {"td", "th"} and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\t")

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.parts.append(value)

    def text(self) -> str:
        rows = ["\t".join(cell.strip() for cell in row.split("\t") if cell.strip())
                for row in "".join(self.parts).splitlines()]
        return "\n".join(row for row in rows if row)


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
    TABLE_TYPES = {"table", "table_body"}
    IMAGE_CAPTION_TYPES = {"image_caption", "figure_caption", "caption"}
    TABLE_CAPTION_TYPES = {"table_caption"}

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
        table_by_label: dict[str, str] = {}
        figure_caption_by_label: dict[str, str] = {}
        table_caption_by_label: dict[str, str] = {}
        sentence_nodes: list[EvidenceNode] = []
        base = Path(base_dir)

        for index, block in enumerate(blocks):
            block_type = str(block.get("type") or block.get("category") or "").lower()
            page = self._page(block)
            bbox = self._bbox(block)
            block_id = str(block.get("id") or f"block:{index}")
            text = self._block_text(block)

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
                raw_path = self._image_path(block)
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
            elif block_type in self.TABLE_TYPES:
                body, caption, footnote, raw_path = self._table_fields(block)
                if not body and not raw_path:
                    result.warnings.append(f"Table block {block_id} has no body or image")
                    continue
                image_path = self._resolve_path(raw_path, base) if raw_path else None
                node_id = f"{paper_id}:table:{index}"
                node = EvidenceNode(
                    node_id=node_id,
                    paper_id=paper_id,
                    node_type=NodeType.TABLE,
                    text="\n".join(value for value in (caption, body, footnote) if value),
                    image_path=image_path,
                    page=page,
                    bbox=bbox,
                    parser_block_id=block_id,
                    provenance={"parser": "mineru"},
                    attributes={"source_index": index, "table_html": self._table_html(block)},
                )
                result.nodes[node_id] = node
                label = normalize_table_label(caption)
                if label:
                    table_by_label[label] = node_id
                    node.attributes["table_label"] = label
                if caption:
                    caption_id = f"{paper_id}:caption:{index}:table"
                    result.nodes[caption_id] = EvidenceNode(
                        caption_id,
                        paper_id,
                        NodeType.CAPTION,
                        text=caption,
                        page=page,
                        bbox=bbox,
                        parser_block_id=block_id,
                        provenance={"parser": "mineru", "embedded_in_table": True},
                        attributes={"source_index": index, "caption_target": "table"},
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
            elif block_type in self.IMAGE_CAPTION_TYPES | self.TABLE_CAPTION_TYPES and text:
                node_id = f"{paper_id}:caption:{index}"
                target = "table" if block_type in self.TABLE_CAPTION_TYPES else "figure"
                node = EvidenceNode(
                    node_id=node_id,
                    paper_id=paper_id,
                    node_type=NodeType.CAPTION,
                    text=text,
                    page=page,
                    bbox=bbox,
                    parser_block_id=block_id,
                    provenance={"parser": "mineru"},
                    attributes={"source_index": index, "caption_target": target},
                )
                result.nodes[node_id] = node
                label = (
                    normalize_table_label(text)
                    if target == "table"
                    else normalize_figure_label(text)
                )
                if label:
                    captions = (
                        table_caption_by_label if target == "table" else figure_caption_by_label
                    )
                    captions[label] = node_id

        for captions, targets, label_key in (
            (figure_caption_by_label, figure_by_label, "figure_label"),
            (table_caption_by_label, table_by_label, "table_label"),
        ):
            for label, caption_id in captions.items():
                target_id = targets.get(label)
                if not target_id:
                    continue
                result.edges.append(
                    EvidenceEdge(
                        caption_id,
                        target_id,
                        RelationType.CAPTION_OF,
                        mandatory_for_closure=True,
                    )
                )
                result.nodes[target_id].attributes["caption_id"] = caption_id
                result.nodes[target_id].attributes[label_key] = label

        # Some MinerU backends keep image and caption as adjacent blocks but do not put a
        # Figure label in the image block. Pair remaining items by page and reading order.
        linked_targets = {
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
            target_type = (
                NodeType.TABLE
                if caption.attributes.get("caption_target") == "table"
                else NodeType.FIGURE
            )
            candidates = [
                node
                for node in result.nodes.values()
                if node.node_type is target_type
                and node.node_id not in linked_targets
                and node.page == caption.page
            ]
            if not candidates:
                continue
            source_index = int(caption.attributes.get("source_index", 0))
            target = min(
                candidates,
                key=lambda node: abs(int(node.attributes.get("source_index", 0)) - source_index),
            )
            result.edges.append(
                EvidenceEdge(
                    caption.node_id,
                    target.node_id,
                    RelationType.CAPTION_OF,
                    confidence=0.9,
                    mandatory_for_closure=True,
                    attributes={"alignment": "same_page_nearest_block"},
                )
            )
            linked_targets.add(target.node_id)
            target.attributes["caption_id"] = caption.node_id

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
            for label in extract_table_labels(sentence.text or ""):
                table_id = table_by_label.get(label)
                if table_id:
                    result.edges.append(
                        EvidenceEdge(
                            sentence.node_id,
                            table_id,
                            RelationType.REFERS_TO,
                            mandatory_for_closure=True,
                        )
                    )
        return result

    @staticmethod
    def _block_text(block: dict[str, Any]) -> str:
        value = block.get("text")
        if value is None and isinstance(block.get("content"), str):
            value = block["content"]
        return " ".join(str(value or "").split())

    @staticmethod
    def _image_path(block: dict[str, Any]) -> Any:
        content = block.get("content") if isinstance(block.get("content"), dict) else {}
        source = (
            content.get("image_source")
            if isinstance(content.get("image_source"), dict)
            else {}
        )
        return (
            block.get("img_path")
            or block.get("image_path")
            or block.get("path")
            or content.get("img_path")
            or source.get("path")
        )

    @classmethod
    def _table_fields(cls, block: dict[str, Any]) -> tuple[str, str, str, Any]:
        content = block.get("content") if isinstance(block.get("content"), dict) else {}
        html = str(block.get("table_body") or block.get("html") or content.get("html") or "")
        markdown = str(block.get("markdown") or content.get("markdown") or "")
        body = cls._html_text(html) if html else markdown.strip()
        return (
            body,
            cls._join_text(block.get("table_caption") or content.get("table_caption")),
            cls._join_text(block.get("table_footnote") or content.get("table_footnote")),
            cls._image_path(block),
        )

    @staticmethod
    def _join_text(value: Any) -> str:
        if isinstance(value, list):
            return " ".join(str(item) for item in value if item)
        return str(value or "").strip()

    @staticmethod
    def _html_text(value: str) -> str:
        parser = _TableHTMLText()
        parser.feed(value)
        return parser.text()

    @staticmethod
    def _table_html(block: dict[str, Any]) -> str:
        content = block.get("content") if isinstance(block.get("content"), dict) else {}
        return str(block.get("table_body") or block.get("html") or content.get("html") or "")

    @staticmethod
    def _resolve_path(raw_path: Any, base: Path) -> str:
        path = Path(str(raw_path))
        return str(path if path.is_absolute() else base / path)

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
