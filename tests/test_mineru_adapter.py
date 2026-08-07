from paper_rag.domain import NodeType, RelationType
from paper_rag.parsing import MinerUAdapter


def test_adapter_builds_cross_modal_relations() -> None:
    blocks = [
        {"type": "image", "img_path": "f1.png", "text": "Fig. 1", "page_idx": 0},
        {"type": "image_caption", "text": "Figure 1. Strength curves.", "page_idx": 0},
        {"type": "text", "text": "As shown in Fig. 1, the strength is improved.", "page_idx": 0},
    ]
    parsed = MinerUAdapter().from_blocks(blocks, "paper")
    assert sum(node.node_type is NodeType.FIGURE for node in parsed.nodes.values()) == 1
    relations = {edge.relation for edge in parsed.edges}
    assert RelationType.CAPTION_OF in relations
    assert RelationType.REFERS_TO in relations
    sentences = [node.text for node in parsed.nodes.values() if node.node_type is NodeType.SENTENCE]
    assert sentences == ["As shown in Fig. 1, the strength is improved."]


def test_adapter_supports_embedded_mineru_caption() -> None:
    parsed = MinerUAdapter().from_blocks(
        [{"type": "image", "img_path": "f1.png", "img_caption": ["Figure 1. Curve"], "page_idx": 0}],
        "paper",
    )
    assert {edge.relation for edge in parsed.edges} == {RelationType.CAPTION_OF}
