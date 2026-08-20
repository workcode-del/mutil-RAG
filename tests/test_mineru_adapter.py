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
        [
            {
                "type": "image",
                "img_path": "f1.png",
                "img_caption": ["Figure 1. Curve"],
                "page_idx": 0,
            }
        ],
        "paper",
    )
    assert {edge.relation for edge in parsed.edges} == {RelationType.CAPTION_OF}


def test_adapter_builds_searchable_multimodal_table() -> None:
    parsed = MinerUAdapter().from_blocks(
        [
            {
                "type": "table",
                "img_path": "table.png",
                "table_caption": ["Table 2. Results"],
                "table_body": "<table><tr><th>Model</th><th>F1</th></tr>"
                "<tr><td>Ours</td><td>91.2</td></tr></table>",
                "page_idx": 1,
            },
            {"type": "text", "text": "Table 2 reports the best result.", "page_idx": 1},
        ],
        "paper",
    )

    table = next(node for node in parsed.nodes.values() if node.node_type is NodeType.TABLE)
    assert table.image_path.endswith("table.png")
    assert "Model\tF1" in table.text
    assert "Ours\t91.2" in table.text
    assert table.page == 2
    assert any(
        edge.relation is RelationType.CAPTION_OF and edge.dst == table.node_id
        for edge in parsed.edges
    )
    assert any(
        edge.relation is RelationType.REFERS_TO and edge.dst == table.node_id
        for edge in parsed.edges
    )


def test_adapter_supports_mineru_v2_table_content() -> None:
    parsed = MinerUAdapter().from_blocks(
        [
            {
                "type": "table",
                "content": {
                    "image_source": {"path": "images/t1.jpg"},
                    "table_caption": ["表1 消融实验"],
                    "html": "<table><tr><td>模块</td><td>Recall</td></tr></table>",
                },
                "page_idx": 0,
            }
        ],
        "paper",
    )

    table = next(node for node in parsed.nodes.values() if node.node_type is NodeType.TABLE)
    assert table.attributes["table_label"] == "1"
    assert table.image_path.endswith("images\\t1.jpg") or table.image_path.endswith("images/t1.jpg")
    assert "模块\tRecall" in table.text
