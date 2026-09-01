from paper_rag.domain import EvidenceNode, NodeType, QuerySpec
from paper_rag.entities import ScientificEntityExtractor, enrich_graph_entities
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.query_understanding import ScientificQueryParser
from paper_rag.retrieval.pcst_candidates import covered_slots, node_entities


def test_query_parser_extracts_scientific_constraints() -> None:
    parsed = ScientificQueryParser().parse(
        QuerySpec("Which alloy has tensile strength above 500 MPa under room temperature?")
    )

    assert parsed.answer_type == "entity_list"
    assert parsed.entity_type == "material"
    assert parsed.metric == "tensile strength"
    assert parsed.operator == "gt"
    assert parsed.value == 500.0
    assert parsed.unit == "MPa"
    assert parsed.conditions == ["room temperature"]
    assert parsed.entities == []
    assert set(parsed.auto_parsed_fields) == {
        "entity_type",
        "metric",
        "operator",
        "value",
        "unit",
        "conditions",
    }


def test_query_parser_supports_chinese_scientific_constraints() -> None:
    parsed = ScientificQueryParser().parse(
        QuerySpec("哪些钛合金材料的拉伸强度超过 500 MPa？")
    )

    assert parsed.answer_type == "entity_list"
    assert parsed.entity_type == "material"
    assert parsed.metric == "tensile strength"
    assert parsed.operator == "gt"
    assert parsed.value == 500.0
    assert parsed.unit == "MPa"


def test_query_parser_preserves_explicit_fields() -> None:
    parsed = ScientificQueryParser().parse(
        QuerySpec(
            "accuracy above 90%",
            metric="custom metric",
            operator="le",
            value=12,
            unit="ms",
        )
    )

    assert (parsed.metric, parsed.operator, parsed.value, parsed.unit) == (
        "custom metric",
        "le",
        12,
        "ms",
    )
    assert not {"metric", "operator", "value", "unit"} & set(parsed.auto_parsed_fields)


def test_scientific_entities_are_typed_and_deduplicated() -> None:
    text = (
        "The ResNet-50 model was evaluated on ImageNet dataset. "
        "Ti-6Al-4V and Fe3O4 materials were tested."
    )

    mentions = ScientificEntityExtractor().extract(text)
    by_name = {mention.normalized: mention.entity_type for mention in mentions}

    assert by_name == {
        "resnet-50": "model",
        "imagenet": "dataset",
        "ti-6al-4v": "material",
        "fe3o4": "material",
    }


def test_entity_enrichment_activates_typed_novelty_and_slot_coverage() -> None:
    graph = EvidenceGraph()
    graph.add_node(
        EvidenceNode(
            "p:s",
            "p",
            NodeType.SENTENCE,
            text="Ti-6Al-4V alloy reaches tensile strength of 550 MPa.",
        )
    )
    enrich_graph_entities(graph)
    query = ScientificQueryParser().parse(
        QuerySpec("Which alloy has tensile strength above 500 MPa?")
    )

    assert node_entities(graph, {"p:s"}, query.entity_type) == {"ti-6al-4v"}
    assert covered_slots(graph, query, {"p:s"}) == {
        "answer",
        "entity_type",
        "metric",
        "operator",
        "value",
        "unit",
    }


def test_entity_enrichment_refreshes_changed_text_without_stale_auto_entities() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:s", "p", NodeType.SENTENCE, text="ResNet-50 model"))
    enrich_graph_entities(graph)

    graph.nodes["p:s"].text = "ImageNet dataset"

    assert enrich_graph_entities(graph) == 1
    assert graph.nodes["p:s"].attributes["entities"] == ["imagenet"]
