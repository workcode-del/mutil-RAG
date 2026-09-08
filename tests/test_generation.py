import sys
from types import SimpleNamespace
from unittest.mock import patch

from paper_rag.domain import (
    EvidenceForest,
    EvidenceNode,
    EvidenceTree,
    NodeType,
    QuerySpec,
)
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.generation.openai_compatible import OpenAICompatibleGenerator, _response_text


def test_openai_generator_uses_plain_chat_completion_content() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("n", "p", NodeType.SENTENCE, text="evidence"))
    forest = EvidenceForest([EvidenceTree("p", {"n"})])
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "plain answer"}}]}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    requests = SimpleNamespace(post=post)
    generator = OpenAICompatibleGenerator(
        "https://example.test/v1",
        "model",
        extra_body={"enable_thinking": False},
    )
    with patch.dict(sys.modules, {"requests": requests}):
        answer = generator.generate(QuerySpec("question"), forest, graph)

    assert answer.text == "plain answer"
    assert answer.evidence_ids is None
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["json"]["enable_thinking"] is False
    assert "response_format" not in captured["json"]


def test_openai_response_text_accepts_text_parts() -> None:
    response = {
        "choices": [
            {"message": {"content": [{"type": "text", "text": "answer"}]}}
        ]
    }

    assert _response_text(response) == "answer"


def test_openai_response_text_falls_back_to_provider_reasoning_fields() -> None:
    reasoning_content = {
        "choices": [{"message": {"content": None, "reasoning_content": "answer one"}}]
    }
    reasoning = {
        "choices": [{"message": {"content": "", "reasoning": {"text": "answer two"}}}]
    }

    assert _response_text(reasoning_content) == "answer one"
    assert _response_text(reasoning) == "answer two"


def test_openai_response_text_prefers_standard_content() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": "final answer",
                    "reasoning_content": "hidden reasoning",
                    "reasoning": "other reasoning",
                }
            }
        ]
    }

    assert _response_text(response) == "final answer"
