import pytest


def test_query_request_is_registered_as_json_body() -> None:
    pytest.importorskip("fastapi")

    from paper_rag.api import create_app

    operation = create_app().openapi()["paths"]["/query"]["post"]

    assert "requestBody" in operation
    assert not any(
        parameter["name"] == "request" and parameter["in"] == "query"
        for parameter in operation.get("parameters", [])
    )
