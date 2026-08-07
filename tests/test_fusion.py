from paper_rag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_does_not_depend_on_raw_score_scale() -> None:
    result = reciprocal_rank_fusion({"gme": ["a", "b"], "reranker": ["b", "a"]})
    assert result["a"] == result["b"]
