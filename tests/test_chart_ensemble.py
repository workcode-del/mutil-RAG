from __future__ import annotations

from paper_rag.chart import ChartExtractionResult, SelfEnsemblingChartExtractor


def test_self_ensemble_uses_cell_median_and_reports_disagreement() -> None:
    outputs = iter(
        [
            "x,y\n1,10\n2,20",
            "x,y\n1,12\n2,20",
            "x,y\n1,11\n2,20",
        ]
    )

    def sample(_path: str) -> ChartExtractionResult:
        return ChartExtractionResult(next(outputs), "ok", extractor="mock")

    result = SelfEnsemblingChartExtractor(sample, repeats=3).extract("figure.png")

    assert "1,11" in result.linearized_table
    assert "2,20" in result.linearized_table
    assert result.extractor == "self-ensemble"
    assert result.uncertainty is not None and result.uncertainty > 0


def test_self_ensemble_rejects_single_sample() -> None:
    def sample(_path: str) -> ChartExtractionResult:
        return ChartExtractionResult("x,y\n1,2", "ok")

    try:
        SelfEnsemblingChartExtractor(sample, repeats=1)
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected a validation error")
