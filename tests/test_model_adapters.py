from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from paper_rag.chart import ChartExtractionResult, SelfEnsemblingChartExtractor
from paper_rag.model_source import resolve_model_reference
from paper_rag.reranking.qwen3_vl import Qwen3VLReranker


def test_existing_local_model_is_preferred(tmp_path) -> None:
    local = tmp_path / "model"
    local.mkdir()
    assert resolve_model_reference("remote/model", local_path=local) == str(local.resolve())


def test_local_mode_forbids_missing_model(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_model_reference(
            "remote/model",
            local_path=tmp_path / "missing",
            source="local",
        )


def test_modelscope_download_returns_cached_snapshot(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "cache" / "remote" / "model"
    snapshot.mkdir(parents=True)

    def fake_download(*, model_id: str, cache_dir: str) -> str:
        assert model_id == "remote/model"
        assert cache_dir == str((tmp_path / "cache").resolve())
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=fake_download))
    assert resolve_model_reference(
        "ignored/model",
        modelscope_id="remote/model",
        cache_dir=tmp_path / "cache",
    ) == str(snapshot.resolve())


def test_self_ensemble_uses_cell_median_and_reports_disagreement() -> None:
    outputs = iter(["x,y\n1,10\n2,20", "x,y\n1,12\n2,20", "x,y\n1,11\n2,20"])

    def sample(_path: str) -> ChartExtractionResult:
        return ChartExtractionResult(next(outputs), "ok", extractor="mock")

    result = SelfEnsemblingChartExtractor(sample, repeats=3).extract("figure.png")

    assert "1,11" in result.linearized_table
    assert "2,20" in result.linearized_table
    assert result.extractor == "self-ensemble"
    assert result.uncertainty is not None and result.uncertainty > 0


def test_self_ensemble_rejects_single_sample() -> None:
    with pytest.raises(ValueError, match="at least two"):
        SelfEnsemblingChartExtractor(
            lambda _path: ChartExtractionResult("x,y\n1,2", "ok"), repeats=1
        )


def test_multimodal_document_normalization() -> None:
    assert Qwen3VLReranker._normalize_document("plain text") == {"text": "plain text"}
    assert Qwen3VLReranker._normalize_document(
        {"image": "figure.png", "text": "caption", "unused": None}
    ) == {"image": "figure.png", "text": "caption"}


def test_multimodal_document_requires_content() -> None:
    with pytest.raises(ValueError):
        Qwen3VLReranker._normalize_document({"text": "", "image": None})
