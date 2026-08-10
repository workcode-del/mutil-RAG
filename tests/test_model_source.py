from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from paper_rag.model_source import resolve_model_reference


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
