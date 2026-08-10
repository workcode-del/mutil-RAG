from __future__ import annotations

from pathlib import Path


def resolve_model_reference(
    model_id: str,
    *,
    local_path: str | Path | None = None,
    modelscope_id: str | None = None,
    source: str = "modelscope",
    cache_dir: str | Path = "data/models",
) -> str:
    """Resolve a model with deterministic local-first, ModelScope-second semantics.

    ``local_path`` is always preferred when it exists. ``source=local`` forbids
    network access. ``source=modelscope`` downloads a missing model once and returns
    the cached snapshot directory, so downstream libraries only receive local paths.
    """
    normalized_source = source.strip().lower()
    if normalized_source not in {"local", "modelscope", "auto"}:
        raise ValueError("model source must be one of: local, modelscope, auto")

    candidates = [local_path, model_id]
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path.resolve())

    if normalized_source == "local":
        requested = local_path or model_id
        raise FileNotFoundError(f"Local model path does not exist: {requested}")

    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover - installation guard
        raise RuntimeError(
            "Install ModelScope in the paper-rag Conda environment: pip install modelscope"
        ) from exc

    resolved_cache = Path(cache_dir).expanduser().resolve()
    resolved_cache.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        model_id=modelscope_id or model_id,
        cache_dir=str(resolved_cache),
    )
    return str(Path(downloaded).resolve())
