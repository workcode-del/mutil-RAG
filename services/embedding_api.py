from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel

from paper_rag.embedding import Qwen3VLEmbedder
from paper_rag.log import configure_logging


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-VL Embedding Service")
model: Qwen3VLEmbedder | None = None


class BatchRequest(BaseModel):
    values: list[str]


@app.on_event("startup")
def load_model() -> None:
    global model
    logger.info("Loading embedding service model")
    model = Qwen3VLEmbedder(
        model_name=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
        official_repo=os.getenv("QWEN3_VL_RETRIEVAL_REPO"),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", "2048")),
        device=os.getenv("EMBEDDING_DEVICE", "cuda"),
        model_source=os.getenv("MODEL_SOURCE", "modelscope"),
        local_path=os.getenv("EMBEDDING_LOCAL_PATH"),
        modelscope_id=os.getenv("EMBEDDING_MODELSCOPE_ID"),
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "data/models"),
    )
    logger.info("Embedding service ready: dimension=%d", model.dimension)


def active_model() -> Qwen3VLEmbedder:
    if model is None:
        raise RuntimeError("Qwen3-VL embedding model is not loaded")
    return model


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "dimension": model.dimension if model is not None else None,
    }


@app.post("/embed/query")
def embed_query(request: BatchRequest) -> dict:
    return {"vectors": active_model().embed_queries(request.values).tolist()}


@app.post("/embed/text")
def embed_text(request: BatchRequest) -> dict:
    return {"vectors": active_model().embed_texts(request.values).tolist()}


@app.post("/embed/image")
def embed_image(request: BatchRequest) -> dict:
    return {"vectors": active_model().embed_images(request.values).tolist()}
