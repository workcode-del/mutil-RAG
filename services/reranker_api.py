from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel

from paper_rag.reranking import Qwen3VLReranker
from paper_rag.log import configure_logging


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-VL Reranker Service")
model: Qwen3VLReranker | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[str | dict[str, object]]


@app.on_event("startup")
def load_model() -> None:
    global model
    logger.info("Loading reranker service model")
    model = Qwen3VLReranker(
        model_name=os.getenv("RERANKER_MODEL", "Qwen/Qwen3-VL-Reranker-2B"),
        official_repo=os.getenv("QWEN3_VL_RETRIEVAL_REPO"),
        device=os.getenv("RERANKER_DEVICE", "cuda"),
        model_source=os.getenv("MODEL_SOURCE", "modelscope"),
        local_path=os.getenv("RERANKER_LOCAL_PATH"),
        modelscope_id=os.getenv("RERANKER_MODELSCOPE_ID"),
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "data/models"),
    )
    logger.info("Reranker service ready")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/rerank")
def rerank(request: RerankRequest) -> dict:
    if model is None:
        raise RuntimeError("Reranker model is not loaded")
    return {"scores": model.score(request.query, request.documents)}
