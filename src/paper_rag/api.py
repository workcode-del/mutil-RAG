import logging
from typing import Any

from paper_rag.domain import QuerySpec
from paper_rag.pipeline import ScientificRAGPipeline


logger = logging.getLogger(__name__)


def create_app(pipeline: ScientificRAGPipeline | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the api dependency group") from exc

    app = FastAPI(title="Scientific Evidence RAG", version="0.1.0")
    app.state.pipeline = pipeline

    class QueryRequest(BaseModel):
        query: str
        answer_type: str = "free_text"
        metric: str | None = None
        operator: str | None = None
        value: float | None = None
        unit: str | None = None
        conditions: list[str] = []

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "pipeline_ready": app.state.pipeline is not None}

    @app.post("/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        active_pipeline = app.state.pipeline
        if active_pipeline is None:
            raise HTTPException(503, "Pipeline is not initialized; see deployment documentation")
        logger.info("Query received: length=%d", len(request.query))
        result = active_pipeline.run(
            QuerySpec(
                query=request.query,
                answer_type=request.answer_type,
                metric=request.metric,
                operator=request.operator,
                value=request.value,
                unit=request.unit,
                conditions=request.conditions,
            ),
            log_stages=True,
        )
        logger.info(
            "Query complete: evidence=%d cost=%d generated=%s",
            len(result.forest.node_ids),
            result.forest.total_cost,
            result.answer is not None,
        )
        return {
            "answer": result.answer.text if result.answer else None,
            "evidence_ids": result.answer.evidence_ids if result.answer else [],
            "forest": [
                {
                    "paper_id": tree.paper_id,
                    "node_ids": sorted(tree.node_ids),
                    "cost": tree.cost,
                    "metadata": tree.metadata,
                }
                for tree in result.forest.trees
            ],
            "total_cost": result.forest.total_cost,
        }

    return app


app = create_app()
