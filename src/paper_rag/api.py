import logging
from typing import Any

from paper_rag.domain import QuerySpec
from paper_rag.pipeline import ScientificRAGPipeline


logger = logging.getLogger(__name__)


def create_app(pipeline: ScientificRAGPipeline | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the api dependency group") from exc

    app = FastAPI(title="Scientific Evidence RAG", version="0.1.0")
    app.state.pipeline = pipeline

    class QueryRequest(BaseModel):
        query: str
        answer_type: str = "free_text"
        entity_type: str | None = None
        metric: str | None = None
        operator: str | None = None
        value: float | None = None
        unit: str | None = None
        conditions: list[str] = Field(default_factory=list)
        required_modalities: list[str] = Field(default_factory=list)
        entities: list[str] = Field(default_factory=list)

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
                entity_type=request.entity_type,
                metric=request.metric,
                operator=request.operator,
                value=request.value,
                unit=request.unit,
                conditions=request.conditions,
                required_modalities=request.required_modalities,
                entities=request.entities,
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
            "query_spec": {
                "answer_type": result.query.answer_type,
                "entity_type": result.query.entity_type,
                "metric": result.query.metric,
                "operator": result.query.operator,
                "value": result.query.value,
                "unit": result.query.unit,
                "conditions": result.query.conditions,
                "required_modalities": result.query.required_modalities,
                "entities": result.query.entities,
                "auto_parsed_fields": result.query.auto_parsed_fields,
            },
            "forest": [
                {
                    "paper_id": tree.paper_id,
                    "node_ids": sorted(tree.node_ids),
                    "evidence": [
                        {
                            "node_id": node_id,
                            "node_type": active_pipeline.graph.nodes[node_id].node_type.value,
                            "text": active_pipeline.graph.nodes[node_id].searchable_text,
                            "image_path": active_pipeline.graph.nodes[node_id].image_path,
                            "page": active_pipeline.graph.nodes[node_id].page,
                            "bbox": active_pipeline.graph.nodes[node_id].bbox.as_list()
                            if active_pipeline.graph.nodes[node_id].bbox
                            else None,
                            "confidence": active_pipeline.graph.nodes[node_id].confidence,
                        }
                        for node_id in sorted(tree.node_ids)
                    ],
                    "cost": tree.cost,
                    "metadata": tree.metadata,
                }
                for tree in result.forest.trees
            ],
            "total_cost": result.forest.total_cost,
        }

    return app


app = create_app()
