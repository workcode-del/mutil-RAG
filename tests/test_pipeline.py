import numpy as np
import pytest

import paper_rag.workflow as workflow
from paper_rag.benchmarking.runner import _embed_queries
from paper_rag.domain import EvidenceNode, NodeType, QuerySpec, SearchHit
from paper_rag.embedding import ExactEmbeddingStore
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.evaluation.runner import EvaluationSample, evaluate
from paper_rag.indexing import compute_base_embeddings
from paper_rag.pipeline import ScientificRAGPipeline
from paper_rag.retrieval.baselines import RankedEvidenceRetriever
from paper_rag.workflow import (
    embedding_cache_is_current,
    embedding_config_digest,
    index_graph,
)


class FakeEmbedder:
    calls = 0

    def embed_queries(self, texts):
        self.calls += 1
        vectors = np.asarray(
            [[len(text), sum(map(ord, text)) % 997] for text in texts], dtype=np.float32
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        if len(texts) > 1:
            vectors[:, 0] += 0.02
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


class RecordingStore:
    def __init__(self):
        self.paper_ids = None
        self.candidate_node_ids = None

    def search(
        self,
        query,
        query_vector,
        node_types,
        per_type_top_k,
        paper_ids=None,
        candidate_node_ids=None,
    ):
        self.paper_ids = paper_ids
        self.candidate_node_ids = candidate_node_ids
        return [SearchHit("p:s", "p", NodeType.SENTENCE, 1.0, {"embedding": 1.0})]


def test_pipeline_applies_sample_scope() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:s", "p", NodeType.SENTENCE, text="answer"))
    store = RecordingStore()
    pipeline = ScientificRAGPipeline(
        graph,
        FakeEmbedder(),
        store,
        RankedEvidenceRetriever(graph, top_k=1, budget=10, image_unit=1),
    )

    sample = EvaluationSample.from_dict(
        {
            "query_id": "q",
            "query": "question",
            "paper_id": "p",
            "relevant_node_ids": ["p:s"],
            "candidate_node_ids": ["p:s"],
        },
        0,
    )
    pipeline.run(
        sample.query,
        paper_ids=sample.paper_ids,
        candidate_node_ids=sample.candidate_node_ids,
    )

    assert store.paper_ids == {"p"}
    assert store.candidate_node_ids == {"p:s"}


def test_batched_queries_preserve_results_and_report_latency() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:s", "p", NodeType.SENTENCE, text="answer"))
    embedder = FakeEmbedder()
    pipeline = ScientificRAGPipeline(
        graph,
        embedder,
        RecordingStore(),
        RankedEvidenceRetriever(graph, top_k=1, budget=10, image_unit=1),
    )

    samples = [
        EvaluationSample.from_dict(
            {"query_id": str(index), "query": query, "relevant_node_ids": ["p:s"]},
            index,
        )
        for index, query in enumerate(("question", "another question"))
    ]
    sample = samples[0]
    vectors, embedding_ms, min_cosine = _embed_queries(embedder, samples, batch_size=128)
    embedder.calls = 0
    online = pipeline.run(QuerySpec("question"))
    cached = pipeline.run(QuerySpec("question"), query_vector=vectors["0"])
    report = evaluate(
        pipeline,
        [sample],
        query_vectors=vectors,
        query_embedding_ms=embedding_ms,
    )
    metrics = report["details"][0]["metrics"]

    assert embedder.calls == 1
    assert online.hits == cached.hits
    assert online.forest == cached.forest
    assert 0.999 <= min_cosine < 1.0
    assert metrics["query_embedding_amortized_ms"] == embedding_ms
    assert metrics["latency_ms"] == metrics["retrieval_latency_ms"] + embedding_ms


def test_query_request_is_registered_as_json_body() -> None:
    pytest.importorskip("fastapi")
    from paper_rag.api import create_app

    operation = create_app().openapi()["paths"]["/query"]["post"]

    assert "requestBody" in operation
    assert not any(
        parameter["name"] == "request" and parameter["in"] == "query"
        for parameter in operation.get("parameters", [])
    )


def test_exact_embedding_store_ranks_with_sample_scope() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:a", "p", NodeType.SENTENCE, text="a"))
    graph.add_node(EvidenceNode("p:b", "p", NodeType.SENTENCE, text="b"))
    graph.add_node(EvidenceNode("q:c", "q", NodeType.SENTENCE, text="c"))
    store = ExactEmbeddingStore(
        graph,
        {
            "p:a": np.asarray([1.0, 0.0]),
            "p:b": np.asarray([0.0, 1.0]),
            "q:c": np.asarray([1.0, 0.0]),
        },
    )

    hits = store.search(
        "query",
        np.asarray([1.0, 0.0]),
        [NodeType.SENTENCE],
        2,
        paper_ids={"p"},
        candidate_node_ids={"p:a", "p:b", "q:c"},
    )

    assert [hit.node_id for hit in hits] == ["p:a", "p:b"]


def test_embedding_cache_signature_changes_with_model_configuration() -> None:
    first = {
        "embedding": {"model": "model-a", "dimension": 2, "query_instruction": "retrieve"},
        "runtime": {"device": "cuda"},
    }
    second = {
        "embedding": {"model": "model-b", "dimension": 2, "query_instruction": "retrieve"},
        "runtime": {"device": "cuda"},
    }

    assert embedding_config_digest(first) != embedding_config_digest(second)


def test_table_uses_mixed_embedding_when_image_is_available() -> None:
    graph = EvidenceGraph()
    graph.add_node(
        EvidenceNode(
            "p:t",
            "p",
            NodeType.TABLE,
            text="Model | F1",
            image_path="table.png",
        )
    )

    class MixedEmbedder:
        dimension = 2
        items = None

        def embed_mixed(self, items):
            self.items = items
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        def embed_texts(self, texts):
            raise AssertionError("table should use mixed embedding")

        def embed_images(self, paths):
            return np.empty((0, 2), dtype=np.float32)

    embedder = MixedEmbedder()
    embeddings, report = compute_base_embeddings(graph, embedder)

    assert embedder.items == [{"text": "Model | F1", "image": "table.png"}]
    assert set(embeddings) == {"p:t"}
    assert report.table_nodes == 1


def test_benchmark_embedding_cache_is_model_aware_and_skips_qdrant(
    tmp_path, monkeypatch
) -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:a", "p", NodeType.SENTENCE, text="a"))
    graph_path = tmp_path / "graph.json"
    cache_path = tmp_path / "base_embeddings.npz"
    save_graph(graph, graph_path)
    first = {"embedding": {"model": "model-a", "dimension": 2}}
    second = {"embedding": {"model": "model-b", "dimension": 2}}

    class CacheEmbedder:
        dimension = 2

        def embed_texts(self, texts):
            return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

        def embed_images(self, paths):
            return np.empty((len(paths), 2), dtype=np.float32)

    def reject_qdrant(_config):
        raise AssertionError("Benchmark embedding cache must not build Qdrant")

    monkeypatch.setattr(workflow, "load_yaml", lambda _path: first)
    monkeypatch.setattr(workflow, "build_embedder", lambda _config: CacheEmbedder())
    monkeypatch.setattr(workflow, "build_vector_store", reject_qdrant)

    index_graph(graph_path, "unused.yaml", cache_path, upsert_vector_store=False)

    assert embedding_cache_is_current(graph_path, first, cache_path)
    assert not embedding_cache_is_current(graph_path, second, cache_path)
