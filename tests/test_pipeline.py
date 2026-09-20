import numpy as np
import pytest

import paper_rag.workflow as workflow
from paper_rag.benchmarking.runner import _embed_queries
from paper_rag.bootstrap import validate_graph_embedding_config
from paper_rag.io import write_json
from paper_rag.domain import EvidenceNode, NodeType, QuerySpec, SearchHit
from paper_rag.embedding import ExactEmbeddingStore
from paper_rag.evidence_graph import EvidenceGraph, save_graph
from paper_rag.evidence_graph import build_figure_text_views
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


class RecordingReranker:
    documents = None

    def score(self, query, documents):
        self.documents = documents
        return [1.0 - index * 0.1 for index in range(len(documents))]


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


def test_pipeline_reranks_only_configured_top_n() -> None:
    graph = EvidenceGraph()
    for index in range(3):
        graph.add_node(EvidenceNode(f"p:s{index}", "p", NodeType.SENTENCE, text=str(index)))

    class ThreeHitStore(RecordingStore):
        def search(self, *args, **kwargs):
            return [
                SearchHit(
                    f"p:s{index}",
                    "p",
                    NodeType.SENTENCE,
                    3.0 - index,
                    {"embedding": 3.0 - index},
                )
                for index in range(3)
            ]

    reranker = RecordingReranker()
    pipeline = ScientificRAGPipeline(
        graph,
        FakeEmbedder(),
        ThreeHitStore(),
        RankedEvidenceRetriever(graph, top_k=3, budget=100, image_unit=1),
        reranker=reranker,
        reranker_top_n=2,
    )

    result = pipeline.run(QuerySpec("question"))

    assert reranker.documents == ["0", "1"]
    assert "reranker" not in result.hits[2].score_components


def test_rerank_quota_preserves_low_ranked_visual_candidates() -> None:
    graph = EvidenceGraph()
    hits = [SearchHit(f"p:{i}", "p", NodeType.SENTENCE, 10 - i) for i in range(5)]
    hits += [SearchHit("p:f", "p", NodeType.FIGURE, 0.1)]
    pipeline = ScientificRAGPipeline(
        graph, None, RecordingStore(), None, reranker_top_n=3, reranker_min_per_type=1,
    )
    assert [h.node_id for h in pipeline._rerank_candidates(hits)] == ["p:0", "p:1", "p:f"]
    pipeline.reranker_top_n = 1
    assert len(pipeline._rerank_candidates(hits)) == 1


def test_empty_recall_still_records_zero_reranker_coverage():
    class EmptyStore(RecordingStore):
        def search(self, *args, **kwargs):
            return []

    graph = EvidenceGraph()
    pipeline = ScientificRAGPipeline(
        graph, None, EmptyStore(),
        RankedEvidenceRetriever(graph, top_k=1, budget=10, image_unit=1),
        reranker=RecordingReranker(),
    )
    result = pipeline.run(QuerySpec("question"))
    assert result.stages == {"candidate": [], "reranker_input": []}


def test_partial_rerank_does_not_reward_exposure_alone() -> None:
    hits = [
        SearchHit("a", "p", NodeType.SENTENCE, 1.0, {"embedding": 1.0}),
        SearchHit("b", "p", NodeType.FIGURE, 0.8, {"embedding": 0.8, "reranker": 0.9}),
    ]
    pipeline = ScientificRAGPipeline(
        EvidenceGraph(), None, RecordingStore(), None, complete_reranker_ranking=True,
    )
    pipeline._fuse_hits(hits)
    assert [h.node_id for h in hits] == ["a", "b"]
    assert "reranker" not in hits[0].score_components


def test_weighted_reranking_can_promote_a_reserved_image() -> None:
    hits = [
        SearchHit("a", "p", NodeType.SENTENCE, 1.0, {"embedding": 1.0, "reranker": 0.1}),
        SearchHit("b", "p", NodeType.SENTENCE, 0.8, {"embedding": 0.8}),
        SearchHit("c", "p", NodeType.FIGURE, 0.5, {"embedding": 0.5, "reranker": 0.9}),
    ]
    pipeline = ScientificRAGPipeline(
        EvidenceGraph(), None, RecordingStore(), None,
        complete_reranker_ranking=True, fusion_weights={"embedding": 1, "reranker": 2},
    )
    pipeline._fuse_hits(hits)
    assert hits[0].node_id == "c"


def test_figure_text_mixture_preserves_existing_description_and_id() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode(
        "p:f", "p", NodeType.FIGURE, image_path="figure.png",
        attributes={"text_view": "existing description"},
    ))

    class Embedder:
        dimension = 2

        def embed_images(self, paths):
            return np.tile([2.0, 0.0], (len(paths), 1))

        def embed_texts(self, texts):
            assert texts == ["existing description"]
            return np.tile([0.0, 3.0], (len(texts), 1))

    build_figure_text_views(graph)
    build_figure_text_views(graph)
    assert graph.nodes["p:f"].searchable_text == "existing description"
    embeddings, _ = compute_base_embeddings(graph, Embedder(), figure_text_weight=0.5)
    assert set(embeddings) == {"p:f"}
    np.testing.assert_allclose(embeddings["p:f"], [2 ** -0.5, 2 ** -0.5])
    pure, _ = compute_base_embeddings(graph, Embedder(), figure_text_weight=0)
    np.testing.assert_allclose(pure["p:f"], [2, 0])
    assert embedding_config_digest({"embedding": {"figure_text_weight": 0}}) != (
        embedding_config_digest({"embedding": {"figure_text_weight": 0.5}})
    )


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


def test_graph_artifacts_cannot_silently_mix_embedding_representations(tmp_path):
    config = {"embedding": {"figure_text_weight": 0.25}}
    write_json(tmp_path / "training.json", {})
    with pytest.raises(ValueError, match="Legacy graph artifacts"):
        validate_graph_embedding_config(config, tmp_path)
    write_json(tmp_path / "training.json", {
        "embedding_config_sha256": embedding_config_digest(config),
    })
    validate_graph_embedding_config(config, tmp_path)
    with pytest.raises(ValueError, match="different embedding configuration"):
        validate_graph_embedding_config({"embedding": {"figure_text_weight": 0}}, tmp_path)


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
