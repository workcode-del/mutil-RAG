import numpy as np

from paper_rag.benchmarking.runner import _embed_queries
from paper_rag.domain import EvidenceNode, NodeType, QuerySpec, SearchHit
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.evaluation.runner import EvaluationSample, evaluate
from paper_rag.pipeline import ScientificRAGPipeline
from paper_rag.retrieval.baselines import RankedEvidenceRetriever


class FakeEmbedder:
    calls = 0

    def embed_queries(self, texts):
        self.calls += 1
        vectors = np.asarray(
            [[len(text), sum(map(ord, text)) % 997] for text in texts], dtype=np.float32
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


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


def test_pipeline_passes_sample_paper_scope_to_candidate_store() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:s", "p", NodeType.SENTENCE, text="answer"))
    store = RecordingStore()
    pipeline = ScientificRAGPipeline(
        graph,
        FakeEmbedder(),
        store,
        RankedEvidenceRetriever(graph, top_k=1, budget=10, image_unit=1),
    )

    pipeline.run(QuerySpec("question"), paper_ids={"p"}, candidate_node_ids={"p:s"})

    assert store.paper_ids == {"p"}
    assert store.candidate_node_ids == {"p:s"}


def test_pipeline_uses_precomputed_query_vector() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:s", "p", NodeType.SENTENCE, text="answer"))
    embedder = FakeEmbedder()
    pipeline = ScientificRAGPipeline(
        graph,
        embedder,
        RecordingStore(),
        RankedEvidenceRetriever(graph, top_k=1, budget=10, image_unit=1),
    )

    vector = embedder.embed_queries(["question"])[0]
    embedder.calls = 0
    online = pipeline.run(QuerySpec("question"))
    cached = pipeline.run(QuerySpec("question"), query_vector=vector)

    assert embedder.calls == 1
    assert online.hits == cached.hits
    assert online.forest == cached.forest


def test_batched_query_embeddings_match_single_query_embeddings() -> None:
    samples = [
        EvaluationSample.from_dict(
            {"query_id": str(index), "query": query, "relevant_node_ids": ["p:s"]},
            index,
        )
        for index, query in enumerate(("first question", "second question", "third"))
    ]
    embedder = FakeEmbedder()

    batched, _ = _embed_queries(embedder, samples, batch_size=2)

    for sample in samples:
        expected = embedder.embed_queries([sample.query.query])[0]
        np.testing.assert_allclose(batched[sample.query_id], expected, rtol=1e-6, atol=1e-6)


def test_batched_latency_is_reported_separately() -> None:
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("p:s", "p", NodeType.SENTENCE, text="answer"))
    pipeline = ScientificRAGPipeline(
        graph,
        FakeEmbedder(),
        RecordingStore(),
        RankedEvidenceRetriever(graph, top_k=1, budget=10, image_unit=1),
    )
    sample = EvaluationSample.from_dict(
        {"query_id": "q", "query": "question", "relevant_node_ids": ["p:s"]}, 0
    )

    report = evaluate(
        pipeline,
        [sample],
        query_vectors={"q": FakeEmbedder().embed_queries(["question"])[0]},
        query_embedding_ms=3.0,
    )
    metrics = report["details"][0]["metrics"]

    assert metrics["query_embedding_amortized_ms"] == 3.0
    assert metrics["latency_ms"] == metrics["retrieval_latency_ms"] + 3.0


def test_evaluation_sample_reads_candidate_scope() -> None:
    sample = EvaluationSample.from_dict(
        {
            "query_id": "q",
            "query": "question",
            "paper_id": "p",
            "relevant_node_ids": ["p:s"],
            "candidate_node_ids": ["p:s", "p:n"],
        },
        0,
    )

    assert sample.candidate_node_ids == {"p:s", "p:n"}
