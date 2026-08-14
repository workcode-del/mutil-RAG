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
