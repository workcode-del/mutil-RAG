import numpy as np

from paper_rag.domain import EvidenceNode, NodeType, QuerySpec, SearchHit
from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.pipeline import ScientificRAGPipeline
from paper_rag.retrieval.baselines import RankedEvidenceRetriever
from paper_rag.evaluation.runner import EvaluationSample


class FakeEmbedder:
    def embed_queries(self, texts):
        return np.zeros((len(texts), 2), dtype=np.float32)


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
