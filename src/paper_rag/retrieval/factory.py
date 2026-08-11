from __future__ import annotations

from paper_rag.evidence_graph import EvidenceGraph
from paper_rag.retrieval.base import EvidenceRetriever
from paper_rag.retrieval.baselines import PCSTEvidenceRetriever, RankedEvidenceRetriever
from paper_rag.retrieval.ec_bfr import ECBFRConfig, EvidenceClosureBudgetedForestRetriever


RETRIEVAL_METHODS = ("top_k", "one_hop", "ppr", "pcst", "pcst_closure", "ec_bfr")


def build_evidence_retriever(
    method: str,
    graph: EvidenceGraph,
    config: ECBFRConfig,
    *,
    selection_top_k: int = 10,
) -> EvidenceRetriever:
    normalized = method.strip().lower()
    if normalized == "ec_bfr":
        return EvidenceClosureBudgetedForestRetriever(graph, config)
    if normalized in {"pcst", "pcst_closure"}:
        return PCSTEvidenceRetriever(graph, config, apply_closure=normalized == "pcst_closure")
    if normalized in {"top_k", "one_hop", "ppr"}:
        return RankedEvidenceRetriever(
            graph,
            top_k=selection_top_k,
            budget=config.budget,
            image_unit=config.image_unit,
            hops=1 if normalized == "one_hop" else 0,
            use_ppr=normalized == "ppr",
        )
    raise ValueError(f"Unsupported retrieval method: {method}. Choose from {RETRIEVAL_METHODS}")
