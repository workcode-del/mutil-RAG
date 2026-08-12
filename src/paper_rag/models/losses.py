from __future__ import annotations


def relation_info_nce(anchor, positive, negatives, temperature: float = 0.07):
    """Relation-supervised InfoNCE used for Figure-Caption/Mention/ChartData pairs."""
    import torch
    import torch.nn.functional as functional

    anchor = functional.normalize(anchor, dim=-1)
    positive = functional.normalize(positive, dim=-1)
    negatives = functional.normalize(negatives, dim=-1)
    positive_logit = (anchor * positive).sum(dim=-1, keepdim=True)
    negative_logits = torch.einsum("bd,bnd->bn", anchor, negatives)
    logits = torch.cat([positive_logit, negative_logits], dim=1) / temperature
    labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return functional.cross_entropy(logits, labels)


def query_evidence_margin_loss(query, positive, negative, margin: float = 0.2):
    """Train Wq and the graph adapter in the same 256-dimensional retrieval space."""
    import torch.nn.functional as functional

    query = functional.normalize(query, dim=-1)
    positive = functional.normalize(positive, dim=-1)
    negative = functional.normalize(negative, dim=-1)
    positive_score = (query * positive).sum(dim=-1)
    negative_score = (query * negative).sum(dim=-1)
    return functional.relu(margin - positive_score + negative_score).mean()
