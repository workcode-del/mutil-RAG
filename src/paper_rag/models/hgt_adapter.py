from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HGTConfig:
    input_dimension: int = 2048
    hidden_dimension: int = 256
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1


def create_hgt_model(metadata: tuple[list[str], list[tuple[str, str, str]]], config: HGTConfig):
    """Create the trainable SRMG adapter lazily so base package imports without PyTorch."""
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import HGTConv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install graph training dependencies before creating HGT") from exc

    node_types = metadata[0]

    class SRMGHGT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.type_projection = nn.ModuleDict(
                {
                    node_type: nn.Linear(config.input_dimension, config.hidden_dimension)
                    for node_type in node_types
                }
            )
            self.convs = nn.ModuleList(
                [
                    HGTConv(
                        config.hidden_dimension,
                        config.hidden_dimension,
                        metadata,
                        heads=config.heads,
                    )
                    for _ in range(config.layers)
                ]
            )
            self.query_projection = nn.Sequential(
                nn.Linear(config.input_dimension, config.hidden_dimension),
                nn.GELU(),
                nn.Linear(config.hidden_dimension, config.hidden_dimension),
            )
            self.dropout = nn.Dropout(config.dropout)

        def encode_graph(
            self,
            x_dict: dict[str, Any],
            edge_index_dict: dict[tuple[str, str, str], Any],
        ) -> dict[str, Any]:
            hidden = {
                node_type: self.type_projection[node_type](features)
                for node_type, features in x_dict.items()
            }
            for conv in self.convs:
                messages = conv(hidden, edge_index_dict)
                # PyG may omit types without incoming messages. Residuals keep them valid.
                hidden = {
                    node_type: torch.nn.functional.normalize(
                        residual
                        + self.dropout(messages.get(node_type, torch.zeros_like(residual))),
                        dim=-1,
                    )
                    for node_type, residual in hidden.items()
                }
            return hidden

        def encode_query(self, query_embedding: Any) -> Any:
            return torch.nn.functional.normalize(self.query_projection(query_embedding), dim=-1)

    return SRMGHGT()
