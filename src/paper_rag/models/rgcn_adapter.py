from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RGCNConfig:
    input_dimension: int = 2048
    hidden_dimension: int = 256
    layers: int = 2
    num_bases: int = 8
    dropout: float = 0.1


def create_rgcn_model(
    metadata: tuple[list[str], list[tuple[str, str, str]]],
    config: RGCNConfig,
):
    """Create a typed homogeneous R-GCN with the same projections as SRMG-HGT."""
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import RGCNConv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install graph training dependencies before creating R-GCN") from exc

    node_types, edge_types = metadata
    relation_ids = {edge_type: index for index, edge_type in enumerate(edge_types)}
    num_relations = max(len(edge_types), 1)
    num_bases = min(max(config.num_bases, 1), num_relations)

    class ScientificRGCN(nn.Module):
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
                    RGCNConv(
                        config.hidden_dimension,
                        config.hidden_dimension,
                        num_relations,
                        num_bases=num_bases,
                        aggr="mean",
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
            present_types = [node_type for node_type in node_types if node_type in x_dict]
            projected = {
                node_type: self.type_projection[node_type](x_dict[node_type])
                for node_type in present_types
            }
            offsets: dict[str, int] = {}
            cursor = 0
            for node_type in present_types:
                offsets[node_type] = cursor
                cursor += projected[node_type].shape[0]
            hidden = torch.cat([projected[node_type] for node_type in present_types], dim=0)
            edges = []
            relation_values = []
            for edge_type in edge_types:
                edge_index = edge_index_dict.get(edge_type)
                if edge_index is None:
                    continue
                source_type, _, target_type = edge_type
                offset = torch.tensor(
                    [[offsets[source_type]], [offsets[target_type]]],
                    device=edge_index.device,
                )
                edges.append(edge_index + offset)
                relation_values.append(
                    torch.full(
                        (edge_index.shape[1],),
                        relation_ids[edge_type],
                        dtype=torch.long,
                        device=edge_index.device,
                    )
                )
            edge_index = (
                torch.cat(edges, dim=1)
                if edges
                else torch.empty((2, 0), dtype=torch.long, device=hidden.device)
            )
            edge_type_tensor = (
                torch.cat(relation_values)
                if relation_values
                else torch.empty((0,), dtype=torch.long, device=hidden.device)
            )
            for conv in self.convs:
                messages = conv(hidden, edge_index, edge_type_tensor)
                combined = hidden + self.dropout(messages)
                hidden = torch.nn.functional.normalize(combined.float(), dim=-1).to(
                    combined.dtype
                )
            result = {}
            for node_type in present_types:
                start = offsets[node_type]
                result[node_type] = hidden[start : start + projected[node_type].shape[0]]
            return result

        def encode_query(self, query_embedding: Any) -> Any:
            projected = self.query_projection(query_embedding)
            return torch.nn.functional.normalize(projected.float(), dim=-1).to(
                projected.dtype
            )

    return ScientificRGCN()
