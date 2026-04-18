from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class IdentityAdapter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1)

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


class LinearResidualAdapter(nn.Module):
    def __init__(self, dim: int, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.residual_scale = residual_scale
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return F.normalize(x + self.residual_scale * (self.proj(x) - x), dim=-1)

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


class ResidualMLPAdapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 768, dropout: float = 0.1, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return F.normalize(x + self.residual_scale * self.net(x), dim=-1)

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


class ConcatLinearItemAdapter(nn.Module):
    def __init__(self, query_dim: int, item_dim: int, title_weight: float = 0.5) -> None:
        super().__init__()
        if item_dim != query_dim * 2:
            raise ValueError(f"concat_linear expects item_dim=2*query_dim, got query_dim={query_dim}, item_dim={item_dim}")
        self.proj = nn.Linear(item_dim, query_dim)
        self.title_weight = title_weight
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            eye = torch.eye(query_dim)
            self.proj.weight[:, :query_dim] = title_weight * eye
            self.proj.weight[:, query_dim:] = (1.0 - title_weight) * eye

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_item(x)

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1)

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x.float()), dim=-1)


def build_adapter(
    name: str,
    dim: int,
    hidden_dim: int = 768,
    dropout: float = 0.1,
    residual_scale: float = 0.2,
    item_dim: int | None = None,
    title_weight: float = 0.5,
) -> nn.Module:
    if name == "identity":
        return IdentityAdapter()
    if name == "linear":
        return LinearResidualAdapter(dim=dim, residual_scale=residual_scale)
    if name == "residual_mlp":
        return ResidualMLPAdapter(dim=dim, hidden_dim=hidden_dim, dropout=dropout, residual_scale=residual_scale)
    if name == "concat_linear":
        return ConcatLinearItemAdapter(query_dim=dim, item_dim=item_dim or dim * 2, title_weight=title_weight)
    raise ValueError(f"Unsupported adapter: {name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
