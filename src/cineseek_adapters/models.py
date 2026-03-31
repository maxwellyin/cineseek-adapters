from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class IdentityAdapter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1)


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


def build_adapter(name: str, dim: int, hidden_dim: int = 768, dropout: float = 0.1, residual_scale: float = 0.2) -> nn.Module:
    if name == "identity":
        return IdentityAdapter()
    if name == "linear":
        return LinearResidualAdapter(dim=dim, residual_scale=residual_scale)
    if name == "residual_mlp":
        return ResidualMLPAdapter(dim=dim, hidden_dim=hidden_dim, dropout=dropout, residual_scale=residual_scale)
    raise ValueError(f"Unsupported adapter: {name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

