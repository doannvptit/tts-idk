from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class ResidualPostNetConfig:
    layer0_dim: int
    llm_hidden_dim: int
    output_dim: int
    model_dim: int
    num_layers: int = 16
    num_steps: int = 6
    dropout: float = 0.0


class SharedResidualMLP(nn.Module):
    """Shared residual operator reused across refinement steps."""

    def __init__(self, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, condition], dim=-1))


class ResidualPostNet(nn.Module):
    """
    Predict Moss acoustic continuous embeddings from:
    - codebook layer-0 embeddings
    - LLM hidden states at the same token positions

    The network uses:
    - one input MLP to project fused inputs into a latent state
    - one shared residual MLP reused for iterative refinement
    """

    def __init__(self, config: ResidualPostNetConfig) -> None:
        super().__init__()
        self.config = config
        fused_dim = config.layer0_dim + config.llm_hidden_dim
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, config.model_dim * 2),
            nn.SiLU(),
            nn.Linear(config.model_dim * 2, config.model_dim),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.shared_residual = SharedResidualMLP(config.model_dim, config.dropout)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.num_layers * config.output_dim),
        )

    def forward(self, layer0_embeddings: torch.Tensor, llm_hidden_states: torch.Tensor) -> torch.Tensor:
        aligned_count = min(layer0_embeddings.shape[0], llm_hidden_states.shape[0])
        layer0 = layer0_embeddings[:aligned_count]
        hidden = llm_hidden_states[:aligned_count].to(layer0.device)
        condition = self.condition_proj(torch.cat([layer0, hidden], dim=-1))
        state = self.state_proj(condition)
        for _ in range(self.config.num_steps):
            state = state + self.shared_residual(state, condition)
        output = self.output_proj(state)
        return output.view(aligned_count, self.config.num_layers, self.config.output_dim)
