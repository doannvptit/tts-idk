from __future__ import annotations

from torch import nn

from model.components.mha import CausalSelfAttention


class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        inner_dim = d_model * mlp_ratio
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_head=n_head, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask=None):
        x = x + self.attn(self.ln_1(x), causal_mask=causal_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
