from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sum_preserving_fusion(
    maps: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Fuse layer maps with weights while preserving equal-fusion sum scale.

    Args:
        maps: [B, L, 1, H, W] or [B, L, C, H, W]
        weights: [B, L]
        valid_mask: [B, L] bool
    Returns:
        [B, C, H, W]  (channel dim preserved; typically C=1)
    """
    n_valid = valid_mask.sum(dim=1).clamp_min(1).to(maps.dtype)
    return (maps * weights[:, :, None, None, None]).sum(dim=1) * n_valid[:, None, None, None]


class DLCM(nn.Module):
    """Dynamic Layer Contribution Module.

    Shared MLP 18->64->32, 16-D layer embedding, 8-D context projection,
    and a zero-initialized scalar scorer with masked softmax (+ optional floor).
    """

    def __init__(
        self,
        max_layer_id: int = 24,
        alpha: float = 0.0,
        alpha_init: float = 0.1,
        alpha_warmup_fraction: float = 0.2,
    ) -> None:
        super().__init__()
        if max_layer_id < 1:
            raise ValueError("max_layer_id must be >= 1")
        self.max_layer_id = max_layer_id
        self.alpha = float(alpha)
        self.alpha_init = float(alpha_init)
        self.alpha_warmup_fraction = float(alpha_warmup_fraction)

        self.mlp = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.layer_emb = nn.Embedding(max_layer_id + 1, 16)
        self.ctx_proj = nn.Linear(8, 32)
        self.scorer = nn.Linear(32 + 16 + 32, 1)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)

    def alpha_for_progress(self, progress: float) -> float:
        """Linear decay of alpha from alpha_init to 0 over the first warmup fraction."""
        if progress < 0.0:
            progress = 0.0
        warm = self.alpha_warmup_fraction
        if warm <= 0.0 or progress >= warm:
            return 0.0
        return self.alpha_init * (1.0 - progress / warm)

    def set_progress(self, progress: float) -> None:
        self.alpha = self.alpha_for_progress(progress)

    def forward(
        self,
        layer_desc: torch.Tensor,
        checkpoint_context: torch.Tensor,
        layer_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        alpha: float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            layer_desc: [B, L, 18]
            checkpoint_context: [B, 8]
            layer_ids: [B, L] int (1..max_layer_id)
            valid_mask: [B, L] bool
            alpha: optional override for floor mixing
        Returns:
            weights [B, L] (invalid layers exactly 0; valid sum to 1)
        """
        if layer_desc.ndim != 3 or layer_desc.shape[-1] != 18:
            raise ValueError("layer_desc must have shape [B, L, 18]")
        if checkpoint_context.ndim != 2 or checkpoint_context.shape[-1] != 8:
            raise ValueError("checkpoint_context must have shape [B, 8]")

        b, l, _ = layer_desc.shape
        if layer_ids.shape != (b, l) or valid_mask.shape != (b, l):
            raise ValueError("layer_ids and valid_mask must have shape [B, L]")

        if torch.any(layer_ids < 0) or torch.any(layer_ids > self.max_layer_id):
            raise ValueError(f"layer_ids must be in [0, {self.max_layer_id}]")

        h = self.mlp(layer_desc)  # [B, L, 32]
        e = self.layer_emb(layer_ids.long())  # [B, L, 16]
        c = self.ctx_proj(checkpoint_context).unsqueeze(1).expand(-1, l, -1)  # [B, L, 32]
        scores = self.scorer(torch.cat([h, e, c], dim=-1)).squeeze(-1)  # [B, L]

        neg = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~valid_mask, neg)
        weights = F.softmax(masked_scores, dim=-1)
        weights = weights * valid_mask.to(weights.dtype)

        # Renormalize in case of numerical underflow on all-masked rows
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        floor_alpha = self.alpha if alpha is None else float(alpha)
        if floor_alpha > 0.0:
            n_valid = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(weights.dtype)
            uniform = valid_mask.to(weights.dtype) / n_valid
            weights = (1.0 - floor_alpha) * weights + floor_alpha * uniform
            weights = weights * valid_mask.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Exact zeros on invalid positions
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        return weights
