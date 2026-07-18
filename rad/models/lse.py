from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GainPrediction:
    """LSE outputs at one early-exit checkpoint."""

    mean: torch.Tensor
    log_variance: torch.Tensor
    sufficiency_logit: torch.Tensor


class LSE(nn.Module):
    """Localization Sufficiency Estimator.

    Shared MLP ``state_dim -> 64 -> 32`` plus a learned depth embedding for
    configured early checkpoints (default 12 and 18). Mean uses softplus;
    log-variance is clamped to ``[-8, 4]``.
    """

    def __init__(
        self,
        state_dim: int,
        early_depths: Sequence[int] = (12, 18),
        depth_emb_dim: int = 16,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        depths = tuple(int(d) for d in early_depths)
        if len(depths) < 1:
            raise ValueError("early_depths must be non-empty")
        if len(set(depths)) != len(depths):
            raise ValueError("early_depths must be unique")

        self.state_dim = int(state_dim)
        self.early_depths = depths
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self._depth_to_index = {d: i for i, d in enumerate(depths)}

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.depth_emb = nn.Embedding(len(depths), depth_emb_dim)
        self.fuse = nn.Linear(32 + depth_emb_dim, 32)
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)
        self.suf_head = nn.Linear(32, 1)

    def _encode_depth(self, depth_id: torch.Tensor) -> torch.Tensor:
        """Map absolute depth ids to embedding indices. depth_id: [B] -> [B]."""
        if depth_id.ndim != 1:
            raise ValueError("depth_id must have shape [B]")
        idx = torch.zeros_like(depth_id)
        for d, i in self._depth_to_index.items():
            idx = torch.where(depth_id == d, torch.full_like(depth_id, i), idx)
        # Any unsupported depth raises
        known = torch.zeros_like(depth_id, dtype=torch.bool)
        for d in self._depth_to_index:
            known = known | (depth_id == d)
        if not bool(known.all()):
            bad = depth_id[~known].unique().tolist()
            raise ValueError(
                f"depth_id values {bad} not in early_depths {self.early_depths}"
            )
        return idx

    def forward(self, state: torch.Tensor, depth_id: torch.Tensor) -> GainPrediction:
        """
        Args:
            state: [B, state_dim] selector / descriptor features at the checkpoint
            depth_id: [B] absolute checkpoint depths (e.g. 12 or 18)
        """
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape [B, {self.state_dim}]")
        if depth_id.shape[0] != state.shape[0]:
            raise ValueError("depth_id batch size must match state")

        h = self.mlp(state)
        emb = self.depth_emb(self._encode_depth(depth_id))
        h = F.relu(self.fuse(torch.cat([h, emb], dim=-1)))
        mean = F.softplus(self.mean_head(h).squeeze(-1))
        log_variance = self.logvar_head(h).squeeze(-1).clamp(self.logvar_min, self.logvar_max)
        sufficiency_logit = self.suf_head(h).squeeze(-1)
        return GainPrediction(
            mean=mean,
            log_variance=log_variance,
            sufficiency_logit=sufficiency_logit,
        )


def heteroscedastic_gaussian_nll(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-sample Gaussian NLL with predicted log-variance. -> [B]"""
    inv_var = torch.exp(-log_variance)
    return 0.5 * (log_variance + (target - mean).square() * inv_var + math.log(2 * math.pi))


def lse_loss(
    pred: GainPrediction,
    target_gain: torch.Tensor,
    target_sufficient: torch.Tensor,
    *,
    sufficiency_weight: float = 0.5,
) -> torch.Tensor:
    """Heteroscedastic NLL + 0.5 * BCEWithLogits sufficiency loss (scalar)."""
    nll = heteroscedastic_gaussian_nll(pred.mean, pred.log_variance, target_gain).mean()
    bce = F.binary_cross_entropy_with_logits(
        pred.sufficiency_logit,
        target_sufficient.to(dtype=pred.sufficiency_logit.dtype),
    )
    return nll + float(sufficiency_weight) * bce
