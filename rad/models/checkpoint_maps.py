from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from rad.types import CheckpointOutput


def anomaly_map_from_tokens(
    anomaly_features: torch.Tensor,
    normal_features: torch.Tensor,
    patch_tokens: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Mirror train.generate_anomaly_map_from_tokens; returns [B, H, W]."""
    batch_size = anomaly_features.shape[0]

    anomaly_features_norm = F.normalize(anomaly_features, dim=1, eps=1e-8)
    normal_features_norm = F.normalize(normal_features, dim=1, eps=1e-8)
    patch_tokens_norm = F.normalize(patch_tokens, dim=2, eps=1e-8)

    anomaly_sim = torch.cosine_similarity(
        patch_tokens_norm,
        anomaly_features_norm.unsqueeze(1),
        dim=2,
    )
    normal_sim = torch.cosine_similarity(
        patch_tokens_norm,
        normal_features_norm.unsqueeze(1),
        dim=2,
    )
    anomaly_score = anomaly_sim - normal_sim
    if torch.isnan(anomaly_score).any():
        anomaly_score = torch.nan_to_num(anomaly_score, nan=0.0)

    num_patches = anomaly_score.shape[1]
    side = int(num_patches**0.5)
    if side * side != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a perfect square")

    anomaly_map = anomaly_score.reshape(batch_size, side, side)
    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(1),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return anomaly_map


class CheckpointMapGenerator:
    """Build causal checkpoint-conditioned anomaly maps A_{l|d}."""

    def __init__(
        self,
        image_size: int,
        candidate_layers: Sequence[int] | None = None,
    ) -> None:
        self.image_size = image_size
        self.candidate_layers = (
            tuple(candidate_layers) if candidate_layers is not None else None
        )

    def build(
        self,
        depth: int,
        outputs: Mapping[int, CheckpointOutput],
    ) -> dict[int, torch.Tensor]:
        if depth not in outputs:
            raise KeyError(f"checkpoint depth {depth} missing from outputs")

        ref = outputs[depth]
        layer_ids = [layer for layer in outputs if layer <= depth]
        if self.candidate_layers is not None:
            allowed = set(self.candidate_layers)
            layer_ids = [layer for layer in layer_ids if layer in allowed]
        layer_ids = sorted(layer_ids)

        maps: dict[int, torch.Tensor] = {}
        for layer in layer_ids:
            amap = anomaly_map_from_tokens(
                ref.anomaly_token,
                ref.normal_token,
                outputs[layer].patch_tokens,
                self.image_size,
            )
            maps[layer] = amap.unsqueeze(1)
        return maps
