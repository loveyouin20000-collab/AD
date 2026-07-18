from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from rad.losses.distillation import confidence_weighted_distillation
from rad.losses.localization import boundary_l1_loss, sample_localization_error
from rad.models.descriptors import (
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM, sum_preserving_fusion


@dataclass(frozen=True)
class FusionLossWeights:
    lambda_loc: dict[int, float] | None = None
    map_kd: float = 0.5
    boundary_kd: float = 0.2
    contribution: float = 0.5

    def loc_weight(self, depth: int) -> float:
        defaults = {12: 0.5, 18: 0.75, 24: 1.0}
        table = self.lambda_loc if self.lambda_loc is not None else defaults
        return float(table.get(depth, 1.0))


@dataclass
class FusionForwardResult:
    total_loss: torch.Tensor
    fused_logits: dict[int, torch.Tensor]
    weights: dict[int, torch.Tensor]
    sample_errors: dict[int, torch.Tensor]
    loss_terms: dict[str, torch.Tensor]


def compute_fusion_objective(
    dlcm: nn.Module,
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    training_fraction: float = 0.0,
) -> FusionForwardResult:
    """Pure fusion objective shared by staged and joint trainers."""
    train_depths: tuple[int, ...] = tuple(config["train_depths"])
    loss_weights: FusionLossWeights = config.get("loss_weights") or FusionLossWeights()
    layer_extractor: LayerDescriptorExtractor = config["layer_extractor"]
    context_extractor: CheckpointContextExtractor = config["context_extractor"]
    normalizer: DescriptorNormalizer | None = config.get("normalizer")

    if hasattr(dlcm, "set_progress"):
        dlcm.set_progress(training_fraction)

    maps_by_depth: dict[int, torch.Tensor] = batch["maps_by_depth"]
    layer_ids_by_depth: dict[int, torch.Tensor] = batch["layer_ids_by_depth"]
    mask = batch["mask"]
    image_label = batch["image_label"]
    teacher_logits = batch["teacher_logits"]
    shapley_by_depth: dict[int, dict[str, torch.Tensor]] = batch["shapley_by_depth"]

    total = teacher_logits.new_tensor(0.0)
    loss_terms: dict[str, torch.Tensor] = {}
    fused_logits: dict[int, torch.Tensor] = {}
    weights_by_depth: dict[int, torch.Tensor] = {}
    sample_errors: dict[int, torch.Tensor] = {}
    prev_fused: torch.Tensor | None = None

    teacher_prob = torch.sigmoid(teacher_logits)

    for depth in train_depths:
        maps = maps_by_depth[depth]
        layer_ids = layer_ids_by_depth[depth]
        b, l = maps.shape[:2]
        valid_mask = torch.ones(b, l, dtype=torch.bool, device=maps.device)

        maps_4d = maps.squeeze(2) if maps.ndim == 5 else maps
        layer_desc = layer_extractor(maps_4d, valid_mask=valid_mask)
        if normalizer is not None:
            flat = layer_desc.reshape(b * l, -1)
            flat = normalizer.transform(flat)
            layer_desc = flat.view(b, l, -1)
        ctx = context_extractor(
            maps_4d,
            valid_mask=valid_mask,
            layer_ids=layer_ids,
            prev_fused=prev_fused,
        )
        weights = dlcm(layer_desc, ctx, layer_ids, valid_mask)
        fused = sum_preserving_fusion(maps, weights, valid_mask)

        per_sample_err = sample_localization_error(fused, mask, image_label)
        loc = per_sample_err.mean()
        map_kd = confidence_weighted_distillation(fused, teacher_logits)
        boundary_kd = boundary_l1_loss(torch.sigmoid(fused), teacher_prob).mean()

        target_dist = shapley_by_depth[depth]["distribution"].to(weights.device)
        w = weights.clamp_min(1e-8)
        t = target_dist.clamp_min(1e-8)
        contrib_kl = (w * (w.log() - t.log())).sum(dim=-1).mean()

        depth_loss = (
            loss_weights.loc_weight(depth) * loc
            + loss_weights.map_kd * map_kd
            + loss_weights.boundary_kd * boundary_kd
            + loss_weights.contribution * contrib_kl
        )
        total = total + depth_loss

        entropy = -(w * w.log()).sum(dim=-1).mean()
        loss_terms[f"loc_{depth}"] = loc.detach()
        loss_terms[f"map_kd_{depth}"] = map_kd.detach()
        loss_terms[f"boundary_kd_{depth}"] = boundary_kd.detach()
        loss_terms[f"contrib_kl_{depth}"] = contrib_kl.detach()
        loss_terms[f"weight_entropy_{depth}"] = entropy.detach()
        loss_terms[f"avg_weight_{depth}"] = weights.detach().mean()

        fused_logits[depth] = fused
        weights_by_depth[depth] = weights
        sample_errors[depth] = per_sample_err
        prev_fused = fused.detach()

    loss_terms["loss"] = total
    return FusionForwardResult(
        total_loss=total,
        fused_logits=fused_logits,
        weights=weights_by_depth,
        sample_errors=sample_errors,
        loss_terms=loss_terms,
    )


class FusionTrainer(nn.Module):
    """Train DLCM on cached causal maps; ViT/SCA/transforms stay frozen (not loaded)."""

    def __init__(
        self,
        dlcm: DLCM,
        layer_extractor: LayerDescriptorExtractor | None = None,
        context_extractor: CheckpointContextExtractor | None = None,
        normalizer: DescriptorNormalizer | None = None,
        loss_weights: FusionLossWeights | None = None,
        train_depths: tuple[int, ...] = (12, 18, 24),
        candidate_layers: tuple[int, ...] = (6, 12, 18, 24),
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.dlcm = dlcm
        self.layer_extractor = layer_extractor or LayerDescriptorExtractor()
        self.context_extractor = context_extractor or CheckpointContextExtractor(
            backbone_depth=max(candidate_layers)
        )
        self.normalizer = normalizer
        self.loss_weights = loss_weights or FusionLossWeights()
        self.train_depths = tuple(train_depths)
        self.candidate_layers = tuple(candidate_layers)
        self.freeze_backbone = bool(freeze_backbone)

    def trainable_parameters(self):
        if self.freeze_backbone:
            return self.dlcm.parameters()
        return self.parameters()

    def trainable_parameter_names(self) -> list[str]:
        if self.freeze_backbone:
            return [n for n, _ in self.dlcm.named_parameters()]
        return [n for n, _ in self.named_parameters()]

    def _fusion_config(self) -> dict[str, Any]:
        return {
            "train_depths": self.train_depths,
            "loss_weights": self.loss_weights,
            "layer_extractor": self.layer_extractor,
            "context_extractor": self.context_extractor,
            "normalizer": self.normalizer,
        }

    def compute_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        result = compute_fusion_objective(
            self.dlcm,
            batch,
            self._fusion_config(),
            training_fraction=0.0,
        )
        return result.loss_terms

    def training_step(
        self,
        batch: dict[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = self.compute_losses(batch)
        metrics["loss"].backward()
        optimizer.step()
        return {k: float(v.detach()) for k, v in metrics.items()}


def pixel_average_precision(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Micro-averaged pixel AP over a batch (CPU sklearn)."""
    from sklearn.metrics import average_precision_score

    probs = torch.sigmoid(logits).detach().cpu().reshape(-1).numpy()
    gt = mask.detach().cpu().reshape(-1).numpy()
    if gt.max() < 0.5:
        return float((probs < 0.5).mean())
    return float(average_precision_score(gt, probs))
