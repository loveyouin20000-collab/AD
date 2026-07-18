from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def _describe(
        self,
        maps: torch.Tensor,
        layer_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        prev_fused: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # maps: [B, L, 1, H, W]
        maps_4d = maps.squeeze(2) if maps.ndim == 5 else maps
        layer_desc = self.layer_extractor(maps_4d, valid_mask=valid_mask)
        if self.normalizer is not None:
            b, l, d = layer_desc.shape
            flat = layer_desc.reshape(b * l, d)
            flat = self.normalizer.transform(flat)
            layer_desc = flat.view(b, l, d)
        ctx = self.context_extractor(
            maps_4d,
            valid_mask=valid_mask,
            layer_ids=layer_ids,
            prev_fused=prev_fused,
        )
        return layer_desc, ctx

    def compute_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        maps_by_depth: dict[int, torch.Tensor] = batch["maps_by_depth"]
        layer_ids_by_depth: dict[int, torch.Tensor] = batch["layer_ids_by_depth"]
        mask = batch["mask"]
        image_label = batch["image_label"]
        teacher_logits = batch["teacher_logits"]
        shapley_by_depth: dict[int, dict[str, torch.Tensor]] = batch["shapley_by_depth"]

        total = teacher_logits.new_tensor(0.0)
        metrics: dict[str, torch.Tensor] = {}
        prev_fused: torch.Tensor | None = None

        teacher_prob = torch.sigmoid(teacher_logits)

        for depth in self.train_depths:
            maps = maps_by_depth[depth]  # [B, L, 1, H, W]
            layer_ids = layer_ids_by_depth[depth]
            b, l = maps.shape[:2]
            valid_mask = torch.ones(b, l, dtype=torch.bool, device=maps.device)

            layer_desc, ctx = self._describe(maps, layer_ids, valid_mask, prev_fused)
            weights = self.dlcm(layer_desc, ctx, layer_ids, valid_mask)
            fused = sum_preserving_fusion(maps, weights, valid_mask)  # [B,1,H,W]

            loc = sample_localization_error(fused, mask, image_label).mean()
            map_kd = confidence_weighted_distillation(fused, teacher_logits)
            boundary_kd = boundary_l1_loss(torch.sigmoid(fused), teacher_prob).mean()

            target_dist = shapley_by_depth[depth]["distribution"].to(weights.device)
            # KL(weights || target): weights * (log w - log t)
            w = weights.clamp_min(1e-8)
            t = target_dist.clamp_min(1e-8)
            contrib_kl = (w * (w.log() - t.log())).sum(dim=-1).mean()

            depth_loss = (
                self.loss_weights.loc_weight(depth) * loc
                + self.loss_weights.map_kd * map_kd
                + self.loss_weights.boundary_kd * boundary_kd
                + self.loss_weights.contribution * contrib_kl
            )
            total = total + depth_loss

            entropy = -(w * w.log()).sum(dim=-1).mean()
            metrics[f"loc_{depth}"] = loc.detach()
            metrics[f"map_kd_{depth}"] = map_kd.detach()
            metrics[f"boundary_kd_{depth}"] = boundary_kd.detach()
            metrics[f"contrib_kl_{depth}"] = contrib_kl.detach()
            metrics[f"weight_entropy_{depth}"] = entropy.detach()
            metrics[f"avg_weight_{depth}"] = weights.detach().mean()

            prev_fused = fused.detach()

        metrics["loss"] = total
        return metrics

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
        # No positive pixels: define AP as 1 if preds are all low else 0
        return float((probs < 0.5).mean())
    return float(average_precision_score(gt, probs))
