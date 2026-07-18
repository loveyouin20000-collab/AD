from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rad.checkpoints.manifest_v1 import (
    load_manifest,
    sha256_file,
)
from rad.models.descriptors import (
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM, sum_preserving_fusion
from rad.models.lse import LSE
from rad.targets.residual_gain import build_gain_target_record
from rad.trainers.fusion_trainer import FusionLossWeights, compute_fusion_objective
from rad.trainers.lse_trainer import compute_lse_objective

__all__ = [
    "GateResult",
    "JointStepOutput",
    "JointTrainer",
    "NoRegressionThresholds",
    "StagedCheckpointInfo",
    "compute_cost_weight",
    "evaluate_no_regression",
    "sha256_file",
    "soft_expected_depth_ratio",
    "validate_staged_checkpoint_pair",
]


@dataclass(frozen=True)
class StagedCheckpointInfo:
    fusion_path: Path
    lse_path: Path
    fusion_sha256: str
    lse_sha256: str
    candidate_layers: tuple[int, ...]
    source_dataset: str
    split_manifest_hash: str
    preprocessing_hash: str
    teacher_checkpoint_hash: str
    descriptor_stats_hash: str
    reference_full_depth_metrics: dict[str, float]


@dataclass(frozen=True)
class NoRegressionThresholds:
    max_pixel_ap_drop: float
    max_pro_drop: float
    max_mean_error_relative_increase: float


@dataclass(frozen=True)
class GateResult:
    passed: bool
    pixel_ap_drop: float
    pro_drop: float
    mean_error_relative_increase: float
    reasons: tuple[str, ...]


@dataclass
class JointStepOutput:
    total_loss: torch.Tensor
    fusion_loss: torch.Tensor
    lse_loss: torch.Tensor
    compute_loss: torch.Tensor
    compute_weight: float
    soft_expected_depth_ratio: torch.Tensor
    online_targets_detached: bool


def compute_cost_weight(
    progress: float,
    final_weight: float,
    ramp_fraction: float,
) -> float:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0,1]")
    if final_weight < 0.0:
        raise ValueError("final_weight must be nonnegative")
    if not 0.0 < ramp_fraction <= 1.0:
        raise ValueError("ramp_fraction must be in (0,1]")
    return final_weight * min(progress / ramp_fraction, 1.0)


def soft_expected_depth_ratio(
    sufficiency_logits: torch.Tensor,
    exit_layers: tuple[int, ...],
    final_depth: int,
    temperature: float,
) -> torch.Tensor:
    if sufficiency_logits.ndim != 2 or sufficiency_logits.shape[1] != len(exit_layers):
        raise ValueError("sufficiency_logits must have shape [B, number_of_exit_layers]")
    if tuple(sorted(set(exit_layers))) != exit_layers:
        raise ValueError("exit_layers must be strictly increasing")
    if exit_layers and exit_layers[-1] >= final_depth:
        raise ValueError("every exit layer must be shallower than final_depth")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    q = torch.sigmoid(sufficiency_logits / temperature)
    survival = torch.ones(q.shape[0], device=q.device, dtype=q.dtype)
    expected = torch.zeros_like(survival)
    for index, depth in enumerate(exit_layers):
        stop_probability = survival * q[:, index]
        expected = expected + stop_probability * (float(depth) / float(final_depth))
        survival = survival * (1.0 - q[:, index])
    expected = expected + survival
    return expected


def evaluate_no_regression(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    thresholds: NoRegressionThresholds,
) -> GateResult:
    pixel_ap_drop = reference["pixel_ap"] - candidate["pixel_ap"]
    pro_drop = reference["pro"] - candidate["pro"]
    ref_err = reference["mean_sample_error"]
    mean_error_relative_increase = (candidate["mean_sample_error"] - ref_err) / max(
        abs(ref_err), 1e-8
    )

    reasons: list[str] = []
    if pixel_ap_drop > thresholds.max_pixel_ap_drop:
        reasons.append("pixel_ap_drop")
    if pro_drop > thresholds.max_pro_drop:
        reasons.append("pro_drop")
    if mean_error_relative_increase > thresholds.max_mean_error_relative_increase:
        reasons.append("mean_error_relative_increase")

    return GateResult(
        passed=len(reasons) == 0,
        pixel_ap_drop=float(pixel_ap_drop),
        pro_drop=float(pro_drop),
        mean_error_relative_increase=float(mean_error_relative_increase),
        reasons=tuple(reasons),
    )


def _manifest_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".manifest.json")


def validate_staged_checkpoint_pair(
    fusion_checkpoint: str | Path,
    lse_checkpoint: str | Path,
    *,
    expected_candidate_layers: tuple[int, ...] | None = None,
) -> StagedCheckpointInfo:
    fusion_path = Path(fusion_checkpoint)
    lse_path = Path(lse_checkpoint)
    fusion_manifest = load_manifest(_manifest_path(fusion_path))
    lse_manifest = load_manifest(_manifest_path(lse_path))

    fusion_hash = sha256_file(fusion_path)
    lse_hash = sha256_file(lse_path)
    if fusion_hash != fusion_manifest.checkpoint_sha256:
        raise ValueError(
            "checkpoint_sha256 mismatch: expected "
            f"{fusion_manifest.checkpoint_sha256}, got {fusion_hash}"
        )
    if lse_hash != lse_manifest.checkpoint_sha256:
        raise ValueError(
            "checkpoint_sha256 mismatch: expected "
            f"{lse_manifest.checkpoint_sha256}, got {lse_hash}"
        )

    if fusion_manifest.stage != "fusion":
        raise ValueError(f"fusion checkpoint stage must be fusion, got {fusion_manifest.stage}")
    if lse_manifest.stage != "lse":
        raise ValueError(f"lse checkpoint stage must be lse, got {lse_manifest.stage}")
    if fusion_manifest.status != "passed":
        raise ValueError("fusion checkpoint status must be passed")
    if lse_manifest.status != "passed":
        raise ValueError("lse checkpoint status must be passed")

    if fusion_manifest.candidate_layers != lse_manifest.candidate_layers:
        raise ValueError("candidate_layers mismatch between fusion and lse manifests")
    if (
        expected_candidate_layers is not None
        and fusion_manifest.candidate_layers != expected_candidate_layers
    ):
        raise ValueError("candidate_layers do not match resolved config")

    provenance_keys = (
        "source_dataset",
        "split_manifest_hash",
        "preprocessing_hash",
        "teacher_checkpoint_hash",
        "descriptor_stats_hash",
    )
    for key in provenance_keys:
        if getattr(fusion_manifest, key) != getattr(lse_manifest, key):
            raise ValueError(f"provenance mismatch on {key}")

    if lse_manifest.upstream_fusion_checkpoint_hash != fusion_hash:
        raise ValueError("upstream_fusion_checkpoint_hash does not match fusion checkpoint")

    if fusion_manifest.reference_full_depth_metrics is None:
        raise ValueError("fusion manifest requires reference_full_depth_metrics")

    return StagedCheckpointInfo(
        fusion_path=fusion_path,
        lse_path=lse_path,
        fusion_sha256=fusion_hash,
        lse_sha256=lse_hash,
        candidate_layers=fusion_manifest.candidate_layers,
        source_dataset=fusion_manifest.source_dataset,
        split_manifest_hash=fusion_manifest.split_manifest_hash,
        preprocessing_hash=fusion_manifest.preprocessing_hash,
        teacher_checkpoint_hash=fusion_manifest.teacher_checkpoint_hash,
        descriptor_stats_hash=fusion_manifest.descriptor_stats_hash,
        reference_full_depth_metrics={
            k: float(v) for k, v in fusion_manifest.reference_full_depth_metrics.items()
        },
    )


class JointTrainer(nn.Module):
    """Joint DLCM+LSE fine-tuning with differentiable compute surrogate."""

    def __init__(
        self,
        *,
        dlcm: DLCM,
        lse: LSE,
        layer_extractor: LayerDescriptorExtractor,
        context_extractor: CheckpointContextExtractor,
        train_depths: tuple[int, ...],
        candidate_layers: tuple[int, ...],
        early_depths: tuple[int, ...],
        full_depth: int,
        fusion_loss_weight: float = 1.0,
        lse_loss_weight: float = 0.5,
        compute_final_weight: float = 0.05,
        compute_ramp_fraction: float = 0.2,
        compute_target_depth_ratio: float = 0.75,
        soft_exit_temperature: float = 1.0,
        epsilon_gain: float = 0.05,
        epsilon_absolute: float = 0.5,
        sufficiency_weight: float = 0.5,
        fusion_loss_weights: FusionLossWeights | None = None,
        normalizer: DescriptorNormalizer | None = None,
        grad_clip_norm: float = 5.0,
    ) -> None:
        super().__init__()
        self.dlcm = dlcm
        self.lse = lse
        self.layer_extractor = layer_extractor
        self.context_extractor = context_extractor
        self.normalizer = normalizer
        self.train_depths = tuple(train_depths)
        self.candidate_layers = tuple(candidate_layers)
        self.early_depths = tuple(early_depths)
        self.full_depth = int(full_depth)
        self.fusion_loss_weight = float(fusion_loss_weight)
        self.lse_loss_weight = float(lse_loss_weight)
        self.compute_final_weight = float(compute_final_weight)
        self.compute_ramp_fraction = float(compute_ramp_fraction)
        self.compute_target_depth_ratio = float(compute_target_depth_ratio)
        self.soft_exit_temperature = float(soft_exit_temperature)
        self.epsilon_gain = float(epsilon_gain)
        self.epsilon_absolute = float(epsilon_absolute)
        self.sufficiency_weight = float(sufficiency_weight)
        self.fusion_loss_weights = fusion_loss_weights or FusionLossWeights()
        self.grad_clip_norm = float(grad_clip_norm)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.dlcm.parameters()
        yield from self.lse.parameters()

    def trainable_parameter_names(self) -> list[str]:
        names: list[str] = []
        for name, _ in self.dlcm.named_parameters():
            names.append(f"dlcm.{name}")
        for name, _ in self.lse.named_parameters():
            names.append(f"lse.{name}")
        return names

    def _fusion_config(self) -> dict[str, Any]:
        return {
            "train_depths": self.train_depths,
            "loss_weights": self.fusion_loss_weights,
            "layer_extractor": self.layer_extractor,
            "context_extractor": self.context_extractor,
            "normalizer": self.normalizer,
        }

    def _lse_config(self) -> dict[str, Any]:
        return {
            "early_depths": self.early_depths,
            "sufficiency_weight": self.sufficiency_weight,
        }

    def _build_lse_states(
        self,
        batch: Mapping[str, Any],
    ) -> dict[int, torch.Tensor]:
        maps_by_depth: dict[int, torch.Tensor] = batch["maps_by_depth"]
        layer_ids_by_depth: dict[int, torch.Tensor] = batch["layer_ids_by_depth"]
        states: dict[int, torch.Tensor] = {}
        prev_fused: torch.Tensor | None = None

        for depth in self.early_depths:
            maps = maps_by_depth[depth]
            layer_ids = layer_ids_by_depth[depth]
            b, n_layers = maps.shape[:2]
            valid_mask = torch.ones(b, n_layers, dtype=torch.bool, device=maps.device)
            maps_4d = maps.squeeze(2) if maps.ndim == 5 else maps
            layer_desc = self.layer_extractor(maps_4d, valid_mask=valid_mask)
            if self.normalizer is not None:
                flat = layer_desc.reshape(b * n_layers, -1)
                flat = self.normalizer.transform(flat)
                layer_desc = flat.view(b, n_layers, -1)
            ctx = self.context_extractor(
                maps_4d,
                valid_mask=valid_mask,
                layer_ids=layer_ids,
                prev_fused=prev_fused,
            )
            weights = self.dlcm(layer_desc, ctx, layer_ids, valid_mask)
            fused = sum_preserving_fusion(maps, weights, valid_mask)
            states[depth] = torch.cat([layer_desc.mean(dim=1), ctx], dim=-1)
            prev_fused = fused

        return states

    def _online_targets(
        self,
        sample_errors: Mapping[int, torch.Tensor],
    ) -> tuple[dict[int, dict[str, torch.Tensor]], bool]:
        record = build_gain_target_record(
            sample_errors,
            epsilon_gain=self.epsilon_gain,
            epsilon_absolute=self.epsilon_absolute,
            early_depths=self.early_depths,
            full_depth=self.full_depth,
            stop_gradient=False,
        )
        target_by_depth: dict[int, dict[str, torch.Tensor]] = {}
        all_detached = True
        for depth in self.early_depths:
            gain = record["gains"][depth].to(device=sample_errors[depth].device)
            if gain.ndim == 0:
                gain = gain.expand_as(sample_errors[depth])
            elif gain.ndim == 1 and gain.shape[0] != sample_errors[depth].shape[0]:
                gain = gain.reshape(-1)
            sufficient = record["sufficient"][depth].to(
                device=sample_errors[depth].device, dtype=sample_errors[depth].dtype
            )
            if sufficient.dtype == torch.bool:
                sufficient = sufficient.float()
            gain = gain.detach()
            sufficient = sufficient.detach()
            all_detached = all_detached and not gain.requires_grad and not sufficient.requires_grad
            target_by_depth[depth] = {"gain": gain, "sufficient": sufficient}
        return target_by_depth, all_detached

    def train_step(
        self,
        batch: Mapping[str, Any],
        progress: float,
        optimizer: torch.optim.Optimizer,
    ) -> JointStepOutput:
        self.train()
        optimizer.zero_grad(set_to_none=True)

        fusion_result = compute_fusion_objective(
            self.dlcm,
            batch,
            self._fusion_config(),
            training_fraction=progress,
        )
        target_by_depth, targets_detached = self._online_targets(fusion_result.sample_errors)
        state_by_depth = self._build_lse_states(batch)
        lse_result = compute_lse_objective(
            self.lse,
            state_by_depth,
            target_by_depth,
            self._lse_config(),
        )

        depth_ratio = soft_expected_depth_ratio(
            lse_result.sufficiency_logits,
            self.early_depths,
            self.full_depth,
            self.soft_exit_temperature,
        )
        compute_loss = torch.relu(
            depth_ratio.mean() - self.compute_target_depth_ratio
        )
        compute_weight = compute_cost_weight(
            progress,
            self.compute_final_weight,
            self.compute_ramp_fraction,
        )
        total_loss = (
            self.fusion_loss_weight * fusion_result.total_loss
            + self.lse_loss_weight * lse_result.total_loss
            + compute_weight * compute_loss
        )

        total_loss.backward()
        params = list(self.trainable_parameters())
        torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
        optimizer.step()

        return JointStepOutput(
            total_loss=total_loss,
            fusion_loss=fusion_result.total_loss.detach(),
            lse_loss=lse_result.total_loss.detach(),
            compute_loss=compute_loss.detach(),
            compute_weight=compute_weight,
            soft_expected_depth_ratio=depth_ratio.detach(),
            online_targets_detached=targets_detached,
        )
