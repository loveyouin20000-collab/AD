from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from rad.models.checkpoint_maps import CheckpointMapGenerator
from rad.models.descriptors import (
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM, sum_preserving_fusion
from rad.models.lse import LSE, GainPrediction
from rad.models.policy import ExitSignals, PolicyProfile, should_exit
from rad.types import CheckpointOutput, StageCache


@dataclass
class AdaptiveResult:
    """Batch-size-1 (or small-B) adaptive inference output."""

    final_map: torch.Tensor
    image_score: torch.Tensor
    selected_depth: int
    checkpoint_trace: list[int]
    weights: dict[int, torch.Tensor] = field(default_factory=dict)
    gain_predictions: dict[int, GainPrediction] = field(default_factory=dict)
    timing_breakdown: dict[str, float] = field(default_factory=dict)
    exit_decisions: dict[int, bool] = field(default_factory=dict)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start: float, end: float) -> float:
    return float((end - start) * 1000.0)


def compute_exit_signals(
    fused: torch.Tensor,
    prev_fused: torch.Tensor | None,
    *,
    temperature: float = 1.0,
) -> ExitSignals:
    """Derive policy signals from the current fused map.

    - image_score: max of temperature-scaled sigmoid map in [0, 1]
    - map_uncertainty: spatial entropy of softmax over HxW (normalized by log HW)
    - stability: mean abs change vs previous fused (0 if none)
    """
    logits = fused / max(float(temperature), 1e-6)
    probs = torch.sigmoid(logits)
    image_score = float(probs.flatten(1).max(dim=1).values.mean().item())

    flat = logits.flatten(2)
    p = F.softmax(flat, dim=-1)
    entropy = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)
    max_h = float(torch.log(torch.tensor(float(flat.shape[-1]), device=fused.device)))
    map_uncertainty = float((entropy / max(max_h, 1e-8)).mean().item())

    if prev_fused is None:
        stability = 0.0
    else:
        stability = float((fused - prev_fused).abs().mean().item())

    return ExitSignals(
        map_uncertainty=map_uncertainty,
        image_score=image_score,
        stability=stability,
    )


class AdaptiveEngine(nn.Module):
    """True batch-size-1 adaptive early-exit inference over configurable checkpoints."""

    def __init__(
        self,
        *,
        visual: nn.Module,
        map_generator: CheckpointMapGenerator,
        dlcm: DLCM,
        lse: LSE,
        layer_extractor: LayerDescriptorExtractor,
        context_extractor: CheckpointContextExtractor,
        profile: PolicyProfile,
        candidate_layers: Sequence[int],
        early_depths: Sequence[int],
        full_depth: int,
        image_size: int,
        normalizer: DescriptorNormalizer | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        layers = tuple(sorted(int(x) for x in candidate_layers))
        if len(layers) < 1:
            raise ValueError("candidate_layers must be non-empty")
        if layers != tuple(sorted(set(layers))):
            raise ValueError("candidate_layers must be unique")
        early = tuple(sorted(int(x) for x in early_depths))
        if any(d not in layers for d in early):
            raise ValueError("early_depths must be a subset of candidate_layers")
        if int(full_depth) not in layers:
            raise ValueError("full_depth must be in candidate_layers")

        self.visual = visual
        self.map_generator = map_generator
        self.dlcm = dlcm
        self.lse = lse
        self.layer_extractor = layer_extractor
        self.context_extractor = context_extractor
        self.normalizer = normalizer
        self.profile = profile
        self.candidate_layers = layers
        self.early_depths = early
        self.full_depth = int(full_depth)
        self.image_size = int(image_size)
        self.temperature = float(temperature)

    def _device(self) -> torch.device:
        return next(self.dlcm.parameters()).device

    def _fuse_at_depth(
        self,
        outputs: Mapping[int, CheckpointOutput],
        depth: int,
        prev_fused: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused [B,1,H,W], weights [B,L], state [B, state_dim]."""
        device = self._device()
        maps_dict = self.map_generator.build(depth, outputs)
        avail = sorted(maps_dict.keys())
        stacked = torch.stack([maps_dict[l] for l in avail], dim=1).to(device)  # [B,L,1,H,W]
        b, l = stacked.shape[:2]
        layer_ids = torch.tensor([avail], dtype=torch.long, device=device).expand(b, -1)
        valid = torch.ones(b, l, dtype=torch.bool, device=device)
        maps_4d = stacked.squeeze(2)
        layer_desc = self.layer_extractor(maps_4d, valid_mask=valid)
        if self.normalizer is not None:
            flat = layer_desc.reshape(b * l, -1)
            flat = self.normalizer.transform(flat)
            layer_desc = flat.view(b, l, -1)
        ctx = self.context_extractor(
            maps_4d,
            valid_mask=valid,
            layer_ids=layer_ids,
            prev_fused=prev_fused,
        )
        weights = self.dlcm(layer_desc, ctx, layer_ids, valid)
        fused = sum_preserving_fusion(stacked, weights, valid)
        state = torch.cat([layer_desc.mean(dim=1), ctx], dim=-1)
        return fused, weights, state

    @torch.no_grad()
    def infer(
        self,
        image: torch.Tensor,
        *,
        force_full_depth: bool = False,
        measure_timing: bool = False,
    ) -> AdaptiveResult:
        """Run adaptive (or forced full-depth) inference.

        True early exit: only executes ViT blocks up to the selected depth and
        never builds maps for deeper checkpoints.
        """
        if image.ndim != 4:
            raise ValueError("image must have shape [B, C, H, W]")
        device = image.device
        import time

        timing: dict[str, float] = {
            "backbone": 0.0,
            "maps": 0.0,
            "dlcm": 0.0,
            "lse": 0.0,
            "policy": 0.0,
            "total": 0.0,
        }

        def tick() -> float:
            if measure_timing:
                _sync_if_cuda(device)
                return time.perf_counter()
            return 0.0

        t_all0 = tick()
        t0 = tick()
        cache: StageCache = self.visual.prepare_stage(image)
        timing["backbone"] += _elapsed_ms(t0, tick()) if measure_timing else 0.0

        outputs: dict[int, CheckpointOutput] = {}
        checkpoint_trace: list[int] = []
        weights_by_depth: dict[int, torch.Tensor] = {}
        gains: dict[int, GainPrediction] = {}
        exits: dict[int, bool] = {}
        prev_fused: torch.Tensor | None = None
        last_fused: torch.Tensor | None = None
        last_score = torch.zeros(image.shape[0], device=device)
        selected = self.full_depth

        for depth in self.candidate_layers:
            t0 = tick()
            out, cache = self.visual.run_to(cache, depth)
            timing["backbone"] += _elapsed_ms(t0, tick()) if measure_timing else 0.0
            outputs[depth] = out
            checkpoint_trace.append(depth)

            decide = depth in self.early_depths or depth == self.full_depth
            if not decide:
                continue

            t0 = tick()
            # maps + descriptors + dlcm measured under maps/dlcm
            maps_t0 = t0
            fused, weights, state = self._fuse_at_depth(outputs, depth, prev_fused)
            t1 = tick()
            if measure_timing:
                # attribute half to maps, half to dlcm for breakdown stability
                half = _elapsed_ms(maps_t0, t1) * 0.5
                timing["maps"] += half
                timing["dlcm"] += half
            weights_by_depth[depth] = weights.detach().cpu()
            last_fused = fused
            signals = compute_exit_signals(
                fused, prev_fused, temperature=self.temperature
            )
            last_score = torch.tensor(
                [signals.image_score] * image.shape[0],
                device=device,
                dtype=fused.dtype,
            )
            # Prefer batch-max of sigmoid for reported image_score tensor
            last_score = torch.sigmoid(fused / max(self.temperature, 1e-6)).flatten(1).max(
                dim=1
            ).values

            if depth in self.early_depths and not force_full_depth:
                t0 = tick()
                depth_id = torch.full(
                    (image.shape[0],), depth, dtype=torch.long, device=state.device
                )
                pred = self.lse(state, depth_id)
                timing["lse"] += _elapsed_ms(t0, tick()) if measure_timing else 0.0
                gains[depth] = GainPrediction(
                    mean=pred.mean.detach().cpu(),
                    log_variance=pred.log_variance.detach().cpu(),
                    sufficiency_logit=pred.sufficiency_logit.detach().cpu(),
                )
                t0 = tick()
                do_exit = bool(should_exit(pred, signals, self.profile))
                timing["policy"] += _elapsed_ms(t0, tick()) if measure_timing else 0.0
                exits[depth] = do_exit
                if do_exit:
                    selected = depth
                    prev_fused = fused
                    break

            prev_fused = fused
            selected = depth

        assert last_fused is not None
        if measure_timing:
            timing["total"] = _elapsed_ms(t_all0, tick())

        return AdaptiveResult(
            final_map=last_fused.squeeze(1).detach(),
            image_score=last_score.detach(),
            selected_depth=int(selected),
            checkpoint_trace=list(checkpoint_trace),
            weights=weights_by_depth,
            gain_predictions=gains,
            timing_breakdown=timing if measure_timing else {},
            exit_decisions=exits,
        )

    @torch.no_grad()
    def fuse_full_depth(self, image: torch.Tensor) -> AdaptiveResult:
        """Dynamic-fusion-only path: always run to full_depth with no early exit."""
        return self.infer(image, force_full_depth=True, measure_timing=False)

    @torch.no_grad()
    def timed_infer(
        self,
        image: torch.Tensor,
        *,
        force_full_depth: bool = False,
        warmup: int = 50,
        repetitions: int = 1,
    ) -> AdaptiveResult:
        """Warm up, then measure a timed inference (CUDA-synchronized segments)."""
        for _ in range(max(0, int(warmup))):
            self.infer(image, force_full_depth=force_full_depth, measure_timing=False)
        last: AdaptiveResult | None = None
        for _ in range(max(1, int(repetitions))):
            last = self.infer(
                image, force_full_depth=force_full_depth, measure_timing=True
            )
        assert last is not None
        return last
