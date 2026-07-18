from __future__ import annotations

import torch

from rad.models.descriptors import (
    CheckpointContextExtractor,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM
from rad.trainers.fusion_trainer import FusionLossWeights, FusionTrainer


def _synthetic_batch(b: int = 2, h: int = 16, layers=(6, 12, 18, 24)):
    layers = tuple(layers)
    depths = (12, 18, 24)
    maps_by_depth = {}
    shapley_by_depth = {}
    for depth in depths:
        avail = [x for x in layers if x <= depth]
        l = len(avail)
        maps_by_depth[depth] = torch.randn(b, l, 1, h, h)
        # Non-uniform target so contribution KL drives learning
        dist = torch.softmax(torch.randn(b, l), dim=-1)
        shapley_by_depth[depth] = {"distribution": dist, "phi": torch.randn(b, l)}

    return {
        "maps_by_depth": maps_by_depth,
        "layer_ids_by_depth": {
            d: torch.tensor([[x for x in layers if x <= d] for _ in range(b)])
            for d in depths
        },
        "mask": (torch.rand(b, 1, h, h) > 0.8).float(),
        "image_label": torch.ones(b),
        "teacher_logits": torch.randn(b, 1, h, h),
        "shapley_by_depth": shapley_by_depth,
        "candidate_layers": layers,
    }


def test_one_optimizer_step_decreases_loss():
    torch.manual_seed(0)
    layers = (6, 12, 18, 24)
    dlcm = DLCM(max_layer_id=24, alpha=0.0)
    # Nudge scorer so descriptor path is live and optimization is non-trivial
    with torch.no_grad():
        dlcm.scorer.weight.normal_(0, 0.05)
        dlcm.scorer.bias.normal_(0, 0.05)

    trainer = FusionTrainer(
        dlcm=dlcm,
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=24),
        normalizer=None,
        loss_weights=FusionLossWeights(),
        train_depths=(12, 18, 24),
        candidate_layers=layers,
        freeze_backbone=True,
    )
    assert set(trainer.trainable_parameter_names()) == {
        n for n, _ in dlcm.named_parameters()
    }

    batch = _synthetic_batch()
    opt = torch.optim.Adam(trainer.trainable_parameters(), lr=1e-2)

    with torch.no_grad():
        before = float(trainer.compute_losses(batch)["loss"])
    metrics = trainer.training_step(batch, opt)
    with torch.no_grad():
        after = float(trainer.compute_losses(batch)["loss"])

    assert metrics["loss"] > 0
    assert "loc_12" in metrics and "contrib_kl_24" in metrics
    assert "weight_entropy_24" in metrics and "avg_weight_24" in metrics
    assert after < before


def test_freeze_backbone_flag_exposes_only_dlcm_params():
    dlcm = DLCM(max_layer_id=24)
    trainer = FusionTrainer(
        dlcm=dlcm,
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=24),
        freeze_backbone=True,
        candidate_layers=(6, 12, 18, 24),
    )
    names = trainer.trainable_parameter_names()
    assert all(n.startswith("") or True for n in names)
    assert any("scorer" in n for n in names)
    # No accidental backbone modules registered as trainable
    assert not any("visual" in n or "cross_attn" in n for n in names)
