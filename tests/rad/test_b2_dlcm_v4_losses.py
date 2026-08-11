"""V3 loss aggregation tests."""

from __future__ import annotations

import torch

from rad.phase_b import b2_dlcm_v4 as v3
from rad.phase_b import b2_dlcm_v4_training as training


def _batch_payload(model: v3.B2DLCMV4, records: list[dict], categories: list[str]):
    depth_payload = {}
    for depth in model.prediction_depths:
        descs = torch.stack([r["descriptors"][depth] for r in records], dim=0)
        out = model.forward_training(descs, prediction_depth=depth)
        depth_payload[depth] = {
            "gt_deployment_logits": out.gt_deployment_logits,
            "teacher_allocation_logits": out.teacher_allocation_logits,
            "gt_signed": out.gt_signed,
            "teacher_signed": out.teacher_signed,
            "p_gt": torch.stack([r["p_gt"][depth] for r in records], dim=0),
            "p_t": torch.stack([r["p_t"][depth] for r in records], dim=0),
            "phi_gt": torch.stack([r["phi_gt"][depth] for r in records], dim=0),
            "phi_t": torch.stack([r["phi_t"][depth] for r in records], dim=0),
        }
    return depth_payload


def test_only_gt_deployment_uses_relative_smooth_max() -> None:
    records = training.build_hermetic_v4_records()
    train = [r for r in records if r["split"] == "training"]
    bottle = [r for r in train if r["category"] == "bottle"][:2]
    carpet = [r for r in train if r["category"] == "carpet"][:2]
    batch = bottle + carpet
    cats = [r["category"] for r in batch]
    model = v3.B2DLCMV4(seed=17)
    payload = _batch_payload(model, batch, cats)
    _loss, parts = v3.total_dlcm_v4_loss(payload, categories=cats)
    assert parts["aggregation"]["gt_deployment"] == "uniform_relative_smooth_max"
    assert parts["aggregation"]["teacher_allocation"] == "sample_mean"
    assert parts["aggregation"]["gt_signed"] == "sample_mean"
    assert parts["aggregation"]["teacher_signed"] == "sample_mean"
    assert abs(parts["tau"] - 0.05) < 1e-15


def test_coefficients_and_depth_equal_weight() -> None:
    assert v3.TEACHER_ALLOC_WEIGHT == 0.25
    assert v3.GT_SIGNED_WEIGHT == 0.25
    assert v3.TEACHER_SIGNED_WEIGHT == 0.0625
    records = training.build_hermetic_v4_records()
    train = [r for r in records if r["split"] == "training"]
    batch = [r for r in train if r["category"] == "bottle"][:2] + [
        r for r in train if r["category"] == "carpet"
    ][:2]
    model = v3.B2DLCMV4(seed=29)
    payload = _batch_payload(model, batch, [r["category"] for r in batch])
    _loss, parts = v3.total_dlcm_v4_loss(payload, categories=[r["category"] for r in batch])
    weights = [parts["depths"][d]["weight"] for d in (12, 18, 24)]
    assert all(abs(w - 1 / 3) < 1e-12 for w in weights)


def test_category_not_model_input() -> None:
    model = v3.B2DLCMV4(seed=17)
    records = training.build_hermetic_v4_records()
    r = next(x for x in records if x["split"] == "training")
    desc = r["descriptors"][24].unsqueeze(0)
    out = model.forward_training(desc, prediction_depth=24)
    assert out.gt_deployment_logits.shape[0] == 1
    # forward_training signature has no category argument
    import inspect

    sig = inspect.signature(model.forward_training)
    assert "category" not in sig.parameters
