"""Batch-matched uniform-relative regret tests for V4."""

from __future__ import annotations

import math
import struct

import pytest
import torch

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_v4 as v4
from rad.phase_b import b2_dlcm_v4_training as training


def _batch() -> tuple[list[dict], list[str], v4.B2DLCMV4]:
    records = training.build_hermetic_v4_records()
    train = [r for r in records if r["split"] == "training"]
    batch = [r for r in train if r["category"] == "bottle"][:2] + [
        r for r in train if r["category"] == "carpet"
    ][:2]
    cats = [r["category"] for r in batch]
    model = v4.B2DLCMV4(seed=17)
    return batch, cats, model


def test_model_and_uniform_share_same_targets_and_batch() -> None:
    batch, cats, model = _batch()
    p = torch.stack([r["p_gt"][24] for r in batch], dim=0)
    descs = torch.stack([r["descriptors"][24] for r in batch], dim=0)
    out = model.forward_training(descs, prediction_depth=24)
    matched = v4.batch_matched_relative_regrets(p, out.gt_deployment_logits, cats)
    # Reconstruct category means independently and confirm identity.
    model_kl = v4.category_mean_allocation_kl(p, out.gt_deployment_logits, cats)
    uni = v4.frozen_uniform_logits(p.shape[0], p.shape[-1], device=p.device)
    uni_kl = v4.category_mean_allocation_kl(p, uni, cats)
    for cat in ("bottle", "carpet"):
        assert float(matched["model_kl"][cat]) == pytest.approx(float(model_kl[cat]), abs=0)
        assert float(matched["uniform_kl"][cat]) == pytest.approx(float(uni_kl[cat]), abs=0)
        expected = float(model_kl[cat]) - float(uni_kl[cat])
        got = float(matched["regrets"][cat])
        assert got == pytest.approx(expected, abs=0)
        # IEEE float32 bit identity for direct subtraction on detached scalars.
        a = torch.tensor(float(model_kl[cat]), dtype=torch.float32)
        b = torch.tensor(float(uni_kl[cat]), dtype=torch.float32)
        bits_ref = struct.unpack(">I", struct.pack(">f", float(a - b)))[0]
        bits_got = struct.unpack(">I", struct.pack(">f", got))[0]
        assert bits_got == bits_ref


def test_no_slack_clamp_or_abs_on_negative_regret() -> None:
    # Force model better than uniform → negative regret.
    p = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]], dtype=torch.float32)
    # Logits matching p closely → low model KL; uniform worse → negative regret.
    model_logits = torch.log(p.clamp_min(1e-8))
    cats = ["bottle", "bottle", "carpet", "carpet"]
    matched = v4.batch_matched_relative_regrets(p, model_logits, cats)
    for cat in ("bottle", "carpet"):
        r = float(matched["regrets"][cat])
        assert r < 0.0
        assert math.isfinite(r)
        # Prove we did not abs/clamp: raw subtraction equals stored regret.
        raw = float(matched["model_kl"][cat] - matched["uniform_kl"][cat])
        assert r == pytest.approx(raw, abs=0)


def test_uniform_baseline_matches_reference_weights() -> None:
    logits = v4.frozen_uniform_logits(3, 4, device=torch.device("cpu"))
    weights = torch.softmax(logits, dim=-1)
    ref = v1.reference_uniform_weights(4, dtype=torch.float32)
    assert bool(torch.equal(weights[0], ref))


def test_relative_smooth_max_worse_regret_larger_grad() -> None:
    rb = torch.tensor(-0.1, requires_grad=True)
    rc = torch.tensor(0.2, requires_grad=True)
    loss = v4._relative_smooth_max_from_tensors(rb, rc, tau=0.05)
    loss.backward()
    assert rc.grad is not None and rb.grad is not None
    assert float(rc.grad) > float(rb.grad)
