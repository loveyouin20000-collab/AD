"""RED/GREEN tests for B2-05A DLCM loss contracts."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from rad.phase_b import b2_dlcm as subject


def _manual_kl(p: torch.Tensor, log_w: torch.Tensor) -> torch.Tensor:
    # Sum over players; zero targets contribute exactly 0.
    contrib = torch.zeros((), dtype=torch.float32)
    for i in range(p.numel()):
        pi = p[i]
        if float(pi) > 0.0:
            contrib = contrib + pi * (torch.log(pi) - log_w[i])
    return contrib


def test_allocation_kl_exact_no_epsilon_zero_targets() -> None:
    w_logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    log_w = F.log_softmax(w_logits, dim=0)
    p = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32)
    kl = subject.allocation_kl(p.unsqueeze(0), w_logits.unsqueeze(0))
    expected = _manual_kl(p, log_w)
    assert torch.equal(kl, expected)
    # Cross-entropy form would differ by constant; contract reports KL.
    ce = -(p * log_w).sum()
    assert not torch.equal(kl, ce)


def test_allocation_depth_averages_gt_teacher_equally() -> None:
    logits = torch.zeros(1, 2)
    p_gt = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    p_t = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    loss, parts = subject.allocation_loss(logits, p_gt, p_t)
    kl_gt = subject.allocation_kl(p_gt, logits)
    kl_t = subject.allocation_kl(p_t, logits)
    assert torch.equal(parts["gt_kl"], kl_gt)
    assert torch.equal(parts["teacher_kl"], kl_t)
    assert torch.equal(loss, 0.5 * kl_gt + 0.5 * kl_t)


def test_huber_delta_one() -> None:
    pred = torch.tensor([0.0, 2.0, -3.0], dtype=torch.float32)
    target = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)
    # errors: -0.5, 2.0, -3.0 → 0.5*0.25, 2-0.5, 3-0.5 = 0.125, 1.5, 2.5
    loss = subject.huber_loss(pred.unsqueeze(0), target.unsqueeze(0), delta=1.0)
    expected = torch.tensor((0.125 + 1.5 + 2.5) / 3.0, dtype=torch.float32)
    assert torch.allclose(loss, expected)


def test_ranking_tolerance_softplus_and_no_pairs() -> None:
    pred = torch.tensor([[0.2, -0.1, 0.5]], dtype=torch.float32)
    # Differences within 1e-6 are ties.
    target = torch.tensor([[1.0, 1.0 + 5e-7, 0.0]], dtype=torch.float32)
    loss, meta = subject.pairwise_ranking_loss(
        pred, target, tie_tolerance=1e-6
    )
    # Valid unordered pairs: (0,2) and (1,2) only — |1-0|>1e-6 and |1+5e-7-0|>1e-6
    # pair (0,1) is a tie.
    assert meta["valid_pair_count"] == 2
    i, j = 0, 2
    s = math.copysign(1.0, float(target[0, i] - target[0, j]))
    l02 = F.softplus(torch.tensor(-s * (pred[0, i] - pred[0, j])))
    i, j = 1, 2
    s = math.copysign(1.0, float(target[0, i] - target[0, j]))
    l12 = F.softplus(torch.tensor(-s * (pred[0, i] - pred[0, j])))
    assert torch.allclose(loss, (l02 + l12) / 2.0)

    constant = torch.ones(1, 3)
    zero, meta0 = subject.pairwise_ranking_loss(pred, constant, tie_tolerance=1e-6)
    assert float(zero) == 0.0
    assert meta0["valid_pair_count"] == 0


def test_signed_loss_coefficients() -> None:
    pred = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    target = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    loss, parts = subject.signed_loss(
        pred, target, huber_delta=1.0, ranking_weight=0.25, tie_tolerance=1e-6
    )
    huber, _ = subject.huber_loss(pred, target, delta=1.0), None
    huber = subject.huber_loss(pred, target, delta=1.0)
    rank, _ = subject.pairwise_ranking_loss(pred, target, tie_tolerance=1e-6)
    assert torch.allclose(loss, huber + 0.25 * rank)
    assert torch.equal(parts["huber"], huber)
    assert torch.equal(parts["ranking"], rank)


def test_total_loss_depth_equal_and_signed_weight() -> None:
    # Synthetic one-sample batch with three depths.
    batch = {
        12: {
            "deployment_logits": torch.zeros(1, 2),
            "gt_signed": torch.zeros(1, 2),
            "teacher_signed": torch.zeros(1, 2),
            "p_gt": torch.tensor([[0.7, 0.3]], dtype=torch.float32),
            "p_t": torch.tensor([[0.4, 0.6]], dtype=torch.float32),
            "phi_gt": torch.tensor([[0.1, -0.1]], dtype=torch.float32),
            "phi_t": torch.tensor([[0.2, -0.2]], dtype=torch.float32),
        },
        18: {
            "deployment_logits": torch.zeros(1, 3),
            "gt_signed": torch.zeros(1, 3),
            "teacher_signed": torch.zeros(1, 3),
            "p_gt": torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32),
            "p_t": torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32),
            "phi_gt": torch.tensor([[0.1, 0.0, -0.1]], dtype=torch.float32),
            "phi_t": torch.tensor([[0.0, 0.1, -0.1]], dtype=torch.float32),
        },
        24: {
            "deployment_logits": torch.zeros(1, 4),
            "gt_signed": torch.zeros(1, 4),
            "teacher_signed": torch.zeros(1, 4),
            "p_gt": torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32),
            "p_t": torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32),
            "phi_gt": torch.tensor([[0.1, 0.05, -0.05, -0.1]], dtype=torch.float32),
            "phi_t": torch.tensor([[0.2, 0.0, 0.0, -0.2]], dtype=torch.float32),
        },
    }
    total, breakdown = subject.total_dlcm_loss(
        batch,
        signed_loss_weight=0.25,
        ranking_weight=0.25,
        huber_delta=1.0,
        tie_tolerance=1e-6,
        depth_weights={12: 1 / 3, 18: 1 / 3, 24: 1 / 3},
    )
    # Manual: each depth L_d = L_alloc + 0.25 * mean(L_signed_gt, L_signed_t)
    depth_losses = []
    for _depth, payload in batch.items():
        alloc, _ = subject.allocation_loss(
            payload["deployment_logits"], payload["p_gt"], payload["p_t"]
        )
        s_gt, _ = subject.signed_loss(
            payload["gt_signed"], payload["phi_gt"], huber_delta=1.0, ranking_weight=0.25
        )
        s_t, _ = subject.signed_loss(
            payload["teacher_signed"], payload["phi_t"], huber_delta=1.0, ranking_weight=0.25
        )
        depth_losses.append(alloc + 0.25 * 0.5 * (s_gt + s_t))
    expected = sum(depth_losses) / 3.0
    assert torch.allclose(total, expected)
    # No depth-24 implicit upweight: equal depth weights.
    assert breakdown["depth_weights"] == {12: pytest.approx(1 / 3), 18: pytest.approx(1 / 3), 24: pytest.approx(1 / 3)}


def test_gradient_boundaries_allocation_vs_signed() -> None:
    model = subject.B2DLCM(seed=17)
    x = torch.randn(2, 2, 18)
    p_gt = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
    p_t = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    phi_gt = torch.randn(2, 2)
    phi_t = torch.randn(2, 2)

    out = model.forward_training(x, prediction_depth=12)
    alloc, _ = subject.allocation_loss(out.deployment_logits, p_gt, p_t)
    model.zero_grad(set_to_none=True)
    alloc.backward(retain_graph=True)
    assert model.deployment_head.weight.grad is not None
    assert model.gt_signed_head.weight.grad is None
    assert model.teacher_signed_head.weight.grad is None
    assert model.layer_embedding.weight.grad is not None

    model.zero_grad(set_to_none=True)
    out = model.forward_training(x, prediction_depth=12)
    s_gt, _ = subject.signed_loss(out.gt_signed, phi_gt)
    s_gt.backward(retain_graph=True)
    assert model.gt_signed_head.weight.grad is not None
    assert model.deployment_head.weight.grad is None
    assert model.teacher_signed_head.weight.grad is None

    model.zero_grad(set_to_none=True)
    out = model.forward_training(x, prediction_depth=12)
    s_t, _ = subject.signed_loss(out.teacher_signed, phi_t)
    s_t.backward()
    assert model.teacher_signed_head.weight.grad is not None
    assert model.deployment_head.weight.grad is None
    assert model.gt_signed_head.weight.grad is None
