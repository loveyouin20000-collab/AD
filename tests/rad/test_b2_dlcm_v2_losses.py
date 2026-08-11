"""RED/GREEN tests for B2-05C1 V2 losses and gradient isolation."""

from __future__ import annotations

import torch

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_v2 as subject


def test_loss_coefficients_exact() -> None:
    assert subject.TEACHER_ALLOC_WEIGHT == 0.25
    assert subject.GT_SIGNED_WEIGHT == 0.25
    assert subject.TEACHER_SIGNED_WEIGHT == 0.0625
    n = 2
    p = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    phi = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    g_logits = torch.zeros(1, n, requires_grad=True)
    t_logits = torch.zeros(1, n, requires_grad=True)
    g_signed = torch.zeros(1, n, requires_grad=True)
    t_signed = torch.zeros(1, n, requires_grad=True)
    single = {
        12: {
            "gt_deployment_logits": g_logits,
            "teacher_allocation_logits": t_logits,
            "gt_signed": g_signed,
            "teacher_signed": t_signed,
            "p_gt": p,
            "p_t": p.flip(-1),
            "phi_gt": phi,
            "phi_t": -phi,
        }
    }
    loss, parts = subject.total_dlcm_v2_loss(single, depth_weights={12: 1.0})
    gt_kl = v1.allocation_kl(p, g_logits)
    t_kl = v1.allocation_kl(p.flip(-1), t_logits)
    s_gt, _ = v1.signed_loss(g_signed, phi)
    s_t, _ = v1.signed_loss(t_signed, -phi)
    expected = gt_kl + 0.25 * s_gt + 0.25 * t_kl + 0.0625 * s_t
    assert torch.allclose(loss, expected)
    assert torch.allclose(parts["depths"][12]["gt_deploy_kl"], gt_kl)
    assert torch.allclose(parts["depths"][12]["teacher_alloc_kl"], t_kl)


def test_target_head_boundaries_in_total_loss() -> None:
    model = subject.B2DLCMV2(seed=17)
    model.train()
    n = 2
    x = torch.randn(2, n, 18)
    out = model.forward_training(x, prediction_depth=12)
    p_gt = torch.softmax(torch.randn(2, n), dim=-1)
    p_t = torch.softmax(torch.randn(2, n), dim=-1)
    phi_gt = torch.randn(2, n)
    phi_t = torch.randn(2, n)
    loss, _ = subject.total_dlcm_v2_loss(
        {
            12: {
                "gt_deployment_logits": out.gt_deployment_logits,
                "teacher_allocation_logits": out.teacher_allocation_logits,
                "gt_signed": out.gt_signed,
                "teacher_signed": out.teacher_signed,
                "p_gt": p_gt,
                "p_t": p_t,
                "phi_gt": phi_gt,
                "phi_t": phi_t,
            }
        },
        depth_weights={12: 1.0},
    )
    assert bool(torch.isfinite(loss))


def test_actual_gradient_isolation_matrix() -> None:
    model = subject.B2DLCMV2(seed=29)
    matrix = subject.probe_gradient_isolation(model, prediction_depth=12)
    assert matrix["gt_deploy"]["gt_deployment_head"] is True
    assert matrix["gt_deploy"]["shared_trunk"] is True
    assert matrix["gt_deploy"]["teacher_allocation_head"] is False
    assert matrix["gt_deploy"]["gt_signed_head"] is False
    assert matrix["gt_deploy"]["teacher_signed_head"] is False

    assert matrix["teacher_alloc"]["teacher_allocation_head"] is True
    assert matrix["teacher_alloc"]["shared_trunk"] is True
    assert matrix["teacher_alloc"]["gt_deployment_head"] is False
    assert matrix["teacher_alloc"]["gt_signed_head"] is False
    assert matrix["teacher_alloc"]["teacher_signed_head"] is False

    assert matrix["gt_signed"]["gt_signed_head"] is True
    assert matrix["gt_signed"]["shared_trunk"] is True
    assert matrix["gt_signed"]["gt_deployment_head"] is False
    assert matrix["gt_signed"]["teacher_allocation_head"] is False
    assert matrix["gt_signed"]["teacher_signed_head"] is False

    assert matrix["teacher_signed"]["teacher_signed_head"] is True
    assert matrix["teacher_signed"]["shared_trunk"] is True
    assert matrix["teacher_signed"]["gt_deployment_head"] is False
    assert matrix["teacher_signed"]["teacher_allocation_head"] is False
    assert matrix["teacher_signed"]["gt_signed_head"] is False
