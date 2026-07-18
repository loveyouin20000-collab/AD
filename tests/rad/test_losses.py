from __future__ import annotations

import torch

from rad.losses.distillation import (
    confidence_weighted_distillation,
    normalized_binary_entropy,
)
from rad.losses.localization import sample_localization_error


def test_localization_error_finite_and_differentiable_on_normal_masks():
    logits = torch.randn(4, 1, 16, 16, requires_grad=True)
    mask = torch.zeros(4, 1, 16, 16)
    image_label = torch.zeros(4)
    err = sample_localization_error(logits, mask, image_label)
    assert err.shape == (4,)
    assert torch.isfinite(err).all()
    err.mean().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_localization_error_finite_on_one_pixel_and_full_masks():
    logits = torch.randn(3, 1, 8, 8, requires_grad=True)
    one_pixel = torch.zeros(3, 1, 8, 8)
    one_pixel[0, 0, 3, 4] = 1.0
    one_pixel[1, 0, 1, 1] = 1.0
    full = torch.ones(3, 1, 8, 8)
    labels = torch.tensor([1.0, 1.0, 1.0])
    for mask in (one_pixel, full):
        err = sample_localization_error(logits, mask, labels)
        assert err.shape == (3,)
        assert torch.isfinite(err).all()
    err.mean().backward()
    assert logits.grad is not None


def test_localization_uses_soft_components_not_ap_pro():
    # Sanity: zero logits vs empty mask should be finite positive BCE-dominated error
    logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
    mask = torch.zeros(2, 1, 4, 4)
    labels = torch.zeros(2)
    err = sample_localization_error(logits, mask, labels)
    assert torch.isfinite(err).all()
    # SoftDice on empty/empty is near 0; BCEWithLogits(0,0)=ln(2)
    assert (err > 0).all()


def test_distillation_weights_by_one_minus_normalized_entropy():
    teacher = torch.tensor([[[[0.0, 10.0], [10.0, 0.0]]]])  # [1,1,2,2]
    student = torch.zeros(1, 1, 2, 2, requires_grad=True)
    # High |logit| -> low entropy -> high weight; logit 0 -> max entropy -> weight 0
    ent = normalized_binary_entropy(torch.sigmoid(teacher))
    assert torch.allclose(ent[0, 0, 0, 0], torch.tensor(1.0), atol=1e-5)
    assert ent[0, 0, 0, 1] < 0.1
    loss = confidence_weighted_distillation(student, teacher)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    # Near-zero-entropy teacher pixels should dominate the gradient
    assert student.grad[0, 0, 0, 1].abs() > student.grad[0, 0, 0, 0].abs()


def test_distillation_rejects_shape_mismatch():
    student = torch.randn(2, 1, 4, 4)
    teacher = torch.randn(2, 1, 8, 8)
    try:
        confidence_weighted_distillation(student, teacher)
        assert False, "expected ValueError"
    except ValueError:
        pass
