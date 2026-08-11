"""V5 weight mix tests."""

from __future__ import annotations

import torch

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_v5 as v5


def test_beta0_exact_uniform() -> None:
    dyn = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32)
    mixed = v5.mix_uniform_anchored_weights(dyn, 0.0)
    uni = v5.depth_matched_uniform(3)
    assert torch.equal(mixed, uni)
    assert torch.equal(mixed, v1.reference_uniform_weights(3))


def test_beta1_exact_dynamic() -> None:
    dyn = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32)
    mixed = v5.mix_uniform_anchored_weights(dyn, 1.0)
    assert torch.equal(mixed, dyn)


def test_intermediate_convex_fp32() -> None:
    dyn = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32)
    beta = 0.37
    uni = v5.depth_matched_uniform(3)
    mixed = v5.mix_uniform_anchored_weights(dyn, beta)
    expected = (1.0 - beta) * uni + beta * dyn
    assert torch.allclose(mixed, expected, rtol=0, atol=0)
    assert bool(torch.all(mixed >= 0))
    assert abs(float(mixed.sum()) - 1.0) < 1e-6


def test_batch_mix_shape() -> None:
    dyn = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float32)
    mixed = v5.mix_uniform_anchored_weights(dyn, 0.5)
    assert mixed.shape == dyn.shape
    assert torch.allclose(mixed.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_category_not_accepted_by_mix_api() -> None:
    # mix API has no category parameter — calling with unexpected kw should fail via TypeError
    dyn = torch.tensor([0.5, 0.5], dtype=torch.float32)
    try:
        v5.mix_uniform_anchored_weights(dyn, 0.5, category="carpet")  # type: ignore[call-arg]
        raised = False
    except TypeError:
        raised = True
    assert raised
