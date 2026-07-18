from __future__ import annotations

import torch

from rad.models.dlcm import DLCM, sum_preserving_fusion


def test_invalid_layers_receive_exactly_zero_weight():
    model = DLCM(max_layer_id=24)
    b, l = 2, 4
    layer_desc = torch.randn(b, l, 18)
    ctx = torch.randn(b, 8)
    layer_ids = torch.tensor([[6, 12, 18, 24], [6, 12, 18, 24]])
    valid = torch.tensor(
        [[True, True, False, True], [True, False, True, True]],
        dtype=torch.bool,
    )
    weights = model(layer_desc, ctx, layer_ids, valid)
    assert weights.shape == (b, l)
    assert torch.all(weights[~valid] == 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(b), atol=1e-5)


def test_valid_weights_sum_to_one():
    model = DLCM(max_layer_id=24, alpha=0.0)
    layer_desc = torch.randn(3, 3, 18)
    ctx = torch.randn(3, 8)
    layer_ids = torch.tensor([[6, 12, 18], [6, 12, 18], [6, 12, 18]])
    valid = torch.ones(3, 3, dtype=torch.bool)
    weights = model(layer_desc, ctx, layer_ids, valid)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-5)


def test_zero_initialized_scorer_reproduces_equal_fusion():
    model = DLCM(max_layer_id=24, alpha=0.0)
    # Scorer must be zero-init at construction
    assert torch.all(model.scorer.weight == 0)
    assert torch.all(model.scorer.bias == 0)

    b, l, h, w = 2, 3, 8, 8
    maps = torch.randn(b, l, 1, h, w)
    layer_desc = torch.randn(b, l, 18)
    ctx = torch.randn(b, 8)
    layer_ids = torch.tensor([[6, 12, 18], [6, 12, 18]])
    valid = torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool)

    weights = model(layer_desc, ctx, layer_ids, valid)
    # Equal over valid: 1/n_valid
    n_valid = valid.sum(dim=1, keepdim=True).clamp_min(1).to(weights.dtype)
    expected_w = valid.to(weights.dtype) / n_valid
    assert torch.allclose(weights, expected_w, atol=1e-5)

    fused = sum_preserving_fusion(maps, weights, valid)
    # Equal fusion == sum of valid maps
    equal_sum = (maps * valid.to(maps.dtype)[:, :, None, None, None]).sum(dim=1)
    assert torch.allclose(fused, equal_sum, atol=1e-5)


def test_gradients_reach_descriptors_and_scorer():
    model = DLCM(max_layer_id=24, alpha=0.0)
    # Zero-init blocks descriptor grads by design; nudge scorer so the path is live.
    with torch.no_grad():
        model.scorer.weight.normal_(0, 0.01)
        model.scorer.bias.normal_(0, 0.01)
    layer_desc = torch.randn(2, 3, 18, requires_grad=True)
    ctx = torch.randn(2, 8, requires_grad=True)
    layer_ids = torch.tensor([[6, 12, 18], [6, 12, 18]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    maps = torch.randn(2, 3, 1, 4, 4)

    weights = model(layer_desc, ctx, layer_ids, valid)
    fused = sum_preserving_fusion(maps, weights, valid)
    loss = fused.pow(2).mean()
    loss.backward()

    assert layer_desc.grad is not None and layer_desc.grad.abs().sum() > 0
    assert ctx.grad is not None and ctx.grad.abs().sum() > 0
    assert model.scorer.weight.grad is not None
    assert model.scorer.weight.grad.abs().sum() > 0


def test_alpha_floor_mixes_toward_uniform():
    model = DLCM(max_layer_id=24, alpha=0.1)
    # Break symmetry by setting scorer after init
    with torch.no_grad():
        model.scorer.weight.fill_(1.0)
        model.scorer.bias.fill_(0.0)
    layer_desc = torch.randn(1, 2, 18)
    ctx = torch.randn(1, 8)
    layer_ids = torch.tensor([[6, 12]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    weights = model(layer_desc, ctx, layer_ids, valid)
    # With alpha>0, weights are convex mix with uniform 0.5
    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-5)
    # Not pure one-hot even if logits differ strongly
    assert weights.min() >= 0.1 / 2 - 1e-5


def test_alpha_schedule_decays_over_first_20_percent():
    model = DLCM(max_layer_id=24)
    assert abs(model.alpha_for_progress(0.0) - 0.1) < 1e-6
    assert abs(model.alpha_for_progress(0.1) - 0.05) < 1e-6
    assert abs(model.alpha_for_progress(0.2) - 0.0) < 1e-6
    assert abs(model.alpha_for_progress(0.5) - 0.0) < 1e-6
