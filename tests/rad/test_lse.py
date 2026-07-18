from __future__ import annotations

import torch

from rad.models.lse import LSE, GainPrediction, lse_loss


def test_forward_shapes_and_constraints():
    model = LSE(state_dim=26, early_depths=(12, 18))
    b = 4
    state = torch.randn(b, 26)
    depth_id = torch.tensor([12, 18, 12, 18], dtype=torch.long)
    out = model(state, depth_id)
    assert isinstance(out, GainPrediction)
    assert out.mean.shape == (b,)
    assert out.log_variance.shape == (b,)
    assert out.sufficiency_logit.shape == (b,)
    assert torch.all(out.mean >= 0)
    assert torch.all(out.log_variance >= -8)
    assert torch.all(out.log_variance <= 4)


def test_batch_shape_stable_across_depths():
    model = LSE(state_dim=16, early_depths=(12, 18))
    state = torch.randn(3, 16)
    for depths in ([12, 12, 12], [18, 18, 18], [12, 18, 12]):
        out = model(state, torch.tensor(depths, dtype=torch.long))
        assert out.mean.shape == (3,)
        assert out.log_variance.shape == (3,)
        assert out.sufficiency_logit.shape == (3,)


def test_softplus_mean_and_clamped_log_variance():
    model = LSE(state_dim=8, early_depths=(12, 18))
    # Force extreme raw outputs via bias if possible after init
    with torch.no_grad():
        model.mean_head.bias.fill_(10.0)
        model.logvar_head.bias.fill_(100.0)
    state = torch.zeros(2, 8)
    depth = torch.tensor([12, 18], dtype=torch.long)
    out = model(state, depth)
    assert torch.all(out.mean > 0)
    assert torch.all(out.log_variance <= 4.0 + 1e-5)
    with torch.no_grad():
        model.logvar_head.bias.fill_(-100.0)
    out2 = model(state, depth)
    assert torch.all(out2.log_variance >= -8.0 - 1e-5)


def test_lse_loss_heteroscedastic_nll_and_sufficiency():
    pred = GainPrediction(
        mean=torch.tensor([1.0, 0.5]),
        log_variance=torch.tensor([0.0, -1.0]),
        sufficiency_logit=torch.tensor([2.0, -2.0]),
    )
    target_gain = torch.tensor([1.0, 0.5])
    target_suf = torch.tensor([1.0, 0.0])
    loss = lse_loss(pred, target_gain, target_suf)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Perfect mean + confident correct logits should be lower than flipped labels
    good = lse_loss(pred, target_gain, target_suf)
    bad = lse_loss(pred, target_gain, 1.0 - target_suf)
    assert float(good) < float(bad)


def test_gradients_reach_mlp_and_depth_embedding():
    model = LSE(state_dim=10, early_depths=(12, 18))
    state = torch.randn(2, 10, requires_grad=True)
    depth = torch.tensor([12, 18], dtype=torch.long)
    out = model(state, depth)
    loss = out.mean.sum() + out.log_variance.sum() + out.sufficiency_logit.sum()
    loss.backward()
    assert state.grad is not None and float(state.grad.abs().sum()) > 0
    assert model.depth_emb.weight.grad is not None
    assert float(model.depth_emb.weight.grad.abs().sum()) > 0
