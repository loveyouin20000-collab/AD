from __future__ import annotations

import torch

from rad.targets.residual_gain import (
    residual_gains,
    sufficiency,
    build_gain_target_record,
)


def test_residual_gain_formulas():
    errors = {
        12: torch.tensor([1.0, 0.8]),
        18: torch.tensor([0.6, 0.9]),
        24: torch.tensor([0.4, 0.5]),
    }
    gains = residual_gains(errors)
    # g18 = relu(E18 - E24)
    assert torch.allclose(gains[18], torch.tensor([0.2, 0.4]))
    # g12 = relu(E12 - min(E18, E24))
    assert torch.allclose(gains[12], torch.tensor([0.6, 0.3]))


def test_poor_but_equally_poor_full_depth_is_not_sufficient():
    """Equal high error at all depths => zero gain, but not sufficient."""
    poor = torch.tensor([2.0, 2.5])
    errors = {12: poor.clone(), 18: poor.clone(), 24: poor.clone()}
    gains = residual_gains(errors)
    assert torch.allclose(gains[12], torch.zeros(2))
    assert torch.allclose(gains[18], torch.zeros(2))

    # gain below epsilon, but absolute error is still high
    suf12 = sufficiency(
        gains[12],
        errors[12],
        epsilon_gain=0.05,
        epsilon_absolute=0.5,
    )
    suf18 = sufficiency(
        gains[18],
        errors[18],
        epsilon_gain=0.05,
        epsilon_absolute=0.5,
    )
    assert not bool(suf12.any())
    assert not bool(suf18.any())


def test_good_low_error_and_low_gain_is_sufficient():
    errors = {
        12: torch.tensor([0.1]),
        18: torch.tensor([0.08]),
        24: torch.tensor([0.07]),
    }
    gains = residual_gains(errors)
    assert bool(
        sufficiency(gains[12], errors[12], epsilon_gain=0.05, epsilon_absolute=0.5)
    )
    assert bool(
        sufficiency(gains[18], errors[18], epsilon_gain=0.05, epsilon_absolute=0.5)
    )


def test_build_record_stores_raw_errors_and_gains():
    errors = {
        12: torch.tensor(1.2),
        18: torch.tensor(0.9),
        24: torch.tensor(0.5),
    }
    rec = build_gain_target_record(
        errors,
        epsilon_gain=0.05,
        epsilon_absolute=0.5,
        early_depths=(12, 18),
    )
    assert set(rec["errors"].keys()) == {12, 18, 24}
    assert set(rec["gains"].keys()) == {12, 18}
    assert set(rec["sufficient"].keys()) == {12, 18}
    # Raw values preserved for recalibration
    assert float(rec["errors"][24]) == 0.5
    assert abs(float(rec["gains"][18]) - 0.4) < 1e-5  # relu(0.9-0.5)


def test_target_computation_stops_gradients():
    e12 = torch.tensor([1.0], requires_grad=True)
    e18 = torch.tensor([0.8], requires_grad=True)
    e24 = torch.tensor([0.3], requires_grad=True)
    gains = residual_gains({12: e12, 18: e18, 24: e24}, stop_gradient=True)
    assert not gains[12].requires_grad
    assert not gains[18].requires_grad
    # With stop_gradient=False, gains remain differentiable
    live = residual_gains({12: e12, 18: e18, 24: e24}, stop_gradient=False)
    assert live[12].requires_grad
    (live[12] + live[18]).sum().backward()
    assert e12.grad is not None and float(e12.grad.abs().sum()) > 0
