"""Relative Smooth-Max unit tests for V4."""

from __future__ import annotations

import math

import pytest
import torch

from rad.phase_b import b2_dlcm_v4 as v3


def _direct(losses: list[float], tau: float = 0.05) -> float:
    m = max(losses)
    return m + tau * math.log(sum(math.exp((x - m) / tau) for x in losses) / len(losses))


def test_equal_losses_return_common_value() -> None:
    out = v3.smooth_max_normalized([0.3, 0.3], tau=0.05)
    assert float(out) == pytest.approx(0.3)


def test_matches_direct_formula() -> None:
    vals = [0.1, 0.4]
    out = float(v3.smooth_max_normalized(vals, tau=0.05))
    # float32 return vs float64 reference; require tight but practical tolerance.
    assert out == pytest.approx(_direct(vals), rel=0, abs=1e-6)


def test_numerical_stability_large_gap() -> None:
    out = float(v3.smooth_max_normalized([100.0, 0.0], tau=0.05))
    assert math.isfinite(out)
    assert out == pytest.approx(_direct([100.0, 0.0]), rel=0, abs=1e-4)


def test_worse_category_has_larger_gradient() -> None:
    lb = torch.tensor(0.1, requires_grad=True)
    lc = torch.tensor(0.5, requires_grad=True)
    loss = v3._relative_smooth_max_from_tensors(lb, lc, tau=0.05)
    loss.backward()
    assert lc.grad is not None and lb.grad is not None
    assert float(lc.grad) > float(lb.grad)


def test_tau_fixed_and_not_hard_max() -> None:
    soft = float(v3.smooth_max_normalized([0.1, 0.5], tau=0.05))
    assert soft < 0.5
    assert soft > 0.1
    assert abs(v3.SMOOTHMAX_TAU - 0.05) < 1e-15


def test_invalid_tau_fails() -> None:
    with pytest.raises(v3.B2DLCMV4Error) as exc:
        v3.smooth_max_normalized([0.1, 0.2], tau=0.0)
    assert exc.value.code == "B2_DLCM_RELATIVE_SMOOTHMAX_INVALID"


def test_no_groupdro_mutable_state() -> None:
    assert not hasattr(v3, "GroupDROState")
    assert not hasattr(v3, "category_weights")


def test_negative_regret_allowed_and_equal() -> None:
    out = v3.relative_smooth_max_normalized([-0.2, -0.2], tau=0.05)
    assert float(out) == pytest.approx(-0.2)


def test_negative_regrets_match_formula() -> None:
    vals = [-0.3, -0.05]
    out = float(v3.relative_smooth_max_normalized(vals, tau=0.05))
    assert out == pytest.approx(_direct(vals), rel=0, abs=1e-6)
