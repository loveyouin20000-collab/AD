from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rad.calibration.policy_search import search_policy_profiles
from rad.calibration.temperature import apply_temperature, fit_temperature
from rad.models.lse import GainPrediction
from rad.models.policy import (
    ExitSignals,
    PolicyProfile,
    gain_ucb,
    should_exit,
)


def _pred(mean: float, log_var: float = 0.0) -> GainPrediction:
    return GainPrediction(
        mean=torch.tensor([mean]),
        log_variance=torch.tensor([log_var]),
        sufficiency_logit=torch.tensor([0.0]),
    )


def test_ambiguous_image_predictions_continue():
    profile = PolicyProfile(
        name="balanced",
        gain_threshold=0.1,
        kappa=1.0,
        map_uncertainty_threshold=0.5,
        image_confidence_margin=0.3,
        stability_threshold=0.2,
        require_map_uncertainty=True,
        require_image_confidence=True,
        require_stability=False,
    )
    # Low predicted gain UCB, low map uncertainty, but ambiguous image score ~0.5
    signals = ExitSignals(
        map_uncertainty=0.1,
        image_score=0.52,
        stability=0.05,
    )
    assert should_exit(_pred(0.01), signals, profile) is False


def test_confidently_normal_and_anomalous_may_exit():
    profile = PolicyProfile(
        name="balanced",
        gain_threshold=0.1,
        kappa=1.0,
        map_uncertainty_threshold=0.5,
        image_confidence_margin=0.3,
        stability_threshold=0.2,
        require_map_uncertainty=True,
        require_image_confidence=True,
        require_stability=False,
    )
    low_gain = _pred(0.01, log_var=-8.0)
    assert should_exit(
        low_gain,
        ExitSignals(map_uncertainty=0.05, image_score=0.05, stability=0.01),
        profile,
    )
    assert should_exit(
        low_gain,
        ExitSignals(map_uncertainty=0.05, image_score=0.95, stability=0.01),
        profile,
    )


def test_aggressive_uses_gain_ucb_only():
    aggressive = PolicyProfile.aggressive(gain_threshold=0.05, kappa=1.0)
    # Ambiguous image / high map unc / unstable — still exit if UCB low
    signals = ExitSignals(map_uncertainty=0.9, image_score=0.5, stability=0.9)
    assert should_exit(_pred(0.0, log_var=-8.0), signals, aggressive)
    # High residual gain UCB blocks exit
    assert should_exit(_pred(1.0, log_var=0.0), signals, aggressive) is False


def test_conservative_requires_stability():
    cons = PolicyProfile.conservative(
        gain_threshold=0.1,
        kappa=1.0,
        map_uncertainty_threshold=0.5,
        image_confidence_margin=0.3,
        stability_threshold=0.1,
    )
    pred = _pred(0.0, log_var=-8.0)
    good = ExitSignals(map_uncertainty=0.05, image_score=0.9, stability=0.01)
    unstable = ExitSignals(map_uncertainty=0.05, image_score=0.9, stability=0.5)
    assert should_exit(pred, good, cons)
    assert should_exit(pred, unstable, cons) is False


def test_gain_ucb_formula():
    pred = _pred(1.0, log_var=0.0)  # std = 1
    assert float(gain_ucb(pred, kappa=2.0)[0]) == pytest.approx(3.0, abs=1e-5)


def test_temperature_improves_or_matches_nll():
    torch.manual_seed(0)
    logits = torch.randn(64, 1, 8, 8)
    # Correlated labels
    labels = (torch.sigmoid(logits) > 0.5).float()
    t = fit_temperature(logits, labels)
    assert t > 0
    before = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
    after = float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            apply_temperature(logits, t), labels
        )
    )
    assert after <= before + 1e-5


def test_policy_search_returns_three_profiles_and_pareto():
    # Tiny synthetic table: rows with metrics for candidate thresholds
    candidates = [
        {"gain_threshold": 0.05, "kappa": 1.0, "map_uncertainty_threshold": 0.5,
         "image_confidence_margin": 0.3, "stability_threshold": 0.1,
         "pixel_ap_drop": 0.01, "false_safe_exit_rate": 0.02, "expected_depth": 16.0},
        {"gain_threshold": 0.2, "kappa": 0.5, "map_uncertainty_threshold": 0.8,
         "image_confidence_margin": 0.1, "stability_threshold": 0.5,
         "pixel_ap_drop": 0.05, "false_safe_exit_rate": 0.2, "expected_depth": 12.0},
        {"gain_threshold": 0.1, "kappa": 1.5, "map_uncertainty_threshold": 0.4,
         "image_confidence_margin": 0.25, "stability_threshold": 0.15,
         "pixel_ap_drop": 0.02, "false_safe_exit_rate": 0.05, "expected_depth": 14.0},
    ]
    result = search_policy_profiles(
        candidates,
        max_pixel_ap_drop=0.03,
        max_false_safe_exit_rate=0.1,
    )
    assert "pareto" in result and len(result["pareto"]) >= 1
    assert set(result["profiles"]) >= {"conservative", "balanced", "aggressive"}
    for name in ("conservative", "balanced", "aggressive"):
        assert isinstance(result["profiles"][name], PolicyProfile)


def test_calibrate_cli_refuses_target_dataset_path(tmp_path: Path):
    from tools.calibrate_policy import assert_no_target_dataset_path

    with pytest.raises(SystemExit, match="target-dataset"):
        assert_no_target_dataset_path(
            tmp_path / "visa_target_cache",
            source_dataset="mvtec",
            target_datasets=("visa",),
        )
    # Source calibration path is allowed
    assert_no_target_dataset_path(
        "artifacts/cache/mvtec_calibration",
        source_dataset="mvtec",
        target_datasets=("visa",),
    )
