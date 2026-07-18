from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rad.evaluation.export import (
    TransferSamplePrediction,
    export_transfer_predictions,
)
from rad.evaluation.zero_shot import (
    TargetAccessError,
    assert_policy_unchanged,
    compute_stratified_metrics,
    compute_transfer_metrics,
    forbid_target_access_during_calibration,
    load_frozen_policy_profile,
    pixel_average_precision,
    boundary_f_score,
    pro_score_proxy,
)


def test_forbid_target_access_during_calibration_monkeypatch():
    """Calibration must fail if any target sample path/id is touched."""
    source = "mvtec"
    targets = ("visa",)

    with forbid_target_access_during_calibration(
        source_dataset=source,
        target_datasets=targets,
    ) as guard:
        # Source paths are allowed
        guard.check_path("/root/autodl-tmp/data/mvtec/bottle/train/good/000.png")
        guard.check_sample_id("mvtec/bottle/good/000")

        with pytest.raises(TargetAccessError, match="visa"):
            guard.check_path("/root/autodl-tmp/data/Visa/candle/test/bad/000.png")

        with pytest.raises(TargetAccessError, match="visa"):
            guard.check_sample_id("visa/candle/bad/000")


def test_load_frozen_policy_is_unchanged(tmp_path: Path):
    profiles = {
        "profiles": {
            "balanced": {
                "name": "balanced",
                "gain_threshold": 0.02,
                "kappa": 0.5,
                "map_uncertainty_threshold": 0.5,
                "image_confidence_margin": 0.4,
                "stability_threshold": 1.0,
                "require_map_uncertainty": True,
                "require_image_confidence": True,
                "require_stability": False,
            }
        },
        "schema_version": 1,
        "seed": 111,
    }
    path = tmp_path / "policy_profiles.json"
    path.write_text(json.dumps(profiles))
    profile, digest = load_frozen_policy_profile(path, "balanced")
    assert profile.name == "balanced"
    assert_policy_unchanged(path, "balanced", digest)
    # Mutating file must be detected
    profiles["profiles"]["balanced"]["gain_threshold"] = 0.99
    path.write_text(json.dumps(profiles))
    with pytest.raises(ValueError, match="changed"):
        assert_policy_unchanged(path, "balanced", digest)


def test_transfer_metrics_and_stratification_hand_fixture():
    # Two samples: early exit good, full-depth anomaly with residual gain
    preds = np.array(
        [
            [[0.1, 0.1], [0.1, 0.1]],
            [[0.9, 0.1], [0.1, 0.1]],
        ],
        dtype=np.float64,
    )
    full = np.array(
        [
            [[0.1, 0.1], [0.1, 0.1]],
            [[0.95, 0.05], [0.05, 0.05]],
        ],
        dtype=np.float64,
    )
    masks = np.array(
        [
            [[0, 0], [0, 0]],
            [[1, 0], [0, 0]],
        ],
        dtype=np.float64,
    )
    depths = np.array([12, 24])
    gains = np.array([0.0, 0.2])
    labels = np.array([0, 1])

    metrics = compute_transfer_metrics(
        adaptive_maps=preds,
        full_depth_maps=full,
        masks=masks,
        selected_depths=depths,
        residual_gains=gains,
        image_labels=labels,
        epsilon=0.05,
        full_depth=24,
    )
    assert "pixel_ap_drop" in metrics
    assert "pro_drop" in metrics
    assert "boundary_f_score_drop" in metrics
    assert "false_safe_exit_rate" in metrics
    assert "expected_depth" in metrics
    assert metrics["exit_histogram"] == {12: 1, 24: 1}

    strata = compute_stratified_metrics(
        adaptive_maps=preds,
        full_depth_maps=full,
        masks=masks,
        selected_depths=depths,
        residual_gains=gains,
        image_labels=labels,
        images=np.stack([preds, preds], axis=0),  # proxy images
        epsilon=0.05,
        full_depth=24,
    )
    assert "normal" in strata and "anomalous" in strata
    assert "anomaly_area" in strata
    assert "contrast" in strata
    assert "boundary_complexity" in strata


def test_export_writes_predictions_before_summary(tmp_path: Path):
    rows = [
        TransferSamplePrediction(
            sample_id="visa/a",
            dataset="visa",
            selected_depth=12,
            image_label=0,
            residual_gain=0.0,
            pixel_ap_adaptive=0.5,
            pixel_ap_full=0.5,
            pro_adaptive=0.5,
            pro_full=0.5,
            boundary_f_adaptive=0.5,
            boundary_f_full=0.5,
            anomaly_area=0.0,
            contrast_proxy=0.1,
            boundary_complexity=0.0,
        )
    ]
    out = tmp_path / "transfer"
    summary = export_transfer_predictions(rows, output_dir=out, full_depth=24, epsilon=0.05)
    pred_path = out / "per_sample_predictions.jsonl"
    summary_path = out / "summary.json"
    assert pred_path.is_file()
    assert summary_path.is_file()
    assert pred_path.stat().st_mtime <= summary_path.stat().st_mtime
    assert summary["n"] == 1
    assert "depth_distribution" in summary


def test_metric_helpers_hand_values():
    pred = np.array([[0.9, 0.1], [0.1, 0.1]], dtype=np.float64)
    mask = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    ap = pixel_average_precision(pred, mask)
    assert 0.0 <= ap <= 1.0
    bf = boundary_f_score(pred, mask)
    assert 0.0 <= bf <= 1.0
    pro = pro_score_proxy(pred, mask)
    assert 0.0 <= pro <= 1.0
