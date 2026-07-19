from __future__ import annotations

import json
from pathlib import Path

import pytest

from rad.evaluation.zero_shot import (
    TargetAccessError,
    assert_policy_unchanged,
    forbid_target_access_during_calibration,
    load_frozen_policy_profile,
)


def test_zero_shot_contract_policy_loaded_before_target_and_digest_stable(
    tmp_path: Path,
) -> None:
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
    }
    policy_path = tmp_path / "policy_profiles.json"
    policy_path.write_text(json.dumps(profiles), encoding="utf-8")

    with forbid_target_access_during_calibration(
        source_dataset="mvtec",
        target_datasets=("visa",),
    ) as guard:
        guard.check_path(policy_path)
        with pytest.raises(TargetAccessError, match="visa"):
            guard.check_path(tmp_path / "visa" / "meta.json")
        _, digest = load_frozen_policy_profile(policy_path, "balanced")
        assert_policy_unchanged(policy_path, "balanced", digest)

    # After calibration window, digest must still match frozen source policy.
    assert_policy_unchanged(policy_path, "balanced", digest)


def test_zero_shot_cli_source_mentions_paper_metrics_and_adapters() -> None:
    root = Path(__file__).resolve().parents[2]
    src = (root / "tools" / "evaluate_zero_shot_transfer.py").read_text(encoding="utf-8")
    assert "PaperMetrics" in src or "compute_paper_metrics" in src
    assert "get_adapter" in src
    assert "target_tuning" in src
