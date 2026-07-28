from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from rad.evaluation.zero_shot import (
    TargetAccessError,
    forbid_target_access_during_calibration,
    load_frozen_policy_profile,
)
from tests.rad.contracts.zero_shot import (
    assert_source_policy_frozen,
    assert_target_tuning_rejected,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def test_forbid_target_access_during_calibration_monkeypatch():
    """Calibration must fail if any target sample path/id is touched."""
    source = "mvtec"
    targets = ("visa",)

    with forbid_target_access_during_calibration(
        source_dataset=source,
        target_datasets=targets,
    ) as guard:
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
    path.write_text(json.dumps(profiles), encoding="utf-8")
    _profile, digest = load_frozen_policy_profile(path, "balanced")
    assert_source_policy_frozen(path, "balanced", digest)
    profiles["profiles"]["balanced"]["gain_threshold"] = 0.99
    path.write_text(json.dumps(profiles), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        assert_source_policy_frozen(path, "balanced", digest)


def test_target_tuning_true_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "zero_shot_target_tuning.yaml"
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/rad/zero_shot_transfer.yaml").read_text(encoding="utf-8")
    )
    payload["zero_shot"]["target_tuning"] = True
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "tools/evaluate_zero_shot_transfer.py"),
            "--config",
            str(config_path),
            "--seed",
            "111",
            "--output-dir",
            str(tmp_path / "zs_out"),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert_target_tuning_rejected(proc)


def test_pro_score_proxy_is_absent_from_zero_shot_reporting_path() -> None:
    src = (REPO_ROOT / "rad/evaluation/zero_shot.py").read_text(encoding="utf-8")
    reporting = src.split("def compute_transfer_metrics", 1)[1]
    assert "pro_score_proxy" not in reporting
