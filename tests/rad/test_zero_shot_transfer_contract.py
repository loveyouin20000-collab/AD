from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rad.errors import ARTIFACT_INTEGRITY_EXIT_CODE, ArtifactIntegrityError
from rad.evaluation.zero_shot import (
    TargetAccessError,
    assert_policy_eligible_for_evaluation,
    assert_policy_unchanged,
    forbid_target_access_during_calibration,
    load_frozen_policy_profile,
)
from tests.rad.contracts.policy_fixture import write_minimal_policy_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
TEST_POLICY_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "policy_profiles.json"
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
    assert "--calibration-policy" in src


def test_real_evaluation_rejects_test_policy_fixture() -> None:
    with pytest.raises(ArtifactIntegrityError, match="test fixture"):
        assert_policy_eligible_for_evaluation(TEST_POLICY_FIXTURE)


def test_zero_shot_non_dry_run_rejects_test_policy_fixture(tmp_path: Path) -> None:
    out = tmp_path / "zs_out"
    proc = subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "tools" / "evaluate_zero_shot_transfer.py"),
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--calibration-policy",
            str(TEST_POLICY_FIXTURE),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == ARTIFACT_INTEGRITY_EXIT_CODE
    blob = proc.stdout + proc.stderr
    assert "test fixture" in blob.lower() or "artifact integrity" in blob.lower()
    assert not out.exists()


def test_dynamic_policy_fixture_dry_run_ok_non_dry_run_rejected(tmp_path: Path) -> None:
    """Regression: tmp_path policy mirrors static test_fixture contract."""
    out = tmp_path / "zs_out"
    policy_path = tmp_path / "policy_profiles.json"
    write_minimal_policy_fixture(policy_path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["artifact_kind"] == "test_fixture"
    assert payload["eligible_for_evaluation"] is False

    dry_proc = subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "tools" / "evaluate_zero_shot_transfer.py"),
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--calibration-policy",
            str(policy_path),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_proc.returncode == 0, dry_proc.stdout + dry_proc.stderr
    dry_blob = dry_proc.stdout + dry_proc.stderr
    assert "dry-run" in dry_blob.lower()
    assert "policy_digest" in dry_blob
    assert f"policy_path: {policy_path}" in dry_blob
    assert not out.exists()

    eval_proc = subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "tools" / "evaluate_zero_shot_transfer.py"),
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--calibration-policy",
            str(policy_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert eval_proc.returncode == ARTIFACT_INTEGRITY_EXIT_CODE
    eval_blob = eval_proc.stdout + eval_proc.stderr
    assert "test fixture" in eval_blob.lower() or "artifact integrity" in eval_blob.lower()
    assert "build_engine" not in eval_blob
    assert "get_adapter" not in eval_blob
    assert not out.exists()
