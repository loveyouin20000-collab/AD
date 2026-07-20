from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rad.errors import ArtifactIntegrityError
from rad.evaluation.zero_shot import (
    TargetAccessError,
    assert_policy_eligible_for_evaluation,
    forbid_target_access_during_calibration,
    load_frozen_policy_profile,
)
from tests.rad.contracts.policy_fixture import write_minimal_policy_fixture
from tests.rad.contracts.zero_shot import (
    assert_adapter_only_dataset_access,
    assert_missing_policy_artifact_integrity,
    assert_policy_fixture_rejected_for_real_run,
    assert_real_run_rejects_test_fixture,
    assert_source_policy_frozen,
    assert_zero_shot_dry_run_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
ZERO_SHOT_CLI = REPO_ROOT / "tools/evaluate_zero_shot_transfer.py"
TEST_POLICY_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "policy_profiles.json"
)


def _run(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
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
        _profile, digest = load_frozen_policy_profile(policy_path, "balanced")
        assert_source_policy_frozen(policy_path, "balanced", digest)

    assert_source_policy_frozen(policy_path, "balanced", digest)


def test_zero_shot_cli_adapter_only_dataset_access() -> None:
    src = ZERO_SHOT_CLI.read_text(encoding="utf-8")
    assert_adapter_only_dataset_access(src)
    assert "target_tuning" in src
    assert "--calibration-policy" in src


def test_real_evaluation_rejects_test_policy_fixture() -> None:
    with pytest.raises(ArtifactIntegrityError, match="test fixture"):
        assert_policy_eligible_for_evaluation(TEST_POLICY_FIXTURE)


def test_zero_shot_non_dry_run_rejects_test_policy_fixture(tmp_path: Path) -> None:
    out = tmp_path / "zs_out"
    proc = _run(
        [
            PYTHON,
            str(ZERO_SHOT_CLI),
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--calibration-policy",
            str(TEST_POLICY_FIXTURE),
        ]
    )
    assert_policy_fixture_rejected_for_real_run(proc)
    assert not out.exists()


def test_dynamic_policy_fixture_dry_run_ok_non_dry_run_rejected(tmp_path: Path) -> None:
    out = tmp_path / "zs_out"
    policy_path = tmp_path / "policy_profiles.json"
    write_minimal_policy_fixture(policy_path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["artifact_kind"] == "test_fixture"
    assert payload["eligible_for_evaluation"] is False

    dry_proc = _run(
        [
            PYTHON,
            str(ZERO_SHOT_CLI),
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--calibration-policy",
            str(policy_path),
            "--dry-run",
        ]
    )
    assert_zero_shot_dry_run_contract(dry_proc, output_dir=out, policy_path=policy_path)

    eval_proc = _run(
        [
            PYTHON,
            str(ZERO_SHOT_CLI),
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--calibration-policy",
            str(policy_path),
        ]
    )
    assert_real_run_rejects_test_fixture(eval_proc)
    assert not out.exists()


def test_zero_shot_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "zs_out"
    policy_path = tmp_path / "policy_profiles.json"
    write_minimal_policy_fixture(policy_path)
    proc = _run(
        [
            PYTHON,
            str(ZERO_SHOT_CLI),
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--calibration-policy",
            str(policy_path),
            "--dry-run",
        ]
    )
    assert_zero_shot_dry_run_contract(proc, output_dir=out, policy_path=policy_path)


def test_zero_shot_non_dry_run_missing_policy_fails(tmp_path: Path) -> None:
    out = tmp_path / "zs_out"
    missing_policy = tmp_path / "missing_policy_profiles.json"
    proc = _run(
        [
            PYTHON,
            str(ZERO_SHOT_CLI),
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--calibration-policy",
            str(missing_policy),
        ]
    )
    assert_missing_policy_artifact_integrity(proc)
    assert not out.exists()


def test_zero_shot_cli_policy_path_precedence_over_config(tmp_path: Path) -> None:
    out = tmp_path / "zs_out"
    explicit_policy = tmp_path / "explicit_policy_profiles.json"
    write_minimal_policy_fixture(explicit_policy)
    proc = _run(
        [
            PYTHON,
            str(ZERO_SHOT_CLI),
            "--config",
            "configs/rad/zero_shot_transfer.yaml",
            "--seed",
            "111",
            "--output-dir",
            str(out),
            "--calibration-policy",
            str(explicit_policy),
            "--dry-run",
        ]
    )
    assert_zero_shot_dry_run_contract(proc, output_dir=out, policy_path=explicit_policy)
    blob = proc.stdout + proc.stderr
    configured = "artifacts/calibration/policy/policy_profiles.json"
    assert configured not in blob.split("policy_path:")[-1]
