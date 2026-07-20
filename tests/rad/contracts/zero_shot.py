"""Zero-shot transfer test contracts — assertions and parsers only."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rad.errors import ARTIFACT_INTEGRITY_EXIT_CODE
from rad.evaluation.zero_shot import assert_policy_unchanged

ZERO_SHOT_CLI = "tools/evaluate_zero_shot_transfer.py"


def _blob(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


def assert_source_policy_frozen(
    policy_path: Path,
    profile_name: str,
    digest: str,
) -> None:
    assert_policy_unchanged(policy_path, profile_name, digest)


def assert_target_tuning_rejected(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode != 0, _blob(proc)
    blob = _blob(proc).lower()
    assert "target_tuning" in blob or "target-domain tuning" in blob


def assert_policy_fixture_rejected_for_real_run(
    proc: subprocess.CompletedProcess[str],
) -> None:
    assert proc.returncode == ARTIFACT_INTEGRITY_EXIT_CODE, _blob(proc)
    blob = _blob(proc).lower()
    assert "test fixture" in blob or "artifact integrity" in blob


def assert_adapter_only_dataset_access(source: str) -> None:
    assert "_load_visa_index" not in source
    assert "get_adapter" in source
    assert "evaluate_dataset" in source
    assert "compute_paper_metrics" in source
    assert "forbid_target_access_during_calibration" in source
    assert "assert_policy_unchanged" in source
    assert "pro_score_proxy" not in source


def assert_policy_path_precedence(blob: str, explicit_policy_path: Path) -> None:
    resolved = str(explicit_policy_path.resolve())
    assert f"policy_path: {resolved}" in blob or f"policy_path: {explicit_policy_path}" in blob


def assert_dry_run_writes_nothing(
    output_dir: Path,
    proc: subprocess.CompletedProcess[str],
) -> None:
    assert proc.returncode == 0, _blob(proc)
    blob = _blob(proc).lower()
    assert "dry-run" in blob
    assert not output_dir.exists()


def assert_missing_policy_artifact_integrity(
    proc: subprocess.CompletedProcess[str],
) -> None:
    assert proc.returncode == ARTIFACT_INTEGRITY_EXIT_CODE, _blob(proc)
    blob = _blob(proc).lower()
    assert "missing calibration policy" in blob or "artifact integrity" in blob


def assert_zero_shot_dry_run_contract(
    proc: subprocess.CompletedProcess[str],
    *,
    output_dir: Path,
    policy_path: Path | None = None,
) -> None:
    assert_dry_run_writes_nothing(output_dir, proc)
    blob = _blob(proc)
    assert "policy_digest" in blob
    if policy_path is not None:
        assert_policy_path_precedence(blob, policy_path)


def assert_real_run_rejects_test_fixture(proc: subprocess.CompletedProcess[str]) -> None:
    assert_policy_fixture_rejected_for_real_run(proc)
    blob = _blob(proc)
    assert "build_engine" not in blob
    assert "get_adapter" not in blob


def parse_policy_digest(blob: str) -> str | None:
    for line in blob.splitlines():
        stripped = line.strip()
        if stripped.startswith("policy_digest:"):
            return stripped.split(":", 1)[1].strip()
    return None


def assert_policy_digest_unchanged(before: str, after: str) -> None:
    digest_before = parse_policy_digest(before)
    digest_after = parse_policy_digest(after)
    assert digest_before is not None, "missing policy_digest before evaluation"
    assert digest_after is not None, "missing policy_digest after evaluation"
    assert digest_before == digest_after
