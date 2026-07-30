"""B2-04A CLI contract tests for ``tools/create_b2_contribution_targets.py``.

The CLI is exercised both in-process and through a real subprocess with a
hermetic temporary input layout. Nothing here touches the real teacher cache, a
teacher checkpoint, a backbone, VisA, or any target-domain path.

Contract assumed by these tests:

* The argument surface is exactly ``--config``, ``--teacher-cache-manifest``,
  ``--teacher-cache-root``, ``--descriptor-manifest``, ``--descriptor-root``,
  ``--mvtec-root``, ``--output-root``, ``--dry-run``, ``--expected-plan-sha256``
  plus the workspace-mandated ``--seed`` and ``--output-dir``.

* ``--dry-run`` performs the complete scientific computation and writes nothing.

* A non-dry-run invocation with the tracked Gate-C config fails closed with
  ``B2_CONTRIBUTION_OFFICIAL_MATERIALIZATION_NOT_ENABLED``.

* Every invocation emits exactly one machine-readable
  ``B2_CONTRIBUTION_TARGETS_RESULT=`` line.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import rad.phase_b.b2_contribution_targets as subject
import tests.rad.b2_contribution_target_fixtures as fixtures
from tools import create_b2_contribution_targets as cli_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "create_b2_contribution_targets.py"
RESULT_PREFIX = "B2_CONTRIBUTION_TARGETS_RESULT="

FORBIDDEN_FLAGS = (
    "--fixture",
    "--use-fixture",
    "--fixture-variant",
    "--override-hash",
    "--force-hash",
    "--skip-identity",
    "--skip-identity-check",
    "--teacher-checkpoint",
    "--checkpoint",
    "--visa-root",
    "--target-domain-root",
    "--allow-overwrite",
    "--resume",
)


@pytest.fixture(scope="module")
def target_fixture() -> Any:
    return fixtures.build_contribution_target_fixture()


@pytest.fixture(scope="module")
def expected_dry_run(target_fixture: Any) -> dict[str, Any]:
    config = subject.load_contribution_targets_config(fixtures.TRACKED_CONFIG_PATH)
    return subject.dry_run_contribution_targets(
        config=config, inputs=fixtures.fixture_input_bundle(target_fixture)
    )


def _snapshot_tree(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not root.exists():
        return snapshot
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = f"symlink:{os.readlink(path)}"
        elif path.is_dir():
            snapshot[relative] = "dir"
        else:
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _cli_args(
    layout: Any,
    *,
    output_root: Path,
    output_dir: Path,
    config: Path | None = None,
    seed: int = 0,
    dry_run: bool = True,
    expected_plan_sha256: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "--config",
        str(config if config is not None else fixtures.TRACKED_CONFIG_PATH),
        "--teacher-cache-manifest",
        str(layout.teacher_cache_manifest),
        "--teacher-cache-root",
        str(layout.teacher_cache_root),
        "--descriptor-manifest",
        str(layout.descriptor_manifest),
        "--descriptor-root",
        str(layout.descriptor_root),
        "--mvtec-root",
        str(layout.mvtec_root),
        "--output-root",
        str(output_root),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
    ]
    if dry_run:
        args.append("--dry-run")
    if expected_plan_sha256 is not None:
        args.extend(["--expected-plan-sha256", expected_plan_sha256])
    if extra:
        args.extend(extra)
    return args


def _result_json(stdout: str, stderr: str) -> dict[str, Any]:
    combined = f"{stdout}\n{stderr}"
    matches = [
        json.loads(line.removeprefix(RESULT_PREFIX))
        for line in combined.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(matches) == 1, f"expected exactly one result line, got {len(matches)}:\n{combined}"
    return matches[0]


def _run_in_process(argv: list[str]) -> tuple[int, str, str]:
    assert CLI_PATH.is_file(), f"missing production CLI: {CLI_PATH}"
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cli_mod.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(argv: list[str]) -> subprocess.CompletedProcess[str]:
    assert CLI_PATH.is_file(), f"missing production CLI: {CLI_PATH}"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *argv],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=environment,
    )


def test_cli_exposes_exactly_the_required_argument_surface() -> None:
    parser = cli_mod._parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    required = {
        "--config",
        "--teacher-cache-manifest",
        "--teacher-cache-root",
        "--descriptor-manifest",
        "--descriptor-root",
        "--mvtec-root",
        "--output-root",
        "--dry-run",
        "--expected-plan-sha256",
        "--seed",
        "--output-dir",
    }
    assert required <= options
    assert options & set(FORBIDDEN_FLAGS) == set()


def test_cli_source_contains_no_forbidden_flag_or_target_domain_path() -> None:
    source = CLI_PATH.read_text(encoding="utf-8")
    for flag in FORBIDDEN_FLAGS:
        assert flag not in source
    for forbidden in ("VisA", "visa", "load_teacher_bundle", "/root/", "/home/", "autodl"):
        assert forbidden not in source


def test_cli_requires_every_mandatory_argument(tmp_path: Path, target_fixture: Any) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    complete = _cli_args(
        layout, output_root=tmp_path / "out", output_dir=tmp_path / "out" / "run"
    )
    for flag in (
        "--config",
        "--teacher-cache-manifest",
        "--teacher-cache-root",
        "--descriptor-manifest",
        "--descriptor-root",
        "--mvtec-root",
        "--output-root",
    ):
        index = complete.index(flag)
        truncated = complete[:index] + complete[index + 2 :]
        with pytest.raises(SystemExit) as excinfo:
            with contextlib.redirect_stderr(io.StringIO()):
                cli_mod._parser().parse_args(truncated)
        assert excinfo.value.code == 2


def test_cli_dry_run_reports_the_frozen_plan_and_writes_nothing(
    tmp_path: Path, target_fixture: Any, expected_dry_run: dict[str, Any]
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    output_root = tmp_path / "out"
    before = _snapshot_tree(tmp_path)
    code, stdout, stderr = _run_in_process(
        _cli_args(layout, output_root=output_root, output_dir=output_root / "run")
    )
    assert code == 0, stderr
    payload = _result_json(stdout, stderr)
    assert payload["mode"] == "dry_run"
    assert payload["status"] == "passed"
    assert payload["artifact_written"] is False
    assert payload["run_directory_created"] is False
    assert payload["teacher_forward_count"] == 0
    assert payload["planned_samples"] == 32
    assert payload["training_targets"] == 16
    assert payload["calibration_targets"] == 8
    assert payload["evaluation_targets"] == 8
    assert payload["training_samples_for_gt_calibration"] == 16
    assert payload["calibration_samples_for_gt_calibration"] == 0
    assert payload["evaluation_samples_for_gt_calibration"] == 0
    assert payload["training_samples_for_shapley_normalization"] == 16
    assert payload["calibration_samples_for_shapley_normalization"] == 0
    assert payload["evaluation_samples_for_shapley_normalization"] == 0
    assert payload["prediction_depths"] == [12, 18, 24]
    assert payload["coalition_counts"] == {"12": 4, "18": 8, "24": 16}
    assert payload["contribution_plan_scientific_sha256"] == (
        expected_dry_run["contribution_plan_scientific_sha256"]
    )
    for identity in subject.SEVEN_LAYERED_IDENTITY_KEYS:
        assert payload[identity] == expected_dry_run[identity]
    assert payload["official_materialization_enabled"] is False
    assert _snapshot_tree(tmp_path) == before
    assert not output_root.exists()


def test_cli_dry_run_prints_the_human_readable_summary(
    tmp_path: Path, target_fixture: Any
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    code, stdout, stderr = _run_in_process(
        _cli_args(layout, output_root=tmp_path / "out", output_dir=tmp_path / "out" / "run")
    )
    assert code == 0, stderr
    assert "mode = dry_run" in stdout
    assert "status = passed" in stdout
    assert "artifact_written = false" in stdout
    assert "run_directory_created = false" in stdout
    assert "teacher_forward_count = 0" in stdout
    assert "planned_samples = 32" in stdout


def test_cli_dry_run_subprocess_matches_the_in_process_plan(
    tmp_path: Path, target_fixture: Any, expected_dry_run: dict[str, Any]
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    output_root = tmp_path / "out"
    before = _snapshot_tree(tmp_path)
    completed = _run_subprocess(
        _cli_args(layout, output_root=output_root, output_dir=output_root / "run")
    )
    assert completed.returncode == 0, completed.stderr
    payload = _result_json(completed.stdout, completed.stderr)
    assert payload["contribution_plan_scientific_sha256"] == (
        expected_dry_run["contribution_plan_scientific_sha256"]
    )
    assert payload["artifact_written"] is False
    assert _snapshot_tree(tmp_path) == before


def test_cli_non_dry_run_fails_closed_while_official_materialization_is_disabled(
    tmp_path: Path, target_fixture: Any
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    output_root = tmp_path / "out"
    before = _snapshot_tree(tmp_path)
    code, stdout, stderr = _run_in_process(
        _cli_args(
            layout,
            output_root=output_root,
            output_dir=output_root / "run",
            dry_run=False,
            expected_plan_sha256="0" * 64,
        )
    )
    assert code == 1
    payload = _result_json(stdout, stderr)
    assert payload["status"] == "failed"
    assert payload["artifact_written"] is False
    assert payload["code"] == "B2_CONTRIBUTION_OFFICIAL_MATERIALIZATION_NOT_ENABLED"
    assert _snapshot_tree(tmp_path) == before
    assert not output_root.exists()


def test_cli_non_dry_run_subprocess_fails_closed(tmp_path: Path, target_fixture: Any) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    output_root = tmp_path / "out"
    completed = _run_subprocess(
        _cli_args(
            layout,
            output_root=output_root,
            output_dir=output_root / "run",
            dry_run=False,
            expected_plan_sha256="0" * 64,
        )
    )
    assert completed.returncode == 1
    payload = _result_json(completed.stdout, completed.stderr)
    assert payload["code"] == "B2_CONTRIBUTION_OFFICIAL_MATERIALIZATION_NOT_ENABLED"
    assert not output_root.exists()


def test_cli_rejects_a_missing_configuration(tmp_path: Path, target_fixture: Any) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    code, stdout, stderr = _run_in_process(
        _cli_args(
            layout,
            output_root=tmp_path / "out",
            output_dir=tmp_path / "out" / "run",
            config=tmp_path / "absent.json",
        )
    )
    assert code == 1
    assert _result_json(stdout, stderr)["code"] == "B2_CONTRIBUTION_CONFIG_MISSING"


def test_cli_rejects_a_non_integer_seed(tmp_path: Path, target_fixture: Any) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    args = _cli_args(layout, output_root=tmp_path / "out", output_dir=tmp_path / "out" / "run")
    args[args.index("--seed") + 1] = "not-an-int"
    code, _stdout, stderr = _run_in_process(args)
    assert code == 2
    assert "B2_CONTRIBUTION_SEED_INVALID" in stderr


def test_cli_rejects_a_malformed_expected_plan_hash(
    tmp_path: Path, target_fixture: Any
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    code, stdout, stderr = _run_in_process(
        _cli_args(
            layout,
            output_root=tmp_path / "out",
            output_dir=tmp_path / "out" / "run",
            expected_plan_sha256="nope",
        )
    )
    assert code == 1
    assert _result_json(stdout, stderr)["code"] == (
        "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MALFORMED"
    )


def test_cli_dry_run_accepts_a_matching_expected_plan_hash(
    tmp_path: Path, target_fixture: Any, expected_dry_run: dict[str, Any]
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    code, stdout, stderr = _run_in_process(
        _cli_args(
            layout,
            output_root=tmp_path / "out",
            output_dir=tmp_path / "out" / "run",
            expected_plan_sha256=expected_dry_run["contribution_plan_scientific_sha256"],
        )
    )
    assert code == 0, stderr
    payload = _result_json(stdout, stderr)
    assert payload["expected_plan_sha256_matched"] is True


def test_cli_dry_run_rejects_a_mismatched_expected_plan_hash(
    tmp_path: Path, target_fixture: Any
) -> None:
    layout = fixtures.prepare_hermetic_contribution_inputs(tmp_path, fixture=target_fixture)
    code, stdout, stderr = _run_in_process(
        _cli_args(
            layout,
            output_root=tmp_path / "out",
            output_dir=tmp_path / "out" / "run",
            expected_plan_sha256="a" * 64,
        )
    )
    assert code == 1
    assert _result_json(stdout, stderr)["code"] == (
        "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MISMATCH"
    )


def test_cli_never_imports_a_teacher_bundle_or_dataset_adapter() -> None:
    source = CLI_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "load_teacher_bundle",
        "build_backbone",
        "MVTecDataset",
        "torch.load",
        "set_grad_enabled",
        "torch.backends",
    ):
        assert forbidden not in source
