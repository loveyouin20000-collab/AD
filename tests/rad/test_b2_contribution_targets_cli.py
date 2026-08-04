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
import dataclasses
import hashlib
import inspect
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
import tests.rad.test_b2_contribution_targets as domain_tests
from tools import create_b2_contribution_targets as cli_mod
from tools import qualify_b2_contribution_target_reproduction as qualification_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "create_b2_contribution_targets.py"
RESULT_PREFIX = "B2_CONTRIBUTION_TARGETS_RESULT="
QUALIFICATION_JSON_NAME = "b2_04b_contribution_targets_manifest.json"
QUALIFICATION_MARKDOWN_NAME = "b2_04b_contribution_targets_report.md"

NEGATIVE_CONTROL_CASE_IDS = (
    "record_file_byte_drift",
    "record_scientific_hash_drift",
    "coalition_utility_component_drift",
    "raw_utility_drift",
    "centered_value_drift",
    "signed_shapley_drift",
    "allocation_drift",
    "efficiency_residual_above_tolerance",
    "changed_split_membership",
    "training_record_moved_to_calibration",
    "calibration_record_in_gt_fitting",
    "evaluation_record_in_normalization",
    "gt_calibration_statistic_drift",
    "shapley_normalization_statistic_drift",
    "teacher_cache_identity_drift",
    "descriptor_collection_identity_drift",
    "descriptor_record_identity_drift",
    "wrong_split_checkpoint_profile",
    "target_domain_or_visa_source",
    "missing_record",
    "extra_record",
    "orphan_pt",
    "path_traversal",
    "symlink_escape",
    "missing_receipt",
    "receipt_mismatch",
    "output_directory_collision",
    "completed_run_reuse",
    "resume_attempt",
    "wrong_expected_plan_sha",
    "dirty_official_worktree",
    "non_descendant_official_head",
    "moved_or_missing_contract_tag",
    "nonzero_teacher_forward_count",
)

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
        "B2_CONTRIBUTION_EXPECTED_PLAN_INVALID"
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
        "B2_CONTRIBUTION_RECOMPUTED_PLAN_MISMATCH"
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


def _qualification_results_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "semantic_spot_checks": {
            "status": "passed",
            "sample_count": 6,
            "depths": [12, 18, 24],
            "run_a_equals_run_b": True,
        },
        "source_only_audit": {
            "status": "passed",
            "target_domain_record_count": 0,
            "teacher_forward_count": 0,
        },
        "negative_controls": {
            "status": "passed",
            "required": 34,
            "passed": 34,
            "case_ids": list(NEGATIVE_CONTROL_CASE_IDS),
        },
        "validation": {
            name: {"status": "passed", "exit_code": 0, "summary": "passed"}
            for name in ("focused_pytest", "full_cpu_pytest", "ruff", "mypy")
        },
        "descriptor_run_independence": {
            "separate_run_directories": True,
            "separate_verification_passes": True,
            "descriptor_scientific_contents_equal": True,
            "descriptor_file_bytes_equal": True,
            "limitation": (
                "dual-run reproduction does not vary descriptor scientific content"
            ),
        },
    }


def test_qualification_cli_exposes_only_the_required_arguments() -> None:
    options = {
        option
        for action in qualification_mod._parser()._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    assert options == {
        "--config",
        "--run-a",
        "--run-b",
        "--qualification-results",
        "--output-dir",
        "--seed",
        "--dry-run",
    }


def _qualification_runs(tmp_path: Path, target_fixture: Any) -> tuple[Path, Path, Path]:
    config_path = fixtures.write_controlled_official_config(tmp_path)
    config = subject.load_contribution_targets_config(config_path)
    inputs = fixtures.fixture_input_bundle(target_fixture)
    expected = subject.run_contribution_target_collection(
        config=config, inputs=inputs
    ).plan["contribution_plan_scientific_sha256"]
    runs: list[Path] = []
    for name in ("run-a", "run-b"):
        result = subject.materialize_contribution_target_collection(
            config=config,
            inputs=inputs,
            output_run_dir=tmp_path / name,
            expected_plan_sha256=expected,
        )
        runs.append(result.run_dir)
    return config_path, runs[0], runs[1]


def test_qualification_writer_is_deterministic_atomic_and_dry_run_safe(
    tmp_path: Path, target_fixture: Any
) -> None:
    config_path, run_a, run_b = _qualification_runs(tmp_path, target_fixture)
    results_path = tmp_path / "qualification-results.json"
    results_path.write_text(
        json.dumps(_qualification_results_payload(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "evidence"
    args = [
        "--config",
        str(config_path),
        "--run-a",
        str(run_a),
        "--run-b",
        str(run_b),
        "--qualification-results",
        str(results_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        "0",
    ]
    assert qualification_mod.main([*args, "--dry-run"]) == 0
    assert not output_dir.exists()
    assert qualification_mod.main(args) == 0
    json_path = output_dir / QUALIFICATION_JSON_NAME
    markdown_path = output_dir / QUALIFICATION_MARKDOWN_NAME
    first_json = json_path.read_bytes()
    first_markdown = markdown_path.read_bytes()
    evidence = json.loads(first_json)
    assert evidence["status"] == "deterministic_dual_contribution_target_reproduction"
    assert evidence["comparison"]["scientifically_equivalent"] is True
    assert evidence["teacher_forward_count"] == 0
    assert len(evidence["runs"]["run_a"]["ordered_record_hashes"]) == 32
    assert set(subject.SEVEN_LAYERED_IDENTITY_KEYS) <= set(
        evidence["runs"]["run_a"]["layered_identities"]
    )
    assert str(tmp_path) not in first_json.decode()
    assert "tensor(" not in first_json.decode()
    assert not list(output_dir.glob("*.tmp"))

    duplicate_dir = tmp_path / "duplicate-evidence"
    duplicate_args = [str(duplicate_dir) if value == str(output_dir) else value for value in args]
    assert qualification_mod.main(duplicate_args) == 0
    assert (duplicate_dir / QUALIFICATION_JSON_NAME).read_bytes() == first_json
    assert (duplicate_dir / QUALIFICATION_MARKDOWN_NAME).read_bytes() == first_markdown
    assert qualification_mod.main(args) == 1


@pytest.mark.parametrize(
    "mutate",
    ["semantic", "source", "negative", "focused_pytest", "full_cpu_pytest", "ruff", "mypy"],
)
def test_qualification_writer_fails_closed_on_missing_decisions(
    tmp_path: Path, target_fixture: Any, mutate: str
) -> None:
    config_path, run_a, run_b = _qualification_runs(tmp_path, target_fixture)
    payload = _qualification_results_payload()
    if mutate == "semantic":
        payload.pop("semantic_spot_checks")
    elif mutate == "source":
        payload.pop("source_only_audit")
    elif mutate == "negative":
        payload["negative_controls"]["case_ids"] = list(NEGATIVE_CONTROL_CASE_IDS[:-1])
    else:
        payload["validation"].pop(mutate)
    results_path = tmp_path / "qualification-results.json"
    results_path.write_text(json.dumps(payload), encoding="utf-8")
    code = qualification_mod.main(
        [
            "--config",
            str(config_path),
            "--run-a",
            str(run_a),
            "--run-b",
            str(run_b),
            "--qualification-results",
            str(results_path),
            "--output-dir",
            str(tmp_path / "evidence"),
            "--seed",
            "0",
        ]
    )
    assert code == 1
    assert not (tmp_path / "evidence").exists()


SCIENTIFIC_COMPARISON_PREDICATES = (
    "layered_identities_equal",
    "record_scientific_hashes_equal",
    "utility_tables_equal",
    "signed_shapley_equal",
    "allocations_equal",
    "gt_calibration_equal",
    "shapley_normalization_equal",
    "coverage_equal",
    "teacher_forward_count_equal",
)


@pytest.fixture(scope="module")
def qualification_bundle(
    tmp_path_factory: pytest.TempPathFactory, target_fixture: Any
) -> tuple[Path, Path, Path, Path]:
    root = tmp_path_factory.mktemp("qualification-bundle")
    config_path, run_a, run_b = _qualification_runs(root, target_fixture)
    results_path = root / "qualification-results.json"
    results_path.write_text(
        json.dumps(_qualification_results_payload(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path, run_a, run_b, results_path


def _qualification_args(
    bundle: tuple[Path, Path, Path, Path], output_dir: Path
) -> list[str]:
    config_path, run_a, run_b, results_path = bundle
    return [
        "--config",
        str(config_path),
        "--run-a",
        str(run_a),
        "--run-b",
        str(run_b),
        "--qualification-results",
        str(results_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        "0",
    ]


@pytest.mark.parametrize("failing_replace", [1, 2])
def test_qualification_writer_leaves_no_partial_output_when_a_replace_fails(
    tmp_path: Path,
    qualification_bundle: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    failing_replace: int,
) -> None:
    output_dir = tmp_path / "evidence"
    args = _qualification_args(qualification_bundle, output_dir)
    production_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(source: Any, destination: Any) -> None:
        calls["count"] += 1
        if calls["count"] == failing_replace:
            raise OSError("injected atomic replace failure")
        production_replace(source, destination)

    monkeypatch.setattr(qualification_mod.os, "replace", flaky_replace)
    assert qualification_mod.main(args) == 1
    assert calls["count"] == failing_replace
    assert not (output_dir / QUALIFICATION_JSON_NAME).exists()
    assert not (output_dir / QUALIFICATION_MARKDOWN_NAME).exists()
    assert _snapshot_tree(output_dir) == {}

    monkeypatch.setattr(qualification_mod.os, "replace", production_replace)
    assert qualification_mod.main(args) == 0
    assert (output_dir / QUALIFICATION_JSON_NAME).is_file()
    assert (output_dir / QUALIFICATION_MARKDOWN_NAME).is_file()


def test_qualification_writer_rejects_an_unverified_collection(
    tmp_path: Path,
    qualification_bundle: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_verify = subject.verify_contribution_target_collection

    def unregistered(**kwargs: Any) -> Any:
        return dataclasses.replace(production_verify(**kwargs))

    monkeypatch.setattr(subject, "verify_contribution_target_collection", unregistered)
    output_dir = tmp_path / "evidence"
    assert qualification_mod.main(_qualification_args(qualification_bundle, output_dir)) == 1
    assert not output_dir.exists()


@pytest.mark.parametrize("predicate", SCIENTIFIC_COMPARISON_PREDICATES)
def test_qualification_writer_rejects_any_false_scientific_predicate(
    tmp_path: Path,
    qualification_bundle: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    predicate: str,
) -> None:
    production_compare = subject.compare_contribution_target_collections

    def degraded(**kwargs: Any) -> Any:
        comparison = production_compare(**kwargs)
        assert getattr(comparison, predicate) is True
        return dataclasses.replace(comparison, **{predicate: False})

    monkeypatch.setattr(subject, "compare_contribution_target_collections", degraded)
    output_dir = tmp_path / "evidence"
    assert qualification_mod.main(_qualification_args(qualification_bundle, output_dir)) == 1
    assert not output_dir.exists()


@dataclasses.dataclass(frozen=True)
class _TeacherForwardProbe:
    manifest: Any
    teacher_forward_count: int


def test_qualification_writer_rejects_a_nonzero_teacher_forward_count(
    qualification_bundle: tuple[Path, Path, Path, Path],
) -> None:
    config_path, run_a, run_b, results_path = qualification_bundle
    config = subject.load_contribution_targets_config(config_path)
    verified_a = subject.verify_contribution_target_collection(config=config, run_dir=run_a)
    verified_b = subject.verify_contribution_target_collection(config=config, run_dir=run_b)
    comparison = subject.compare_contribution_target_collections(
        first=verified_a, second=verified_b
    )
    results = qualification_mod._load_qualification_results(results_path)
    for counts in ((1, 0), (0, 1)):
        with pytest.raises(subject.ContributionTargetError) as excinfo:
            qualification_mod._build_evidence(
                config_path=config_path,
                config=config,
                run_a=_TeacherForwardProbe(verified_a.manifest, counts[0]),
                run_b=_TeacherForwardProbe(verified_b.manifest, counts[1]),
                comparison=comparison,
                qualification_results=results,
                seed=0,
            )
        assert str(getattr(excinfo.value, "code", "")) == (
            "B2_CONTRIBUTION_QUALIFICATION_INVALID"
        )
        assert "teacher forward" in str(excinfo.value)


# Each negative-control ID binds to the concrete domain test case (plus the
# parametrize case ids, when the behavior lives in specific parameters) that
# actually exercises it.
NEGATIVE_CONTROL_TEST_MAP: dict[str, tuple[Any, tuple[str, ...]]] = {
    "record_file_byte_drift": (domain_tests.test_record_file_hash_mismatch_is_detected, ()),
    "record_scientific_hash_drift": (
        domain_tests.test_record_scientific_hash_mismatch_is_detected,
        (),
    ),
    "coalition_utility_component_drift": (
        domain_tests.test_comparison_categorizes_every_scientific_mismatch,
        ("coalition_utility_component_drift",),
    ),
    "raw_utility_drift": (
        domain_tests.test_comparison_categorizes_every_scientific_mismatch,
        ("raw_utility_drift", "empty_coalition_raw_utility_drift"),
    ),
    "centered_value_drift": (
        domain_tests.test_comparison_categorizes_every_scientific_mismatch,
        ("centered_value_drift", "grand_coalition_centered_value_drift"),
    ),
    "signed_shapley_drift": (
        domain_tests.test_comparison_categorizes_every_scientific_mismatch,
        ("signed_shapley_drift", "efficiency_residual_drift"),
    ),
    "allocation_drift": (
        domain_tests.test_comparison_categorizes_every_scientific_mismatch,
        ("allocation_drift",),
    ),
    "efficiency_residual_above_tolerance": (
        domain_tests.test_build_target_record_fails_closed_on_efficiency_violation,
        (),
    ),
    "changed_split_membership": (
        domain_tests.test_comparison_categorizes_every_scientific_mismatch,
        ("changed_split_membership",),
    ),
    "training_record_moved_to_calibration": (
        domain_tests.test_training_access_mode_fails_closed_on_calibration_or_evaluation_records,
        (),
    ),
    "calibration_record_in_gt_fitting": (
        domain_tests.test_gt_map_calibration_rejects_non_training_samples_from_the_fixture,
        (),
    ),
    "evaluation_record_in_normalization": (
        domain_tests.test_shapley_normalization_uses_only_the_sixteen_training_records,
        (),
    ),
    "gt_calibration_statistic_drift": (
        domain_tests.test_tampered_gt_map_calibration_statistics_fail_closed,
        (),
    ),
    "shapley_normalization_statistic_drift": (
        domain_tests.test_tampered_shapley_normalization_statistics_fail_closed,
        (),
    ),
    "teacher_cache_identity_drift": (
        domain_tests.test_comparison_categorizes_every_scientific_mismatch,
        ("teacher_cache_identity_drift",),
    ),
    "descriptor_collection_identity_drift": (
        domain_tests.test_comparison_categorizes_every_scientific_mismatch,
        ("descriptor_collection_identity_drift",),
    ),
    "descriptor_record_identity_drift": (
        domain_tests.test_comparison_categorizes_every_scientific_mismatch,
        ("descriptor_record_identity_drift",),
    ),
    "wrong_split_checkpoint_profile": (
        domain_tests.test_bind_upstream_identities_rejects_upstream_hash_drift,
        ("split_scientific_sha256", "checkpoint_sha256", "execution_profile_sha256"),
    ),
    "target_domain_or_visa_source": (
        domain_tests.test_module_avoids_teacher_loading_and_target_domain,
        (),
    ),
    "missing_record": (
        domain_tests.test_orphan_extra_and_temporary_artifacts_fail_the_integrity_audit,
        (),
    ),
    "extra_record": (
        domain_tests.test_orphan_extra_and_temporary_artifacts_fail_the_integrity_audit,
        (),
    ),
    "orphan_pt": (
        domain_tests.test_orphan_extra_and_temporary_artifacts_fail_the_integrity_audit,
        (),
    ),
    "path_traversal": (
        domain_tests.test_run_relative_paths_reject_traversal,
        ("../escape.pt", "records/../../escape.pt", "/absolute/escape.pt", "records/./"),
    ),
    "symlink_escape": (
        domain_tests.test_symlink_escape_from_the_run_directory_is_rejected,
        (),
    ),
    "missing_receipt": (
        domain_tests.test_missing_and_mismatched_final_manifest_receipt_fail_closed,
        (),
    ),
    "receipt_mismatch": (
        domain_tests.test_missing_and_mismatched_final_manifest_receipt_fail_closed,
        (),
    ),
    "output_directory_collision": (
        domain_tests.test_output_collision_and_resume_fail_closed,
        (),
    ),
    "completed_run_reuse": (domain_tests.test_output_collision_and_resume_fail_closed, ()),
    "resume_attempt": (domain_tests.test_output_collision_and_resume_fail_closed, ()),
    "wrong_expected_plan_sha": (
        domain_tests.test_expected_plan_hash_mismatch_fails_before_any_write,
        (),
    ),
    "dirty_official_worktree": (
        domain_tests.test_official_api_rechecks_repository_identity_before_any_write,
        (),
    ),
    "non_descendant_official_head": (
        domain_tests.test_official_api_rejects_a_non_descendant_head,
        ("parent", "sibling", "unrelated"),
    ),
    "moved_or_missing_contract_tag": (
        domain_tests.test_repository_identity_verifier_rejects_missing_or_moved_tag,
        (),
    ),
    "nonzero_teacher_forward_count": (
        domain_tests.test_verifier_rejects_a_nonzero_manifest_teacher_forward_count,
        (),
    ),
}

# Identity-drift negative controls must be proven by a case that really drifts
# the named scientific identity, never by a sample/link mismatch test. An ID
# standing for several identities must evidence every one of them.
REQUIRED_IDENTITY_EVIDENCE: dict[str, tuple[str, ...]] = {
    "teacher_cache_identity_drift": ("teacher_cache_scientific_sha256",),
    "descriptor_collection_identity_drift": ("descriptor_collection_scientific_sha256",),
    "descriptor_record_identity_drift": ("descriptor_record_scientific_sha256",),
    "wrong_split_checkpoint_profile": (
        "split_scientific_sha256",
        "checkpoint_sha256",
        "execution_profile_sha256",
    ),
}

FORBIDDEN_IDENTITY_EVIDENCE_TESTS = (
    domain_tests.test_bind_upstream_identities_rejects_teacher_descriptor_mismatch,
    domain_tests.test_bind_upstream_identities_rejects_sample_and_split_mismatch,
)


def _parametrized_case_ids(test_case: Any) -> tuple[str, ...]:
    ids: list[str] = []
    for mark in getattr(test_case, "pytestmark", ()):
        if mark.name == "parametrize":
            ids.extend(str(value) for value in mark.args[1])
    return tuple(ids)


def _collected_case_ids(module_path: Path) -> dict[str, set[str]]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(module_path),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    collected: dict[str, set[str]] = {}
    for line in completed.stdout.splitlines():
        _prefix, separator, node = line.partition("::")
        if not separator:
            continue
        name, bracket, parameters = node.strip().partition("[")
        collected.setdefault(name, set())
        if bracket:
            collected[name].add(parameters.removesuffix("]"))
    assert collected, completed.stdout
    return collected


@pytest.fixture(scope="module")
def collected_domain_case_ids() -> dict[str, set[str]]:
    return _collected_case_ids(REPO_ROOT / "tests" / "rad" / "test_b2_contribution_targets.py")


def _evidenced_identity_fields(parameter: str) -> set[str]:
    fields: set[str] = set()
    if parameter.endswith("_sha256"):
        fields.add(parameter)
    categorized = domain_tests.COMPARISON_MISMATCH_CASES.get(parameter)
    if categorized is not None:
        fields.add(categorized[1])
    return fields


def test_every_negative_control_id_maps_to_a_real_asserting_test_case(
    collected_domain_case_ids: dict[str, set[str]],
) -> None:
    assert qualification_mod.NEGATIVE_CONTROL_CASE_IDS == NEGATIVE_CONTROL_CASE_IDS
    assert len(NEGATIVE_CONTROL_CASE_IDS) == 34
    assert len(set(NEGATIVE_CONTROL_CASE_IDS)) == 34
    assert set(NEGATIVE_CONTROL_TEST_MAP) == set(NEGATIVE_CONTROL_CASE_IDS)
    for case_id, (test_case, parameters) in NEGATIVE_CONTROL_TEST_MAP.items():
        assert callable(test_case), case_id
        assert test_case.__name__.startswith("test_"), case_id
        assert test_case.__module__ == domain_tests.__name__, case_id
        source = inspect.getsource(test_case)
        assert "assert " in source or "pytest.raises" in source, case_id
        assert test_case.__name__ in collected_domain_case_ids, case_id
        assert isinstance(parameters, tuple), case_id
        for parameter in parameters:
            assert parameter in collected_domain_case_ids[test_case.__name__], (
                case_id,
                parameter,
            )
            assert parameter in _parametrized_case_ids(test_case), (case_id, parameter)


def test_identity_negative_controls_bind_to_real_identity_drift_cases(
    collected_domain_case_ids: dict[str, set[str]],
) -> None:
    problems: list[str] = []
    for case_id, required_fields in REQUIRED_IDENTITY_EVIDENCE.items():
        test_case, parameters = NEGATIVE_CONTROL_TEST_MAP[case_id]
        if test_case in FORBIDDEN_IDENTITY_EVIDENCE_TESTS:
            problems.append(
                f"{case_id} is bound to the sample/link mismatch test "
                f"{test_case.__name__}"
            )
            continue
        evidenced: set[str] = set()
        for parameter in parameters:
            assert parameter in collected_domain_case_ids[test_case.__name__], (
                case_id,
                parameter,
            )
            evidenced |= _evidenced_identity_fields(parameter)
        missing = sorted(set(required_fields) - evidenced)
        if missing:
            problems.append(
                f"{case_id} evidences {sorted(evidenced)} and misses {missing}"
            )
    assert problems == []
