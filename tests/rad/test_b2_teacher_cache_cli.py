"""B2-02A Task 5: launcher-only teacher-cache dry-run CLI and fail-closed matrix."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.rad.b2_hermetic import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SPLIT_V2,
    load_b2_split_fixture,
    write_b2_split_fixture,
    write_hermetic_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "create_b2_teacher_cache.py"
LAUNCHER_PATH = REPO_ROOT / "tools" / "run_with_execution_profile.py"
PROFILE_PATH = REPO_ROOT / "configs" / "execution" / "frozen_deterministic_math.json"
CONFIG_PATH = REPO_ROOT / "configs" / "phase_b" / "b2_teacher_cache_gate_c.json"
EXPECTED_PROFILE_SHA256 = (
    "7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d"
)
EXPECTED_SPLIT_V1 = (
    "0b9371deb6c55f359a14959c8b46ff50205191b1189a48ee380eafaf28c5791a"
)
RESULT_PREFIX = "B2_TEACHER_CACHE_RESULT="
AUDIT_PREFIX = "B2_TEACHER_CACHE_AUDIT="
PASSED_STATUS = re.compile(r'"status"\s*:\s*"passed"', re.IGNORECASE)
SUBPROCESS_TIMEOUT_SECONDS = 120


def _require_cli() -> None:
    assert CLI_PATH.is_file(), f"B2-02A RED: missing production CLI: {CLI_PATH}"


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in (
        "RAD_EXECUTION_PROFILE_BOOTSTRAPPED",
        "RAD_EXECUTION_PROFILE_PATH",
        "RAD_EXECUTION_PROFILE_SHA256",
    ):
        env.pop(key, None)
    return env


def _snapshot_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            snapshot[relative] = "dir"
    return snapshot


def _cli_args(
    *,
    config: Path,
    split_manifest: Path,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    output_dir: Path,
    output_root: Path,
    seed: int = 111,
    dry_run: bool = False,
    resume: bool = False,
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--output-root",
        str(output_root),
        "--split-manifest",
        str(split_manifest),
        "--checkpoint",
        str(checkpoint),
        "--expected-checkpoint-sha256",
        expected_checkpoint_sha256,
    ]
    if dry_run:
        args.append("--dry-run")
    if resume:
        args.append("--resume")
    if extra:
        args.extend(extra)
    return args


def _harness_source(case: str) -> str:
    return f"""
import json
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from rad.runtime.execution_profile import apply_execution_profile
import rad.phase_b.b2_teacher_cache as cache_mod
from tests.rad import b2_hermetic as hermetic

attestation = apply_execution_profile()

case = {case!r}
cli_path = Path(sys.argv[1]).resolve()
hermetic_checkpoint = Path(sys.argv[2]).resolve()
cli_args = sys.argv[3:]
audit = {{
    "load_teacher_bundle_calls": 0,
}}

exit_code = 0
try:
    module = runpy.run_path(str(cli_path), run_name="b2_teacher_cache_cli")
    module["main"].__globals__["_apply_profile"] = lambda _repo: attestation

    import rad.data.teacher_inference as teacher_inference

    def guarded_load(*args, **kwargs):
        audit["load_teacher_bundle_calls"] += 1
        raise AssertionError("dry-run/production B2-02A must not load VisualAD teacher")

    teacher_inference.load_teacher_bundle = guarded_load

    real_load_config = cache_mod.load_teacher_cache_config
    real_validate_checkpoint = cache_mod.validate_checkpoint_bytes

    def hermetic_load_config(path):
        config = real_load_config(path)
        return replace(config, checkpoint_path=hermetic_checkpoint)

    def hermetic_validate_checkpoint(path, expected_sha256):
        resolved = Path(path).resolve()
        if resolved != hermetic_checkpoint:
            return real_validate_checkpoint(path, expected_sha256)
        if not resolved.is_file():
            raise cache_mod.TeacherCacheError(
                "B2_CACHE_CHECKPOINT_MISSING",
                "hermetic checkpoint fixture is absent",
            )
        if expected_sha256 != hermetic.EXPECTED_CHECKPOINT_SHA256:
            raise cache_mod.TeacherCacheError(
                "B2_CACHE_CHECKPOINT_HASH_MISMATCH",
                "hermetic fixture only accepts the approved production hash",
            )
        return expected_sha256

    cache_mod.load_teacher_cache_config = hermetic_load_config
    cache_mod.validate_checkpoint_bytes = hermetic_validate_checkpoint

    head = subprocess.check_output(
        ["git", "-C", str(cli_path.parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    def force_clean(repo, *, require_clean=True):
        return hermetic.synthetic_teacher_cache_identity(
            head_commit=head,
            worktree_clean=True,
            head_is_descendant=True,
        )

    def force_dirty(repo, *, require_clean=True):
        return hermetic.synthetic_teacher_cache_identity(
            head_commit=head,
            worktree_clean=False,
            head_is_descendant=True,
        )

    def force_non_descendant(repo, *, require_clean=True):
        return hermetic.synthetic_teacher_cache_identity(
            head_commit=head,
            worktree_clean=True,
            head_is_descendant=False,
        )

    if case == "dirty_worktree":
        module["main"].__globals__["_derive_repository_identity"] = force_dirty
    elif case == "non_descendant":
        module["main"].__globals__["_derive_repository_identity"] = force_non_descendant
    elif case != "raw_dirty":
        module["main"].__globals__["_derive_repository_identity"] = force_clean

    if case == "inject_test_teacher":
        class FixtureTeacher:
            artifact_kind = "test_fixture"

        def fake_resolve(*_a, **_k):
            return FixtureTeacher()

        module["main"].__globals__["_resolve_production_teacher"] = fake_resolve

    exit_code = module["main"](cli_args)
finally:
    print({AUDIT_PREFIX!r} + json.dumps(audit, sort_keys=True))
raise SystemExit(exit_code)
"""


def _run_via_launcher(
    tmp_path: Path,
    *,
    case: str = "valid",
    split_manifest: Path | None = None,
    checkpoint: Path | None = None,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    dry_run: bool = True,
    resume: bool = False,
    extra: list[str] | None = None,
    output_root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    _require_cli()
    root = output_root or (tmp_path / "output_root")
    root.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir or (root / "run-dry")
    split = split_manifest or write_b2_split_fixture(tmp_path / "split_manifest.json")
    ckpt = checkpoint or write_hermetic_checkpoint(tmp_path / "hermetic_checkpoint.pth")[0]
    args = _cli_args(
        config=CONFIG_PATH,
        split_manifest=split,
        checkpoint=ckpt,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        output_dir=run_dir,
        output_root=root,
        dry_run=dry_run,
        resume=resume,
        extra=extra,
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_PATH),
            "--profile",
            str(PROFILE_PATH),
            "--expected-sha256",
            EXPECTED_PROFILE_SHA256,
            "--",
            sys.executable,
            "-c",
            _harness_source(case),
            str(CLI_PATH),
            str(ckpt),
            *args,
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    return proc, root, run_dir


def _prefixed_json(proc: subprocess.CompletedProcess[str], prefix: str) -> dict[str, Any]:
    matches = [
        json.loads(line.removeprefix(prefix))
        for line in (proc.stdout + "\n" + proc.stderr).splitlines()
        if line.startswith(prefix)
    ]
    assert len(matches) == 1, (
        f"expected one {prefix!r} record, got {len(matches)}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert isinstance(matches[0], dict)
    return matches[0]


def _result(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return _prefixed_json(proc, RESULT_PREFIX)


def _audit(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return _prefixed_json(proc, AUDIT_PREFIX)


def _assert_no_passed_manifest(root: Path, proc: subprocess.CompletedProcess[str]) -> None:
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert not PASSED_STATUS.search(combined)
    assert "status = passed" not in combined
    assert '"status": "passed"' not in combined
    assert '"status":"passed"' not in combined
    for path in root.rglob("*"):
        if path.is_file() and path.suffix == ".json":
            text = path.read_text(encoding="utf-8")
            assert '"status": "passed"' not in text
            assert '"status":"passed"' not in text


def test_assert_no_passed_manifest_catches_literal_and_json_false_positives(
    tmp_path: Path,
) -> None:
    """Guard: OR-logic would miss a lone literal or JSON passed marker."""

    root = tmp_path / "empty_root"
    root.mkdir()
    literal_only = subprocess.CompletedProcess(
        args=["fake"],
        returncode=1,
        stdout="error\nstatus = passed\n",
        stderr="",
    )
    with pytest.raises(AssertionError):
        _assert_no_passed_manifest(root, literal_only)

    json_only = subprocess.CompletedProcess(
        args=["fake"],
        returncode=1,
        stdout='error\n{"status": "passed"}\n',
        stderr="",
    )
    with pytest.raises(AssertionError):
        _assert_no_passed_manifest(root, json_only)

    compact_json = subprocess.CompletedProcess(
        args=["fake"],
        returncode=1,
        stdout='error\n{"status":"passed"}\n',
        stderr="",
    )
    with pytest.raises(AssertionError):
        _assert_no_passed_manifest(root, compact_json)


def test_valid_dry_run_prints_required_summary_and_writes_nothing(tmp_path: Path) -> None:
    output_root = tmp_path / "output_root"
    output_root.mkdir()
    marker = output_root / "preexisting.txt"
    marker.write_text("keep\n", encoding="utf-8")
    before = _snapshot_tree(output_root)

    proc, root, run_dir = _run_via_launcher(
        tmp_path,
        output_root=output_root,
        output_dir=output_root / "planned-run",
        dry_run=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    after = _snapshot_tree(root)
    assert after == before
    assert not run_dir.exists()

    text = proc.stdout
    assert "mode = dry_run" in text
    assert "status = passed" in text
    assert "artifact_written = false" in text
    assert "run_directory_created = false" in text
    assert "planned_samples = 32" in text
    assert "split_scientific_hash_version = 2" in text
    assert f"split_scientific_sha256 = {EXPECTED_SPLIT_V2}" in text
    assert f"checkpoint_sha256 = {EXPECTED_CHECKPOINT_SHA256}" in text
    assert "candidate_layers = [6,12,18,24]" in text

    payload = _result(proc)
    assert payload["mode"] == "dry_run"
    assert payload["status"] == "passed"
    assert payload["artifact_written"] is False
    assert payload["run_directory_created"] is False
    assert payload["planned_samples"] == 32
    assert payload["split_scientific_hash_version"] == 2
    assert payload["split_scientific_sha256"] == EXPECTED_SPLIT_V2
    assert payload["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert payload["candidate_layers"] == [6, 12, 18, 24]
    assert "intended_manifest_metadata" in payload

    audit = _audit(proc)
    assert audit["load_teacher_bundle_calls"] == 0


def test_altered_v2_hash_fails_closed(tmp_path: Path) -> None:
    split = tmp_path / "altered_v2.json"
    payload = load_b2_split_fixture()
    payload["scientific_hash_contract"]["canonical_scientific_hash_v2"] = "a" * 64
    split.write_text(json.dumps(payload), encoding="utf-8")
    proc, root, _ = _run_via_launcher(tmp_path, split_manifest=split, dry_run=True)
    assert "B2_CACHE_SPLIT_HASH_MISMATCH" in proc.stdout + proc.stderr
    _assert_no_passed_manifest(root, proc)


def test_v1_as_current_identity_fails_closed(tmp_path: Path) -> None:
    split = tmp_path / "v1_current.json"
    payload = load_b2_split_fixture()
    payload["scientific_hash_contract"]["active_version"] = 1
    payload["scientific_hash_contract"]["canonical_scientific_hash_v2"] = EXPECTED_SPLIT_V1
    split.write_text(json.dumps(payload), encoding="utf-8")
    proc, root, _ = _run_via_launcher(tmp_path, split_manifest=split, dry_run=True)
    assert "B2_CACHE_SPLIT_V2_REQUIRED" in proc.stdout + proc.stderr
    _assert_no_passed_manifest(root, proc)


def test_invalid_checkpoint_hash_fails_closed(tmp_path: Path) -> None:
    proc, root, _ = _run_via_launcher(
        tmp_path,
        expected_checkpoint_sha256="0" * 64,
        dry_run=True,
    )
    combined = proc.stdout + proc.stderr
    assert "B2_CACHE_CHECKPOINT_HASH_MISMATCH" in combined
    _assert_no_passed_manifest(root, proc)


def test_missing_launcher_bootstrap_fails_closed(tmp_path: Path) -> None:
    _require_cli()
    root = tmp_path / "output_root"
    root.mkdir()
    run_dir = root / "run"
    split = write_b2_split_fixture(tmp_path / "split_manifest.json")
    checkpoint, _ = write_hermetic_checkpoint(tmp_path / "hermetic_checkpoint.pth")
    args = _cli_args(
        config=CONFIG_PATH,
        split_manifest=split,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        output_dir=run_dir,
        output_root=root,
        dry_run=True,
    )
    proc = subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "B2_CACHE_BOOTSTRAP_REQUIRED" in combined
    assert not any(root.rglob("*.json"))
    assert "status = passed" not in combined


def test_dirty_worktree_fails_closed(tmp_path: Path) -> None:
    proc, root, _ = _run_via_launcher(tmp_path, case="dirty_worktree", dry_run=True)
    assert "B2_CACHE_WORKTREE_DIRTY" in proc.stdout + proc.stderr
    _assert_no_passed_manifest(root, proc)


def test_non_descendant_branch_fails_closed(tmp_path: Path) -> None:
    proc, root, _ = _run_via_launcher(tmp_path, case="non_descendant", dry_run=True)
    assert "B2_CACHE_HEAD_NOT_DESCENDANT" in proc.stdout + proc.stderr
    _assert_no_passed_manifest(root, proc)


def test_output_collision_plan_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "output_root"
    run_dir = root / "existing-run"
    run_dir.mkdir(parents=True)
    (run_dir / "partial_manifest.json").write_text(
        '{"status":"partial"}',
        encoding="utf-8",
    )
    before = _snapshot_tree(root)
    proc, observed_root, _ = _run_via_launcher(
        tmp_path,
        output_root=root,
        output_dir=run_dir,
        dry_run=True,
    )
    assert "B2_CACHE_RUN_EXISTS" in proc.stdout + proc.stderr
    assert _snapshot_tree(observed_root) == before
    _assert_no_passed_manifest(root, proc)


def test_production_rejects_test_teacher_selection_and_injection(tmp_path: Path) -> None:
    _require_cli()
    source = CLI_PATH.read_text(encoding="utf-8")
    assert "--teacher" not in source
    assert "test_fixture" not in source
    assert "FakeTeacher" not in source

    proc_flag, root_flag, _ = _run_via_launcher(
        tmp_path,
        dry_run=True,
        extra=["--teacher-kind", "test_fixture"],
    )
    assert proc_flag.returncode != 0
    _assert_no_passed_manifest(root_flag, proc_flag)

    proc, root, _ = _run_via_launcher(
        tmp_path,
        case="inject_test_teacher",
        dry_run=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "B2_CACHE_TEST_TEACHER_FORBIDDEN" in combined
    _assert_no_passed_manifest(root, proc)
    assert _audit(proc)["load_teacher_bundle_calls"] == 0
