"""B2-03A TDD RED: dry-run CLI contract tests for `tools/create_b2_descriptor_artifacts.py`.

The CLI does not exist yet. Every test below fails RED (missing file / import errors)
until it is written.

--------------------------------------------------------------------------------
Design notes / assumed CLI contract for the GREEN implementer:

* Args: ``--config --teacher-cache-manifest --teacher-cache-root --output-root
  --output-dir --seed --dry-run`` (workspace rule: every CLI supports
  ``--config --seed --output-dir --dry-run``).
* No ``--fixture`` / ``--allow-test-fixture`` / hash-override flag exists anywhere.
  The CLI always calls ``validate_accepted_teacher_cache(..., allow_test_fixture=False)``.
* Dry-run never imports/loads ``rad.data.teacher_inference.load_teacher_bundle`` and
  never touches the filesystem under ``--output-root``/``--output-dir``.
* Dry-run prints (at minimum) the lines:
    mode = dry_run
    status = passed
  plus a single ``B2_DESCRIPTOR_ARTIFACTS_RESULT=<json>`` line with (at least):
    mode, status, artifact_written (bool), teacher_forward_count (int),
    planned_samples, training_samples_for_normalization,
    calibration_samples_for_normalization, evaluation_samples_for_normalization,
    prediction_depths (list[int]), descriptor_dimension (int).
* This CLI is pure hash/JSON/plan arithmetic for --dry-run (no floating-point
  tensor math), so unlike the B2-02A teacher-cache CLI it does not require the
  ``tools/run_with_execution_profile.py`` launcher bootstrap for --dry-run. Real
  (non-dry-run) descriptor extraction is out of scope for these RED tests.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import io
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

import rad.phase_b.b2_teacher_cache as cache_mod
from tests.rad import b2_descriptor_fixtures as fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "create_b2_descriptor_artifacts.py"
RESULT_PREFIX = "B2_DESCRIPTOR_ARTIFACTS_RESULT="


def _require_cli() -> None:
    assert CLI_PATH.is_file(), f"B2-03A RED: missing production CLI: {CLI_PATH}"


def _cli_args(
    *,
    config: Path,
    teacher_cache_manifest: Path,
    teacher_cache_root: Path,
    output_root: Path,
    output_dir: Path,
    seed: int = 0,
    dry_run: bool = True,
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
        "--teacher-cache-manifest",
        str(teacher_cache_manifest),
        "--teacher-cache-root",
        str(teacher_cache_root),
    ]
    if dry_run:
        args.append("--dry-run")
    if extra:
        args.extend(extra)
    return args


def _snapshot_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    import hashlib

    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            snapshot[relative] = "dir"
    return snapshot


def _run_cli_in_process(
    argv: list[str],
    *,
    monkeypatch: pytest.MonkeyPatch,
    guard_backbone: bool = True,
) -> tuple[int, str, str]:
    """Load and execute the CLI's ``main`` in-process (no launcher subprocess needed
    for --dry-run: descriptor-artifacts planning is pure hash/JSON arithmetic)."""

    _require_cli()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    audit: dict[str, int] = {"load_teacher_bundle_calls": 0}
    if guard_backbone:
        import rad.data.teacher_inference as teacher_inference

        def guarded_load(*_args: Any, **_kwargs: Any) -> Any:
            audit["load_teacher_bundle_calls"] += 1
            raise AssertionError(
                "descriptor-artifacts dry-run/CLI must never load the VisualAD teacher"
            )

        monkeypatch.setattr(teacher_inference, "load_teacher_bundle", guarded_load, raising=False)

    namespace = runpy.run_path(str(CLI_PATH), run_name="b2_descriptor_artifacts_cli_under_test")
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    exit_code = 1
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        try:
            exit_code = namespace["main"](argv)
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    assert audit["load_teacher_bundle_calls"] == 0, "backbone must never be loaded by this CLI"
    return exit_code, stdout_buffer.getvalue(), stderr_buffer.getvalue()


def _result_json(stdout: str, stderr: str) -> dict[str, Any]:
    combined = stdout + "\n" + stderr
    matches = [
        json.loads(line.removeprefix(RESULT_PREFIX))
        for line in combined.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(matches) == 1, f"expected exactly one result line, got {len(matches)}:\n{combined}"
    return matches[0]


@pytest.fixture
def cache(tmp_path: Path) -> dict[str, Any]:
    return fixtures.build_descriptor_test_fixture(tmp_path)


@pytest.fixture
def production_manifest_path(tmp_path: Path, cache: dict[str, Any]) -> Path:
    manifest = fixtures.production_like_manifest(cache)
    path = cache["cache_root"] / "teacher_cache_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def descriptor_config_path(tmp_path: Path, cache: dict[str, Any]) -> Path:
    return fixtures.write_descriptor_config_json(tmp_path, cache)


def _default_args(
    tmp_path: Path,
    *,
    cache: dict[str, Any],
    manifest_path: Path,
    config_path: Path,
    dry_run: bool = True,
    extra: list[str] | None = None,
) -> list[str]:
    output_root = tmp_path / "output_root"
    output_root.mkdir(exist_ok=True)
    return _cli_args(
        config=config_path,
        teacher_cache_manifest=manifest_path,
        teacher_cache_root=cache["cache_root"],
        output_root=output_root,
        output_dir=output_root / "planned-run",
        dry_run=dry_run,
        extra=extra,
    )


def test_valid_dry_run_prints_required_summary_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    production_manifest_path: Path,
    descriptor_config_path: Path,
) -> None:
    output_root = tmp_path / "output_root"
    output_root.mkdir()
    marker = output_root / "preexisting.txt"
    marker.write_text("keep\n", encoding="utf-8")
    before = _snapshot_tree(output_root)

    args = _cli_args(
        config=descriptor_config_path,
        teacher_cache_manifest=production_manifest_path,
        teacher_cache_root=cache["cache_root"],
        output_root=output_root,
        output_dir=output_root / "planned-run",
        dry_run=True,
    )
    exit_code, stdout, stderr = _run_cli_in_process(args, monkeypatch=monkeypatch)
    assert exit_code == 0, stdout + stderr
    assert _snapshot_tree(output_root) == before
    assert not (output_root / "planned-run").exists()

    assert "mode = dry_run" in stdout
    assert "status = passed" in stdout
    assert "artifact_written = false" in stdout

    payload = _result_json(stdout, stderr)
    assert payload["mode"] == "dry_run"
    assert payload["status"] == "passed"
    assert payload["artifact_written"] is False
    assert payload["teacher_forward_count"] == 0
    assert payload["planned_samples"] == 32
    assert payload["training_samples_for_normalization"] == 16
    assert payload["calibration_samples_for_normalization"] == 0
    assert payload["evaluation_samples_for_normalization"] == 0
    assert payload["prediction_depths"] == [12, 18, 24]
    assert payload["descriptor_dimension"] == 18


def test_dry_run_never_loads_teacher_bundle_or_backbone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    production_manifest_path: Path,
    descriptor_config_path: Path,
) -> None:
    args = _default_args(
        tmp_path, cache=cache, manifest_path=production_manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(args, monkeypatch=monkeypatch)
    assert exit_code == 0, stdout + stderr
    assert "teacher_forward_count = 0" in stdout or _result_json(stdout, stderr)["teacher_forward_count"] == 0


def test_dry_run_rejects_wrong_cache_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    descriptor_config_path: Path,
) -> None:
    manifest = fixtures.production_like_manifest(cache)
    manifest["cache_scientific_sha256"] = "0" * 64
    manifest_path = cache["cache_root"] / "bad_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = _default_args(
        tmp_path, cache=cache, manifest_path=manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(args, monkeypatch=monkeypatch)
    combined = stdout + stderr
    assert exit_code != 0
    assert "B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH" in combined
    assert "status = passed" not in combined
    assert '"status": "passed"' not in combined


def test_dry_run_rejects_status_not_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    descriptor_config_path: Path,
) -> None:
    manifest = fixtures.production_like_manifest(cache)
    manifest["status"] = "partial"
    manifest_path = cache["cache_root"] / "partial_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = _default_args(
        tmp_path, cache=cache, manifest_path=manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(args, monkeypatch=monkeypatch)
    combined = stdout + stderr
    assert exit_code != 0
    assert "B2_DESC_CACHE_STATUS_NOT_PASSED" in combined
    assert "status = passed" not in combined


def test_dry_run_rejects_test_fixture_artifact_kind_without_override_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    descriptor_config_path: Path,
) -> None:
    # `cache["manifest"]` (unlike `production_like_manifest`) still carries the
    # "artifact_kind": "test_fixture" marker -- the production CLI must reject it
    # unconditionally; there is no flag to bypass this.
    manifest_path = cache["cache_root"] / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(cache["manifest"]), encoding="utf-8")

    args = _default_args(
        tmp_path, cache=cache, manifest_path=manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(args, monkeypatch=monkeypatch)
    combined = stdout + stderr
    assert exit_code != 0
    assert "B2_DESC_CACHE_TEST_FIXTURE_FORBIDDEN" in combined
    assert "status = passed" not in combined


def test_cli_has_no_fixture_override_or_hash_override_flags() -> None:
    _require_cli()
    source = CLI_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "--fixture",
        "--allow-test-fixture",
        "--allow-fixture",
        "--override-hash",
        "--expected-teacher-cache-scientific-sha256",
        "--expected-sample-coverage-sha256",
    ):
        assert forbidden not in source, f"CLI must not expose {forbidden}"


def test_cli_rejects_unrecognized_hash_override_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    production_manifest_path: Path,
    descriptor_config_path: Path,
) -> None:
    args = _default_args(
        tmp_path,
        cache=cache,
        manifest_path=production_manifest_path,
        config_path=descriptor_config_path,
        extra=["--allow-test-fixture"],
    )
    _require_cli()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    namespace = runpy.run_path(str(CLI_PATH), run_name="b2_descriptor_artifacts_cli_under_test")
    with pytest.raises(SystemExit):
        namespace["main"](args)


@pytest.mark.parametrize(
    "missing_flag",
    ["--config", "--teacher-cache-manifest", "--teacher-cache-root", "--output-root", "--output-dir", "--seed"],
)
def test_cli_requires_workspace_mandated_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    production_manifest_path: Path,
    descriptor_config_path: Path,
    missing_flag: str,
) -> None:
    full_args = _default_args(
        tmp_path, cache=cache, manifest_path=production_manifest_path, config_path=descriptor_config_path
    )
    reduced: list[str] = []
    skip_next = False
    for token in full_args:
        if skip_next:
            skip_next = False
            continue
        if token == missing_flag:
            skip_next = True
            continue
        reduced.append(token)
    _require_cli()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    namespace = runpy.run_path(str(CLI_PATH), run_name="b2_descriptor_artifacts_cli_under_test")
    with pytest.raises(SystemExit):
        namespace["main"](reduced)


def test_dry_run_creates_no_manifest_or_descriptor_tensor_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    production_manifest_path: Path,
    descriptor_config_path: Path,
) -> None:
    args = _default_args(
        tmp_path, cache=cache, manifest_path=production_manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(args, monkeypatch=monkeypatch)
    assert exit_code == 0, stdout + stderr
    output_root = tmp_path / "output_root"
    assert not any(output_root.rglob("*.pt"))
    assert not any(output_root.rglob("manifest.json"))
    assert not any(output_root.rglob("*descriptor*"))


def test_dry_run_rejects_manifest_path_outside_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    descriptor_config_path: Path,
) -> None:
    manifest_path = tmp_path / "outside-manifest.json"
    manifest_path.write_text(
        json.dumps(fixtures.production_like_manifest(cache), sort_keys=True),
        encoding="utf-8",
    )
    isolated_root = tmp_path / "isolated-root"
    isolated_root.mkdir()
    args = _cli_args(
        config=descriptor_config_path,
        teacher_cache_manifest=manifest_path,
        teacher_cache_root=isolated_root,
        output_root=tmp_path / "output-root",
        output_dir=tmp_path / "output-root" / "planned-run",
        dry_run=True,
    )
    exit_code, stdout, stderr = _run_cli_in_process(args, monkeypatch=monkeypatch)
    assert exit_code != 0
    assert "B2_DESC_CACHE_MANIFEST_OUTSIDE_ROOT" in stdout + stderr


def test_dry_run_rejects_disk_record_drift_even_if_manifest_claims_valid_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    descriptor_config_path: Path,
) -> None:
    manifest = fixtures.production_like_manifest(cache)
    stable_id = manifest["samples"][0]["stable_sample_id"]

    def mutate_tensor(record: dict[str, Any]) -> dict[str, Any]:
        name = next(key for key in sorted(record["tensors"]) if key.startswith("causal_map:"))
        tensor = record["tensors"][name]["tensor"].clone()
        tensor[0, 0, 0, 0] += 1.0
        record["tensors"][name] = dict(record["tensors"][name])
        record["tensors"][name]["tensor"] = tensor
        record["tensors"][name]["digest"] = cache_mod.canonical_tensor_digest(
            name, tensor, tuple(record["tensors"][name]["dimension_semantics"])
        )
        return record

    fixtures.rewrite_sample_record(cache, manifest, stable_id, mutate_tensor)
    manifest["cache_scientific_sha256"] = cache["teacher_cache_scientific_sha256"]
    manifest["sample_coverage_sha256"] = cache["sample_coverage_sha256"]
    manifest_path = cache["cache_root"] / "drifted-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    args = _default_args(
        tmp_path, cache=cache, manifest_path=manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(args, monkeypatch=monkeypatch)
    assert exit_code != 0
    assert "B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH" in stdout + stderr


def test_production_mode_rejects_test_fixture_manifest_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    descriptor_config_path: Path,
) -> None:
    manifest_path = cache["cache_root"] / "fixture-manifest.json"
    manifest_path.write_text(json.dumps(cache["manifest"], sort_keys=True), encoding="utf-8")
    args = _default_args(
        tmp_path,
        cache=cache,
        manifest_path=manifest_path,
        config_path=descriptor_config_path,
        dry_run=False,
    )
    exit_code, stdout, stderr = _run_cli_in_process(args, monkeypatch=monkeypatch)
    assert exit_code != 0
    assert "B2_DESC_CACHE_TEST_FIXTURE_FORBIDDEN" in stdout + stderr


def test_valid_dry_run_and_production_share_same_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    production_manifest_path: Path,
    descriptor_config_path: Path,
) -> None:
    dry_run_args = _default_args(
        tmp_path,
        cache=cache,
        manifest_path=production_manifest_path,
        config_path=descriptor_config_path,
        dry_run=True,
    )
    dry_exit, dry_stdout, dry_stderr = _run_cli_in_process(
        dry_run_args, monkeypatch=monkeypatch
    )
    assert dry_exit == 0, dry_stdout + dry_stderr
    dry_payload = _result_json(dry_stdout, dry_stderr)

    prod_args = _default_args(
        tmp_path,
        cache=cache,
        manifest_path=production_manifest_path,
        config_path=descriptor_config_path,
        dry_run=False,
    )
    prod_exit, prod_stdout, prod_stderr = _run_cli_in_process(
        prod_args, monkeypatch=monkeypatch
    )
    assert prod_exit == 0, prod_stdout + prod_stderr
    prod_payload = _result_json(prod_stdout, prod_stderr)

    assert dry_payload["teacher_forward_count"] == 0
    assert prod_payload["teacher_forward_count"] == 0
    assert dry_payload["source_teacher_cache_manifest_file_sha256"] == prod_payload[
        "source_teacher_cache_manifest_file_sha256"
    ]
