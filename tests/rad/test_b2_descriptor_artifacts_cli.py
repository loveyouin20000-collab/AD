"""B2-03A dry-run CLI contract tests for `tools/create_b2_descriptor_artifacts.py`.

Repository identity is exercised through temporary Git repositories with a
synthetic annotated integration tag. These tests must not depend on the real
checkout containing ``b2-main-integration-v1``, deep history, or developer tags.

The Gate-C config pin for ``expected_main_tag`` / ``expected_main_commit`` is
temporarily aligned to the hermetic fixture (same pattern as B2 tiny-split
identity tests). Production tracked config and fail-closed git checks stay
unchanged for official invocations.
"""

from __future__ import annotations

import contextlib
import io
import json
import runpy
import subprocess
from pathlib import Path
from typing import Any

import pytest

import rad.phase_b.b2_descriptor_artifacts as artifacts
import rad.phase_b.b2_teacher_cache as cache_mod
from tests.rad import b2_descriptor_fixtures as fixtures
from tools import create_b2_descriptor_artifacts as cli_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "create_b2_descriptor_artifacts.py"
RESULT_PREFIX = "B2_DESCRIPTOR_ARTIFACTS_RESULT="


def _require_cli() -> None:
    assert CLI_PATH.is_file(), f"missing production CLI: {CLI_PATH}"


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


def _result_json(stdout: str, stderr: str) -> dict[str, Any]:
    combined = stdout + "\n" + stderr
    matches = [
        json.loads(line.removeprefix(RESULT_PREFIX))
        for line in combined.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(matches) == 1, f"expected exactly one result line, got {len(matches)}:\n{combined}"
    return matches[0]


def _install_hermetic_identity(
    monkeypatch: pytest.MonkeyPatch, identity_repo: dict[str, Any]
) -> None:
    """Align pinned Gate-C identity constants to a temporary repo and validate there.

    Does not stub validation to succeed: the real ``_validate_main_ancestry`` still
    runs fail-closed against the hermetic Git repository.
    """

    monkeypatch.setattr(
        artifacts, "_EXPECTED_MAIN_TAG", identity_repo["expected_main_tag"]
    )
    monkeypatch.setattr(
        artifacts, "_EXPECTED_MAIN_COMMIT", identity_repo["expected_main_commit"]
    )
    real_validate = cli_mod._validate_main_ancestry

    def validate_on_hermetic_repo(
        repo: Path,
        *,
        expected_main_tag: str,
        expected_main_commit: str,
    ) -> dict[str, Any]:
        _ = repo  # production CLI resolves __file__ repo; tests redirect to hermetic
        return real_validate(
            identity_repo["repo"],
            expected_main_tag=expected_main_tag,
            expected_main_commit=expected_main_commit,
        )

    monkeypatch.setattr(cli_mod, "_validate_main_ancestry", validate_on_hermetic_repo)


def _run_cli_in_process(
    argv: list[str],
    *,
    monkeypatch: pytest.MonkeyPatch,
    identity_repo: dict[str, Any] | None = None,
    guard_backbone: bool = True,
) -> tuple[int, str, str]:
    _require_cli()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if identity_repo is not None:
        _install_hermetic_identity(monkeypatch, identity_repo)
    audit: dict[str, int] = {"load_teacher_bundle_calls": 0}
    if guard_backbone:
        import rad.data.teacher_inference as teacher_inference

        def guarded_load(*_args: Any, **_kwargs: Any) -> Any:
            audit["load_teacher_bundle_calls"] += 1
            raise AssertionError(
                "descriptor-artifacts dry-run/CLI must never load the VisualAD teacher"
            )

        monkeypatch.setattr(
            teacher_inference, "load_teacher_bundle", guarded_load, raising=False
        )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    exit_code = 1
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
        stderr_buffer
    ):
        try:
            exit_code = cli_mod.main(argv)
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    assert audit["load_teacher_bundle_calls"] == 0, "backbone must never be loaded by this CLI"
    return exit_code, stdout_buffer.getvalue(), stderr_buffer.getvalue()


@pytest.fixture
def cache(tmp_path: Path) -> dict[str, Any]:
    return fixtures.build_descriptor_test_fixture(tmp_path / "cache_fixture")


@pytest.fixture
def identity_repo(tmp_path: Path) -> dict[str, Any]:
    return fixtures.build_hermetic_descriptor_identity_repo(tmp_path / "identity")


@pytest.fixture
def production_manifest_path(cache: dict[str, Any]) -> Path:
    manifest = fixtures.production_like_manifest(cache)
    path = cache["cache_root"] / "teacher_cache_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def descriptor_config_path(
    tmp_path: Path, cache: dict[str, Any], identity_repo: dict[str, Any]
) -> Path:
    return fixtures.write_descriptor_config_json(
        tmp_path / "configs",
        cache,
        overrides={
            "expected_main_tag": identity_repo["expected_main_tag"],
            "expected_main_commit": identity_repo["expected_main_commit"],
        },
    )


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
    output_root.mkdir(parents=True, exist_ok=True)
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
    identity_repo: dict[str, Any],
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
    exit_code, stdout, stderr = _run_cli_in_process(
        args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
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
    identity_repo: dict[str, Any],
    production_manifest_path: Path,
    descriptor_config_path: Path,
) -> None:
    args = _default_args(
        tmp_path, cache=cache, manifest_path=production_manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(
        args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
    assert exit_code == 0, stdout + stderr
    assert "teacher_forward_count = 0" in stdout or _result_json(stdout, stderr)["teacher_forward_count"] == 0


def test_dry_run_rejects_wrong_cache_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    identity_repo: dict[str, Any],
    descriptor_config_path: Path,
) -> None:
    manifest = fixtures.production_like_manifest(cache)
    manifest["cache_scientific_sha256"] = "0" * 64
    manifest_path = cache["cache_root"] / "bad_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = _default_args(
        tmp_path, cache=cache, manifest_path=manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(
        args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
    combined = stdout + stderr
    assert exit_code != 0
    assert "B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH" in combined
    assert "status = passed" not in combined
    assert '"status": "passed"' not in combined


def test_dry_run_rejects_status_not_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    identity_repo: dict[str, Any],
    descriptor_config_path: Path,
) -> None:
    manifest = fixtures.production_like_manifest(cache)
    manifest["status"] = "partial"
    manifest_path = cache["cache_root"] / "partial_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = _default_args(
        tmp_path, cache=cache, manifest_path=manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(
        args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
    combined = stdout + stderr
    assert exit_code != 0
    assert "B2_DESC_CACHE_STATUS_NOT_PASSED" in combined
    assert "status = passed" not in combined


def test_dry_run_rejects_test_fixture_artifact_kind_without_override_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    identity_repo: dict[str, Any],
    descriptor_config_path: Path,
) -> None:
    manifest_path = cache["cache_root"] / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(cache["manifest"]), encoding="utf-8")

    args = _default_args(
        tmp_path, cache=cache, manifest_path=manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(
        args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
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
        "--skip-repository-identity",
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
    identity_repo: dict[str, Any],
    production_manifest_path: Path,
    descriptor_config_path: Path,
) -> None:
    args = _default_args(
        tmp_path, cache=cache, manifest_path=production_manifest_path, config_path=descriptor_config_path
    )
    exit_code, stdout, stderr = _run_cli_in_process(
        args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
    assert exit_code == 0, stdout + stderr
    output_root = tmp_path / "output_root"
    assert not any(output_root.rglob("*.pt"))
    assert not any(output_root.rglob("manifest.json"))
    assert not any(output_root.rglob("*descriptor*"))


def test_dry_run_rejects_manifest_path_outside_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    identity_repo: dict[str, Any],
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
    exit_code, stdout, stderr = _run_cli_in_process(
        args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
    assert exit_code != 0
    assert "B2_DESC_CACHE_MANIFEST_OUTSIDE_ROOT" in stdout + stderr


def test_dry_run_rejects_disk_record_drift_even_if_manifest_claims_valid_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    identity_repo: dict[str, Any],
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
    exit_code, stdout, stderr = _run_cli_in_process(
        args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
    assert exit_code != 0
    assert "B2_DESC_CACHE_SCIENTIFIC_HASH_MISMATCH" in stdout + stderr


def test_production_mode_rejects_test_fixture_manifest_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    identity_repo: dict[str, Any],
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
    exit_code, stdout, stderr = _run_cli_in_process(
        args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
    assert exit_code != 0
    assert "B2_DESC_CACHE_TEST_FIXTURE_FORBIDDEN" in stdout + stderr


def test_valid_dry_run_and_production_share_same_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache: dict[str, Any],
    identity_repo: dict[str, Any],
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
        dry_run_args, monkeypatch=monkeypatch, identity_repo=identity_repo
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
        prod_args, monkeypatch=monkeypatch, identity_repo=identity_repo
    )
    assert prod_exit == 0, prod_stdout + prod_stderr
    prod_payload = _result_json(prod_stdout, prod_stderr)

    assert dry_payload["teacher_forward_count"] == 0
    assert prod_payload["teacher_forward_count"] == 0
    assert dry_payload["source_teacher_cache_manifest_file_sha256"] == prod_payload[
        "source_teacher_cache_manifest_file_sha256"
    ]


def test_missing_synthetic_tag_fails_closed(tmp_path: Path) -> None:
    identity = fixtures.build_hermetic_descriptor_identity_repo(
        tmp_path, with_integration_tag=False
    )
    with pytest.raises(
        cli_mod.B2DescriptorArtifactsCLIError,
        match="B2_DESC_REPOSITORY_IDENTITY_UNAVAILABLE",
    ):
        cli_mod._validate_main_ancestry(
            identity["repo"],
            expected_main_tag=identity["expected_main_tag"],
            expected_main_commit=identity["expected_main_commit"],
        )


def test_tag_pointing_to_wrong_commit_fails_closed(tmp_path: Path) -> None:
    identity = fixtures.build_hermetic_descriptor_identity_repo(tmp_path)
    with pytest.raises(cli_mod.B2DescriptorArtifactsCLIError, match="B2_DESC_MAIN_TAG_MOVED"):
        cli_mod._validate_main_ancestry(
            identity["repo"],
            expected_main_tag=identity["expected_main_tag"],
            expected_main_commit="0" * 40,
        )


def test_non_descendant_head_fails_closed(tmp_path: Path) -> None:
    identity = fixtures.build_hermetic_descriptor_identity_repo(tmp_path)
    repo = identity["repo"]
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--orphan", "unrelated"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "rm", "-rf", "."],
        check=False,
        capture_output=True,
        text=True,
    )
    (repo / "orphan.txt").write_text("orphan\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "orphan.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=B2 Descriptor Hermetic",
            "-c",
            "user.email=b2-descriptor-hermetic@example.invalid",
            "commit",
            "-m",
            "unrelated root",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(
        cli_mod.B2DescriptorArtifactsCLIError,
        match="B2_DESC_MAIN_HEAD_NOT_DESCENDANT",
    ):
        cli_mod._validate_main_ancestry(
            repo,
            expected_main_tag=identity["expected_main_tag"],
            expected_main_commit=identity["expected_main_commit"],
        )


def test_dirty_worktree_fails_closed(tmp_path: Path) -> None:
    identity = fixtures.build_hermetic_descriptor_identity_repo(tmp_path)
    (identity["repo"] / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(cli_mod.B2DescriptorArtifactsCLIError, match="B2_DESC_WORKTREE_DIRTY"):
        cli_mod._validate_main_ancestry(
            identity["repo"],
            expected_main_tag=identity["expected_main_tag"],
            expected_main_commit=identity["expected_main_commit"],
        )


def test_observed_head_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = fixtures.build_hermetic_descriptor_identity_repo(tmp_path)
    real_git = cli_mod._git
    head_reads = {"count": 0}

    def flaky_git(repo: Path, *arguments: str, allow_empty: bool = False) -> str:
        if arguments == ("rev-parse", "HEAD"):
            head_reads["count"] += 1
            if head_reads["count"] == 1:
                return real_git(repo, *arguments, allow_empty=allow_empty)
            return "f" * 40
        return real_git(repo, *arguments, allow_empty=allow_empty)

    monkeypatch.setattr(cli_mod, "_git", flaky_git)
    with pytest.raises(
        cli_mod.B2DescriptorArtifactsCLIError,
        match="B2_DESC_REPOSITORY_IDENTITY_MISMATCH",
    ):
        cli_mod._validate_main_ancestry(
            identity["repo"],
            expected_main_tag=identity["expected_main_tag"],
            expected_main_commit=identity["expected_main_commit"],
        )


def test_production_official_config_still_requires_real_main_tag(
    tmp_path: Path,
) -> None:
    """Hermetic success path must not weaken the tracked production tag contract."""

    production = json.loads(
        (
            REPO_ROOT / "configs" / "phase_b" / "b2_descriptor_artifacts_gate_c.json"
        ).read_text(encoding="utf-8")
    )
    assert production["expected_main_tag"] == "b2-main-integration-v1"
    assert (
        production["expected_main_commit"]
        == "51e18ade0231c7488ef582bde1e9694f933e85eb"
    )
    assert artifacts._EXPECTED_MAIN_TAG == "b2-main-integration-v1"
    assert artifacts._EXPECTED_MAIN_COMMIT == "51e18ade0231c7488ef582bde1e9694f933e85eb"
    tagless = fixtures.build_hermetic_descriptor_identity_repo(
        tmp_path, with_integration_tag=False
    )
    with pytest.raises(
        cli_mod.B2DescriptorArtifactsCLIError,
        match="B2_DESC_REPOSITORY_IDENTITY_UNAVAILABLE",
    ):
        cli_mod._validate_main_ancestry(
            tagless["repo"],
            expected_main_tag=production["expected_main_tag"],
            expected_main_commit=production["expected_main_commit"],
        )
