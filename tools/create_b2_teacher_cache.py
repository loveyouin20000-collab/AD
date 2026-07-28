#!/usr/bin/env python3
"""Bootstrap-only B2 teacher-cache CLI: dry-run planning and fail-closed coordination."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

RESULT_PREFIX = "B2_TEACHER_CACHE_RESULT="
APPROVED_SEED = 111
EXPECTED_B2_TAG = "b2-tiny-split-v1"
EXPECTED_B2_COMMIT = "18bac047227754c975b23b46842458a5b41d5e2a"
EXPECTED_CONTRACT_TAG = "b2-teacher-cache-contract-v1"


class B2TeacherCacheCLIError(RuntimeError):
    """A stable, fail-closed CLI contract error."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2TeacherCacheCLIError(code, detail)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or coordinate the B2 Gate-C teacher-cache run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument(
        "--mvtec-root",
        default=None,
        help="MVTec source root for production generation (required unless --dry-run).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_seed(seed_text: str) -> int:
    try:
        seed = int(seed_text)
    except ValueError:
        _fail("B2_CACHE_SEED_INVALID", f"seed must be an integer: {seed_text!r}")
    if seed != APPROVED_SEED:
        _fail("B2_CACHE_SEED_DRIFT", f"seed must be {APPROVED_SEED}, observed {seed}")
    return seed


def _load_json_object(path_text: str, *, missing_code: str, invalid_code: str) -> dict[str, Any]:
    path = Path(path_text)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _fail(missing_code, f"path does not exist: {path}")
    except OSError as exc:
        _fail(invalid_code, f"cannot read {path}: {exc}")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        _fail(invalid_code, f"invalid JSON at {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(invalid_code, f"JSON root must be an object: {path}")
    return payload


def _run_git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        _fail("B2_CACHE_REPOSITORY_IDENTITY_UNAVAILABLE", f"cannot execute git: {exc}")


def _git(repo: Path, *arguments: str, allow_empty: bool = False) -> str:
    completed = _run_git(repo, *arguments)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git failed"
        _fail("B2_CACHE_REPOSITORY_IDENTITY_UNAVAILABLE", detail)
    value = completed.stdout.strip()
    if not value and not allow_empty:
        _fail("B2_CACHE_REPOSITORY_IDENTITY_UNAVAILABLE", "git returned an empty identity")
    return value


def _derive_repository_identity(
    repo: Path,
    *,
    require_clean: bool,
) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    b2_tag_commit = _git(repo, "rev-parse", f"{EXPECTED_B2_TAG}^{{commit}}")
    if b2_tag_commit != EXPECTED_B2_COMMIT:
        _fail("B2_CACHE_B2_TAG_MOVED", "B2 base tag moved")
    ancestry = _run_git(repo, "merge-base", "--is-ancestor", EXPECTED_B2_COMMIT, head)
    if ancestry.returncode == 1:
        head_is_descendant = False
    elif ancestry.returncode == 0:
        head_is_descendant = True
    else:
        detail = ancestry.stderr.strip() or ancestry.stdout.strip() or "git failed"
        _fail("B2_CACHE_REPOSITORY_IDENTITY_UNAVAILABLE", detail)
    contract_probe = _run_git(repo, "rev-parse", f"{EXPECTED_CONTRACT_TAG}^{{commit}}")
    if contract_probe.returncode != 0:
        _fail(
            "B2_CACHE_CONTRACT_TAG_UNRESOLVED",
            f"missing contract tag {EXPECTED_CONTRACT_TAG}",
        )
    contract_commit = contract_probe.stdout.strip()
    contract_ancestry = _run_git(
        repo, "merge-base", "--is-ancestor", contract_commit, head
    )
    if contract_ancestry.returncode == 1:
        _fail(
            "B2_CACHE_CONTRACT_HEAD_NOT_DESCENDANT",
            "HEAD is not a descendant of the teacher-cache contract tag",
        )
    if contract_ancestry.returncode != 0:
        detail = (
            contract_ancestry.stderr.strip()
            or contract_ancestry.stdout.strip()
            or "git failed"
        )
        _fail("B2_CACHE_REPOSITORY_IDENTITY_UNAVAILABLE", detail)
    worktree_clean = not bool(
        _git(repo, "status", "--porcelain", "--untracked-files=all", allow_empty=True)
    )
    if require_clean and not worktree_clean:
        _fail("B2_CACHE_WORKTREE_DIRTY", "official worktree is dirty")
    return {
        "b2_tag_commit": b2_tag_commit,
        "contract_tag_commit": contract_commit,
        "head_commit": head,
        "head_is_descendant": head_is_descendant,
        "worktree_clean": worktree_clean,
    }


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    return value


def _print_summary(payload: Mapping[str, Any]) -> None:
    print(f"mode = {payload['mode']}")
    print(f"status = {payload['status']}")
    print(f"artifact_written = {str(payload['artifact_written']).lower()}")
    print(f"run_directory_created = {str(payload['run_directory_created']).lower()}")
    print(f"planned_samples = {payload['planned_samples']}")
    print(f"split_scientific_hash_version = {payload['split_scientific_hash_version']}")
    print(f"split_scientific_sha256 = {payload['split_scientific_sha256']}")
    print(f"checkpoint_sha256 = {payload['checkpoint_sha256']}")
    layers = ",".join(str(layer) for layer in payload["candidate_layers"])
    print(f"candidate_layers = [{layers}]")
    print(
        RESULT_PREFIX
        + json.dumps(_thaw(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _apply_profile(repo: Path) -> Any:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from rad.runtime.execution_profile import apply_execution_profile

    try:
        return apply_execution_profile()
    except Exception as exc:
        message = str(exc)
        if "B2_BOOTSTRAP" in message or "BOOTSTRAP" in message:
            _fail("B2_CACHE_BOOTSTRAP_REQUIRED", "validated launcher bootstrap is absent")
        raise


def _resolve_production_teacher(checkpoint: Path, candidate_layers: tuple[int, ...]) -> Any:
    """Load the frozen CUDA production teacher once for an official cache run."""

    from rad.phase_b.b2_teacher_cache import ProductionTeacher

    return ProductionTeacher.load(checkpoint, candidate_layers=candidate_layers)


def _execute(args: argparse.Namespace, attestation: Any, repo: Path) -> dict[str, Any]:
    from rad.phase_b.b2_teacher_cache import (
        build_generation_plan,
        build_intended_manifest_metadata,
        claim_new_run_directory,
        generate_production_teacher_cache,
        load_teacher_cache_config,
        require_production_teacher,
        validate_checkpoint_bytes,
        validate_outer_provenance,
    )

    _validate_seed(str(args.seed))
    config = load_teacher_cache_config(Path(args.config))
    split_manifest = _load_json_object(
        str(args.split_manifest),
        missing_code="B2_CACHE_SPLIT_REQUIRED",
        invalid_code="B2_CACHE_SPLIT_REQUIRED",
    )
    checkpoint = Path(args.checkpoint)
    expected_sha = str(args.expected_checkpoint_sha256)
    if expected_sha != config.checkpoint_sha256:
        _fail("B2_CACHE_CHECKPOINT_HASH_MISMATCH", "expected checkpoint hash drifted")
    validate_checkpoint_bytes(checkpoint, expected_sha)

    output_root = Path(args.output_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_root not in output_dir.parents and output_dir != output_root:
        _fail(
            "B2_CACHE_OUTPUT_LOCATION_INVALID",
            "output-dir must live under output-root",
        )

    repository_identity = _derive_repository_identity(repo, require_clean=True)
    bootstrap_validated = os.environ.get("RAD_EXECUTION_PROFILE_BOOTSTRAPPED") == "1"
    provenance = validate_outer_provenance(
        config=config,
        bootstrap_validated=bootstrap_validated,
        execution_profile_sha256=config.execution_profile_sha256,
        runtime_attestation=attestation,
        split_manifest=split_manifest,
        checkpoint_path=checkpoint,
        b2_tag_commit=repository_identity["b2_tag_commit"],
        head_commit=repository_identity["head_commit"],
        head_is_descendant=bool(repository_identity["head_is_descendant"]),
        worktree_clean=bool(repository_identity["worktree_clean"]),
        forbidden_target_access_count=0,
    )
    plan = build_generation_plan(split_manifest, config)
    intended = build_intended_manifest_metadata(
        config=config,
        plan=plan,
        provenance=provenance,
    )

    # Collision intent: refuse an existing run directory unless resuming.
    if output_dir.exists() and not bool(args.resume):
        claim_new_run_directory(output_dir)

    if args.dry_run:
        return {
            "mode": "dry_run",
            "status": "passed",
            "artifact_written": False,
            "run_directory_created": False,
            "planned_samples": len(plan),
            "split_scientific_hash_version": config.split_scientific_hash_version,
            "split_scientific_sha256": config.split_scientific_sha256,
            "checkpoint_sha256": provenance.checkpoint_sha256,
            "candidate_layers": list(config.candidate_layers),
            "intended_manifest_metadata": _thaw(intended),
        }

    if bool(args.resume):
        _fail(
            "B2_CACHE_RESUME_NOT_ENABLED_IN_CLI",
            "B2-02B official generation uses fresh output directories only",
        )
    if output_dir.exists():
        claim_new_run_directory(output_dir)

    teacher = _resolve_production_teacher(checkpoint, config.candidate_layers)
    require_production_teacher(teacher)

    mvtec_root_text = getattr(args, "mvtec_root", None)
    if not mvtec_root_text:
        _fail(
            "B2_CACHE_MVTEC_ROOT_REQUIRED",
            "production generation requires --mvtec-root (no machine-local default)",
        )
    mvtec_root = Path(mvtec_root_text)
    if not mvtec_root.is_dir():
        _fail("B2_CACHE_MVTEC_ROOT_MISSING", f"MVTec root does not exist: {mvtec_root}")

    final = generate_production_teacher_cache(
        run_dir=output_dir,
        repo_root=repo,
        mvtec_root=mvtec_root,
        config=config,
        plan=plan,
        provenance=provenance,
        teacher=teacher,
    )
    return {
        "mode": "generate",
        "status": final.get("status", "passed"),
        "artifact_written": True,
        "run_directory_created": True,
        "planned_samples": len(plan),
        "split_scientific_hash_version": config.split_scientific_hash_version,
        "split_scientific_sha256": config.split_scientific_sha256,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "candidate_layers": list(config.candidate_layers),
        "output_dir": str(output_dir),
        "cache_scientific_sha256": final.get("cache_scientific_sha256"),
        "sample_coverage_sha256": final.get("sample_coverage_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if "-h" in arguments or "--help" in arguments:
        parser.parse_args(arguments)
        return 0

    try:
        repo = Path(__file__).resolve().parents[1]
        attestation = _apply_profile(repo)
        args = parser.parse_args(arguments)
        payload = _execute(args, attestation, repo)
    except B2TeacherCacheCLIError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        code = getattr(exc, "code", None)
        if isinstance(code, str) and code.startswith("B2_"):
            print(str(exc), file=sys.stderr)
            return 1
        raise

    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
