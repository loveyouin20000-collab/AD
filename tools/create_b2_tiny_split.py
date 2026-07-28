#!/usr/bin/env python3
"""Create the fixed B2 Gate-C tiny split under an attested runtime."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

RESULT_PREFIX = "B2_TINY_SPLIT_RESULT="
APPROVED_SEED = 111
EXPECTED_B1_TAG = "b1-strict-independent-v1"
EXPECTED_B1_COMMIT = "3a751b2784a50eb0a08ed49e1db2df0b53608ccc"
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
VISA_CATEGORIES = frozenset(
    {
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    }
)


class B2TinySplitCLIError(RuntimeError):
    """A stable, fail-closed CLI contract error."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2TinySplitCLIError(code, detail)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the source-only B2 Gate-C deterministic tiny split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--creation-timestamp", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_config(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _fail("B2_CONFIG_NOT_FOUND", f"configuration does not exist: {path}")
    except OSError as exc:
        _fail("B2_CONFIG_READ_FAILED", f"cannot read configuration: {exc}")
    try:
        config = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        _fail("B2_CONFIG_JSON_INVALID", f"configuration is not valid JSON: {exc}")
    if not isinstance(config, dict):
        _fail("B2_CONFIG_SCHEMA_INVALID", "configuration must be a JSON object")
    if config.get("source_dataset") == "visa":
        _fail("B2_TARGET_DATASET_FORBIDDEN", "VisA cannot be used as the source")
    categories = config.get("categories")
    if isinstance(categories, list) and any(
        isinstance(category, str) and category.lower() in VISA_CATEGORIES
        for category in categories
    ):
        _fail(
            "B2_TARGET_CATEGORY_FORBIDDEN",
            "the source category list contains a VisA target category",
        )
    return config


def _validate_seed(seed_text: str) -> int:
    try:
        seed = int(seed_text)
    except ValueError:
        _fail("B2_SEED_INVALID", f"seed must be an integer: {seed_text!r}")
    if seed != APPROVED_SEED:
        _fail("B2_SEED_DRIFT", f"seed must be {APPROVED_SEED}, observed {seed}")
    return seed


def _validate_run_id(run_id: str) -> str:
    if not SAFE_RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        _fail("B2_RUN_ID_INVALID", f"unsafe run ID: {run_id!r}")
    return run_id


def _validate_creation_timestamp(value: str) -> str:
    if not UTC_TIMESTAMP.fullmatch(value):
        _fail(
            "B2_CREATION_TIMESTAMP_INVALID",
            "creation timestamp must have second-precision UTC Z format",
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        _fail("B2_CREATION_TIMESTAMP_INVALID", str(exc))
    return value


def _run_git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        _fail("B2_REPOSITORY_IDENTITY_UNAVAILABLE", f"cannot execute git: {exc}")


def _git(repo: Path, *arguments: str, allow_empty: bool = False) -> str:
    completed = _run_git(repo, *arguments)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git failed"
        _fail("B2_REPOSITORY_IDENTITY_UNAVAILABLE", detail)
    value = completed.stdout.strip()
    if not value and not allow_empty:
        _fail("B2_REPOSITORY_IDENTITY_UNAVAILABLE", "git returned an empty identity")
    return value


def _derive_repository_identity(
    repo: Path,
    specification: Mapping[str, Any],
    *,
    require_clean: bool,
) -> dict[str, Any]:
    base = specification.get("b1_base")
    if not isinstance(base, Mapping):
        _fail("B2_CONFIG_SCHEMA_INVALID", "configuration is missing b1_base")
    base_tag = base.get("tag")
    configured_base_commit = base.get("commit")
    if not isinstance(base_tag, str) or not isinstance(configured_base_commit, str):
        _fail("B2_CONFIG_SCHEMA_INVALID", "b1_base tag and commit must be strings")
    if base_tag != EXPECTED_B1_TAG or configured_base_commit != EXPECTED_B1_COMMIT:
        _fail(
            "B2_REPOSITORY_IDENTITY_MISMATCH",
            "configuration does not pin the accepted B1 tag and commit",
        )

    worktree = Path(_git(repo, "rev-parse", "--show-toplevel")).resolve()
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current", allow_empty=True)
    derived_base_commit = _git(repo, "rev-parse", f"{base_tag}^{{commit}}")
    if derived_base_commit != configured_base_commit:
        _fail(
            "B2_REPOSITORY_IDENTITY_MISMATCH",
            "configured base commit does not resolve from the configured base tag",
        )
    ancestry = _run_git(repo, "merge-base", "--is-ancestor", configured_base_commit, head)
    if ancestry.returncode == 1:
        _fail(
            "B2_REPOSITORY_ANCESTRY_MISMATCH",
            "current HEAD is not a descendant of the accepted B1 commit",
        )
    if ancestry.returncode != 0:
        detail = ancestry.stderr.strip() or ancestry.stdout.strip() or "git failed"
        _fail("B2_REPOSITORY_IDENTITY_UNAVAILABLE", detail)
    worktree_clean = not bool(
        _git(repo, "status", "--porcelain", "--untracked-files=all", allow_empty=True)
    )
    if require_clean and not worktree_clean:
        _fail(
            "B2_WORKTREE_DIRTY",
            "authoritative official generation requires a clean worktree",
        )
    identity = {
        "b1_base_tag": base_tag,
        "b1_base_commit": derived_base_commit,
        "generation_git_commit": head,
        "generation_branch": branch,
        "worktree_clean": worktree_clean,
        "worktree_path": str(worktree),
    }
    _validate_repository_identity(identity, observed_head=head)
    return identity


def _validate_repository_identity(
    identity: Mapping[str, Any],
    *,
    observed_head: str,
) -> None:
    if (
        identity.get("b1_base_tag") != EXPECTED_B1_TAG
        or identity.get("b1_base_commit") != EXPECTED_B1_COMMIT
        or identity.get("generation_git_commit") != observed_head
    ):
        _fail(
            "B2_REPOSITORY_IDENTITY_MISMATCH",
            "manifest repository provenance differs from observed Git identity",
        )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    return value


def _result_payload(
    *,
    dry_run: bool,
    manifest: Mapping[str, Any],
    scientific_content: Mapping[str, Any],
    scientific_sha256: str,
    attestation: Any,
    source_snapshot: Any,
) -> dict[str, Any]:
    split_counts = {
        name: len(manifest["splits"][name])
        for name in ("training", "calibration", "evaluation")
    }
    audits = {
        "count_audit": manifest["count_audit"]["passed"],
        "overlap_audit": manifest["overlap_audit"]["passed"],
        "source_only_audit": manifest["source_only_audit"]["passed"],
        "fixture_path_audit": manifest["fixture_path_audit"]["passed"],
        "mask_audit": manifest["mask_audit"]["passed"],
    }
    validation = {
        "execution_profile_attestation": _thaw(attestation.canonical_attestation()),
        "production_adapter_enumeration": {
            "dataset": source_snapshot.source_dataset,
            "root": source_snapshot.resolved_root,
            "split": source_snapshot.source_split,
            "adapter_module": source_snapshot.adapter_module,
            "adapter_class": source_snapshot.adapter_class,
            "record_count": len(source_snapshot.canonical_records),
        },
        "selection": {
            "split_counts": split_counts,
            "selected_count": sum(split_counts.values()),
        },
        "audits": audits,
    }
    return {
        "mode": "dry-run" if dry_run else "official",
        "official_manifest": _thaw(manifest),
        "canonical_scientific_content_v2": _thaw(scientific_content),
        "canonical_scientific_hash_v2": scientific_sha256,
        "canonical_scientific_content": _thaw(scientific_content),
        "canonical_scientific_sha256": scientific_sha256,
        "validation": validation,
    }


def _execute(
    args: argparse.Namespace,
    attestation: Any,
    repo: Path,
) -> dict[str, Any]:
    from rad.artifacts import atomic_write_json
    from rad.phase_b.b2_tiny_split import (
        build_split_manifest,
        canonical_scientific_content_v2,
        collect_source_records,
    )

    specification = _load_config(str(args.config))
    _validate_seed(str(args.seed))
    run_id = _validate_run_id(str(args.run_id))
    creation_timestamp = _validate_creation_timestamp(str(args.creation_timestamp))

    approved_output = repo / "artifacts" / "phase_b" / "b2_gate_c"
    output_base = Path(args.output_dir).resolve()
    if output_base != approved_output.resolve():
        _fail(
            "B2_OUTPUT_LOCATION_INVALID",
            f"output base must be exactly {approved_output.resolve()}",
        )

    repository_identity = _derive_repository_identity(
        repo,
        specification,
        require_clean=not bool(args.dry_run),
    )
    run_directory = output_base / run_id
    if not args.dry_run and run_directory.exists():
        _fail("B2_OUTPUT_COLLISION", f"run directory already exists: {run_directory}")

    source_snapshot = collect_source_records(
        source_root=Path(args.dataset_root),
        specification=specification,
    )
    built = build_split_manifest(
        specification=specification,
        source_snapshot=source_snapshot,
        runtime_attestation=attestation,
        repository_identity=repository_identity,
        run_metadata={
            "run_id": run_id,
            "creation_timestamp": creation_timestamp,
            "output_directory": str(run_directory.resolve()),
        },
    )
    scientific_content = canonical_scientific_content_v2(built.manifest)
    payload = _result_payload(
        dry_run=bool(args.dry_run),
        manifest=built.manifest,
        scientific_content=scientific_content,
        scientific_sha256=built.scientific_sha256,
        attestation=attestation,
        source_snapshot=source_snapshot,
    )
    if not args.dry_run:
        final_repository_identity = _derive_repository_identity(
            repo,
            specification,
            require_clean=True,
        )
        if final_repository_identity != repository_identity:
            _fail(
                "B2_REPOSITORY_IDENTITY_MISMATCH",
                "repository identity changed during official generation",
            )
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
            atomic_write_json(run_directory / "split_manifest.json", _thaw(built.manifest))
        except FileExistsError:
            _fail("B2_OUTPUT_COLLISION", f"run directory already exists: {run_directory}")
        except Exception as exc:
            _fail("B2_ATOMIC_WRITE_FAILED", f"official manifest publication failed: {exc}")
    return payload


def _apply_profile(repo: Path) -> Any:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from rad.runtime.execution_profile import apply_execution_profile

    return apply_execution_profile()


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
    except B2TinySplitCLIError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        code = getattr(exc, "code", None)
        if isinstance(code, str) and code.startswith("B2_"):
            print(str(exc), file=sys.stderr)
            return 1
        raise

    print(
        RESULT_PREFIX
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
