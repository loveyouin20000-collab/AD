#!/usr/bin/env python3
"""B2-03A descriptor-artifacts CLI with disk-authoritative validation parity."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

RESULT_PREFIX = "B2_DESCRIPTOR_ARTIFACTS_RESULT="


class B2DescriptorArtifactsCLIError(RuntimeError):
    """A stable, fail-closed CLI contract error."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DescriptorArtifactsCLIError(code, detail)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an accepted teacher-cache and plan B2 descriptor artifacts "
            "(dry-run only in B2-03A)."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--teacher-cache-manifest", required=True)
    parser.add_argument("--teacher-cache-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


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
        _fail("B2_DESC_REPOSITORY_IDENTITY_UNAVAILABLE", f"cannot execute git: {exc}")


def _git(repo: Path, *arguments: str, allow_empty: bool = False) -> str:
    completed = _run_git(repo, *arguments)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git failed"
        _fail("B2_DESC_REPOSITORY_IDENTITY_UNAVAILABLE", detail)
    value = completed.stdout.strip()
    if not value and not allow_empty:
        _fail("B2_DESC_REPOSITORY_IDENTITY_UNAVAILABLE", "git returned an empty identity")
    return value


def _validate_main_ancestry(
    repo: Path,
    *,
    expected_main_tag: str,
    expected_main_commit: str,
) -> dict[str, Any]:
    tag_commit = _git(repo, "rev-parse", f"{expected_main_tag}^{{commit}}")
    if tag_commit != expected_main_commit:
        _fail(
            "B2_DESC_MAIN_TAG_MOVED",
            f"tag {expected_main_tag} resolves to {tag_commit}, expected {expected_main_commit}",
        )
    head = _git(repo, "rev-parse", "HEAD")
    ancestry = _run_git(repo, "merge-base", "--is-ancestor", expected_main_commit, head)
    if ancestry.returncode == 1:
        _fail(
            "B2_DESC_MAIN_HEAD_NOT_DESCENDANT",
            "HEAD is not a descendant of the expected main integration commit",
        )
    if ancestry.returncode != 0:
        detail = ancestry.stderr.strip() or ancestry.stdout.strip() or "git failed"
        _fail("B2_DESC_REPOSITORY_IDENTITY_UNAVAILABLE", detail)
    porcelain = _git(
        repo, "status", "--porcelain", "--untracked-files=all", allow_empty=True
    )
    if porcelain:
        _fail("B2_DESC_WORKTREE_DIRTY", "official worktree is dirty")
    observed_head = _git(repo, "rev-parse", "HEAD")
    if observed_head != head:
        _fail(
            "B2_DESC_REPOSITORY_IDENTITY_MISMATCH",
            f"observed HEAD {observed_head} differs from captured HEAD {head}",
        )
    return {
        "expected_main_tag": expected_main_tag,
        "expected_main_commit": expected_main_commit,
        "head_commit": head,
        "head_is_descendant": True,
        "worktree_clean": True,
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
    print(f"teacher_forward_count = {payload['teacher_forward_count']}")
    print(f"planned_samples = {payload['planned_samples']}")
    print(
        "training_samples_for_normalization = "
        f"{payload['training_samples_for_normalization']}"
    )
    print(
        "calibration_samples_for_normalization = "
        f"{payload['calibration_samples_for_normalization']}"
    )
    print(
        "evaluation_samples_for_normalization = "
        f"{payload['evaluation_samples_for_normalization']}"
    )
    depths = ",".join(str(depth) for depth in payload["prediction_depths"])
    print(f"prediction_depths = [{depths}]")
    print(f"descriptor_dimension = {payload['descriptor_dimension']}")
    print(
        RESULT_PREFIX
        + json.dumps(_thaw(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        seed = int(args.seed)
    except ValueError:
        print("B2_DESC_SEED_INVALID: seed must be an integer", file=sys.stderr)
        return 2

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from rad.phase_b.b2_descriptor_artifacts import (
        DescriptorArtifactsError,
        build_descriptor_extraction_plan,
        load_descriptor_artifacts_config,
        load_and_validate_accepted_teacher_cache_from_disk,
        materialize_descriptor_artifact_collection,
    )

    try:
        config = load_descriptor_artifacts_config(Path(args.config))
        _validate_main_ancestry(
            repo,
            expected_main_tag=config.expected_main_tag,
            expected_main_commit=config.expected_main_commit,
        )
        validated = load_and_validate_accepted_teacher_cache_from_disk(
            config=config,
            teacher_cache_manifest_path=Path(args.teacher_cache_manifest),
            teacher_cache_root=Path(args.teacher_cache_root),
        )
        plan = build_descriptor_extraction_plan(
            accepted=validated.accepted,
            config=config,
        )
        output_root = Path(args.output_root)
        output_dir = Path(args.output_dir)
        _ = seed, output_root

        payload = {
            "mode": "dry_run" if args.dry_run else "production",
            "status": "passed",
            "artifact_written": False,
            "teacher_forward_count": 0,
            "planned_samples": plan["planned_samples"],
            "training_samples_for_normalization": plan[
                "training_samples_for_normalization"
            ],
            "calibration_samples_for_normalization": plan[
                "calibration_samples_for_normalization"
            ],
            "evaluation_samples_for_normalization": plan[
                "evaluation_samples_for_normalization"
            ],
            "prediction_depths": list(plan["prediction_depths"]),
            "descriptor_dimension": plan["descriptor_dimension"],
            "candidate_layers": list(plan["candidate_layers"]),
            "intended_descriptor_sample_coverage_inputs": plan[
                "intended_descriptor_sample_coverage_inputs"
            ],
            "intended_normalization_training_coverage_inputs": plan[
                "intended_normalization_training_coverage_inputs"
            ],
            "seed": seed,
            "output_dir": str(output_dir),
            "output_root": str(output_root),
            "source_teacher_cache_manifest_file_sha256": (
                validated.source_teacher_cache_manifest_file_sha256
            ),
        }
        if not args.dry_run:
            result = materialize_descriptor_artifact_collection(
                config=config,
                teacher_cache_manifest_path=Path(args.teacher_cache_manifest),
                teacher_cache_root=Path(args.teacher_cache_root),
                output_run_dir=output_dir,
                validated_teacher_cache=validated,
            )
            payload["artifact_written"] = True
            payload["source_teacher_cache_manifest_file_sha256"] = (
                result.source_teacher_cache_manifest_file_sha256
            )
        _print_summary(payload)
        return 0
    except DescriptorArtifactsError as exc:
        print(str(exc), file=sys.stderr)
        print(f"status = failed", file=sys.stderr)
        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "mode": "dry_run" if args.dry_run else "production",
                    "status": "failed",
                    "artifact_written": False,
                    "teacher_forward_count": 0,
                    "error": str(exc),
                    "code": exc.code,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    except B2DescriptorArtifactsCLIError as exc:
        print(str(exc), file=sys.stderr)
        print("status = failed", file=sys.stderr)
        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "mode": "dry_run" if args.dry_run else "production",
                    "status": "failed",
                    "artifact_written": False,
                    "teacher_forward_count": 0,
                    "error": str(exc),
                    "code": exc.code,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
