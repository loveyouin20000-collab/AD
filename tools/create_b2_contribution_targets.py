#!/usr/bin/env python3
"""B2 contribution-target materialization CLI.

``--dry-run`` performs the complete scientific computation (GT map calibration,
all dual-family target records, the training-only Shapley normalization, every
layered identity, and the plan hash) and writes nothing at all.

Non-dry-run behavior is configuration-driven. With the tracked Gate-C
configuration it still fails closed because official materialization is
disabled; with the official B2-04B configuration it materializes one fresh,
fully verified run directory, and the repository identity gate additionally
requires the frozen contract tag, a descendant HEAD, and a clean worktree.

The CLI never loads a teacher checkpoint, never runs a backbone, never touches a
target-domain dataset, and never selects a machine-local path of its own: every
input and output location is supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RESULT_PREFIX = "B2_CONTRIBUTION_TARGETS_RESULT="

_SUMMARY_FIELDS = (
    "mode",
    "status",
    "artifact_written",
    "run_directory_created",
    "teacher_forward_count",
    "planned_samples",
    "training_targets",
    "calibration_targets",
    "evaluation_targets",
    "training_samples_for_gt_calibration",
    "calibration_samples_for_gt_calibration",
    "evaluation_samples_for_gt_calibration",
    "training_samples_for_shapley_normalization",
    "calibration_samples_for_shapley_normalization",
    "evaluation_samples_for_shapley_normalization",
    "contribution_plan_scientific_sha256",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify B2 dual contribution targets against an accepted teacher cache "
            "and descriptor collection (dry-run only in B2-04A)."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-cache-manifest", required=True)
    parser.add_argument("--teacher-cache-root", required=True)
    parser.add_argument("--descriptor-manifest", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-plan-sha256", default=None)
    return parser


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _emit(payload: Mapping[str, Any], *, stream: Any) -> None:
    print(
        RESULT_PREFIX
        + json.dumps(_thaw(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        file=stream,
    )


def _print_summary(payload: Mapping[str, Any]) -> None:
    for field in _SUMMARY_FIELDS:
        value = payload.get(field)
        rendered = str(value).lower() if isinstance(value, bool) else value
        print(f"{field} = {rendered}")
    depths = ",".join(str(depth) for depth in payload["prediction_depths"])
    print(f"prediction_depths = [{depths}]")
    counts = payload["coalition_counts"]
    rendered_counts = ", ".join(f"{depth}: {counts[depth]}" for depth in sorted(counts))
    print(f"coalition_counts = {{{rendered_counts}}}")
    _emit(payload, stream=sys.stdout)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        seed = int(args.seed)
    except ValueError:
        print("B2_CONTRIBUTION_SEED_INVALID: seed must be an integer", file=sys.stderr)
        return 2

    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    from rad.phase_b.b2_contribution_targets import (
        ContributionTargetError,
        dry_run_contribution_targets_from_roots,
        load_contribution_inputs_from_disk,
        load_contribution_targets_config,
        materialize_contribution_target_collection,
    )

    mode = "dry_run" if args.dry_run else "official"
    output_root = Path(args.output_root)
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "contribution_targets"
    try:
        config = load_contribution_targets_config(Path(args.config))
        if args.dry_run:
            payload = dry_run_contribution_targets_from_roots(
                config=config,
                teacher_cache_manifest_path=Path(args.teacher_cache_manifest),
                teacher_cache_root=Path(args.teacher_cache_root),
                descriptor_manifest_path=Path(args.descriptor_manifest),
                descriptor_root=Path(args.descriptor_root),
                seed=seed,
                output_dir=output_dir,
            )
            result = dict(payload)
            result["output_root"] = str(output_root)
            result["source_root"] = str(Path(args.mvtec_root))
            result["expected_plan_sha256_matched"] = _match_expected_plan(
                result["contribution_plan_scientific_sha256"], args.expected_plan_sha256
            )
            _print_summary(result)
            return 0

        inputs = load_contribution_inputs_from_disk(
            config=config,
            teacher_cache_manifest_path=Path(args.teacher_cache_manifest),
            teacher_cache_root=Path(args.teacher_cache_root),
            descriptor_manifest_path=Path(args.descriptor_manifest),
            descriptor_root=Path(args.descriptor_root),
        )
        materialized = materialize_contribution_target_collection(
            config=config,
            inputs=inputs,
            output_run_dir=output_dir,
            expected_plan_sha256=args.expected_plan_sha256,
            repository_root=repository_root,
        )
        official = {
            "mode": mode,
            "status": "passed",
            "artifact_written": True,
            "run_directory_created": True,
            "teacher_forward_count": 0,
            "contribution_plan_scientific_sha256": (
                materialized.contribution_plan_scientific_sha256
            ),
            "run_dir": str(materialized.run_dir),
            "output_root": str(output_root),
            "seed": seed,
        }
        _emit(official, stream=sys.stdout)
        return 0
    except ContributionTargetError as exc:
        print(str(exc), file=sys.stderr)
        print("status = failed", file=sys.stderr)
        _emit(
            {
                "mode": mode,
                "status": "failed",
                "artifact_written": False,
                "run_directory_created": False,
                "teacher_forward_count": 0,
                "error": str(exc),
                "code": exc.code,
            },
            stream=sys.stderr,
        )
        return 1


def _match_expected_plan(recomputed: str, expected: str | None) -> bool:
    """Compare a dry-run plan hash with an optional caller expectation."""

    from rad.phase_b.b2_contribution_targets import ContributionTargetError

    if expected is None:
        return False
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ContributionTargetError(
            "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MALFORMED",
            "the expected plan hash must be 64 lowercase hex characters",
        )
    if expected != recomputed:
        raise ContributionTargetError(
            "B2_CONTRIBUTION_EXPECTED_PLAN_SHA_MISMATCH",
            f"recomputed plan hash {recomputed} does not match the expected {expected}",
        )
    return True


if __name__ == "__main__":
    raise SystemExit(main())
