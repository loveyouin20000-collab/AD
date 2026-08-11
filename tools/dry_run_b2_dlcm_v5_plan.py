#!/usr/bin/env python3
"""Dual-process dry-run helper for accepted V5 calibration plan SHA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v5_official as official  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--descriptor-manifest", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--contribution-target-manifest", required=True)
    parser.add_argument("--contribution-target-root", required=True)
    parser.add_argument("--source-deployment-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-plan-sha256", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = official.official_v5_dry_run(
        config=config,
        descriptor_manifest=args.descriptor_manifest,
        descriptor_root=args.descriptor_root,
        contribution_target_manifest=args.contribution_target_manifest,
        contribution_target_root=args.contribution_target_root,
        source_deployment_checkpoint=args.source_deployment_checkpoint,
        output_root=args.output_root,
        expected_plan_sha256=args.expected_plan_sha256,
        repo_root=_REPO_ROOT,
    )
    for field in (
        "mode",
        "artifact_written",
        "run_directory_created",
        "real_training_started",
        "calibration_started",
        "development_evaluation_started",
        "final_content_resolved",
        "final_materialization_started",
        "final_evaluation_started",
        "teacher_forward_count",
        "beta_grid_count",
        "beta_grid_start",
        "beta_grid_stop",
        "beta_grid_step",
        "calibration_records",
        "loo_fold_count",
        "development_records_declared",
        "final_roster_records_declared",
        "final_records_loaded",
        "source_deployment_identity",
        "source_canonical_seed",
        "accepted_v5_calibration_plan_scientific_sha256",
    ):
        value = summary[field]
        rendered = str(value).lower() if isinstance(value, bool) else value
        print(f"{field} = {rendered}")
    print("B2_DLCM_V5_PLAN_RESULT=" + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", type(exc).__name__)
        print(f"ERROR {code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
