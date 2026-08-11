#!/usr/bin/env python3
"""B2 DLCM training CLI (B2-05A contract dry-run + B2-05B official path)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RESULT_PREFIX = "B2_DLCM_TRAINING_RESULT="

_SUMMARY_FIELDS = (
    "mode",
    "status",
    "artifact_written",
    "run_directory_created",
    "real_training_started",
    "evaluation_unlocked",
    "teacher_forward_count",
)


class B2DLCMCLIError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B2 DLCM training CLI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--descriptor-manifest", required=True)
    parser.add_argument("--descriptor-root", required=True)
    parser.add_argument("--contribution-target-manifest", required=True)
    parser.add_argument("--contribution-target-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-plan-sha256", default=None)
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stage = payload.get("contract_stage")
    if stage not in {"b2_05a", "b2_05b"}:
        raise B2DLCMCLIError("B2_DLCM_CONFIG_STAGE_INVALID", "contract_stage must be b2_05a|b2_05b")
    required = (
        "real_training_enabled",
        "candidate_layers",
        "prediction_depths",
        "seeds",
        "accepted_upstream",
        "authoritative_base_commit",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise B2DLCMCLIError("B2_DLCM_CONFIG_INCOMPLETE", f"missing keys {missing}")
    return payload


def _validate_paths(args: argparse.Namespace) -> None:
    for label, raw in (
        ("config", args.config),
        ("descriptor-manifest", args.descriptor_manifest),
        ("descriptor-root", args.descriptor_root),
        ("contribution-target-manifest", args.contribution_target_manifest),
        ("contribution-target-root", args.contribution_target_root),
    ):
        path = Path(raw)
        if label.endswith("root"):
            if not path.is_dir():
                raise B2DLCMCLIError("B2_DLCM_PATH_INVALID", f"{label} must be a directory")
        elif not path.is_file():
            raise B2DLCMCLIError("B2_DLCM_PATH_INVALID", f"{label} must be a file")


def _validate_common(config: Mapping[str, Any], seed: int) -> None:
    if config["authoritative_base_commit"] != (
        "97a4f497f6f2b096dd4a339555f81e7296ec3035"
    ):
        raise B2DLCMCLIError("B2_DLCM_BASE_COMMIT_MISMATCH", "authoritative base commit mismatch")
    if list(config["candidate_layers"]) != [6, 12, 18, 24]:
        raise B2DLCMCLIError("B2_DLCM_LAYER_CONTRACT", "candidate_layers mismatch")
    if list(config["prediction_depths"]) != [12, 18, 24]:
        raise B2DLCMCLIError("B2_DLCM_DEPTH_CONTRACT", "prediction_depths mismatch")
    if list(config["seeds"]) != [17, 29, 43]:
        raise B2DLCMCLIError("B2_DLCM_SEED_CONTRACT", "seeds must be [17,29,43]")
    if seed not in set(config["seeds"]):
        raise B2DLCMCLIError("B2_DLCM_SEED_UNKNOWN", f"seed {seed} not in contract")


def _emit(payload: Mapping[str, Any]) -> None:
    print(
        RESULT_PREFIX
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _print_summary(payload: Mapping[str, Any]) -> None:
    for field in _SUMMARY_FIELDS:
        value = payload[field]
        rendered = str(value).lower() if isinstance(value, bool) else value
        print(f"{field} = {rendered}")
    # Optional official fields.
    for field in (
        "accepted_dlcm_training_plan_scientific_sha256",
        "training_records",
        "calibration_records",
        "evaluation_records_declared",
        "evaluation_records_loaded",
    ):
        if field in payload:
            print(f"{field} = {payload[field]}")
    _emit(payload)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = _load_config(Path(args.config))
        _validate_paths(args)
        seed = int(args.seed)
        _validate_common(config, seed)
        stage = config["contract_stage"]

        if stage == "b2_05a":
            if config.get("real_training_enabled") is not False:
                raise B2DLCMCLIError(
                    "B2_DLCM_REAL_TRAINING_FLAG",
                    "B2-05A requires real_training_enabled=false",
                )
            if not args.dry_run:
                print("B2_DLCM_REAL_TRAINING_NOT_ENABLED", file=sys.stderr)
                print("B2_DLCM_REAL_TRAINING_NOT_ENABLED")
                return 2
            from rad.phase_b import b2_dlcm_training as training

            output_root = Path(args.output_root)
            result = training.dry_run_complete_contract_validation(
                config=config,
                seed=seed,
                output_root=output_root,
                descriptor_manifest=args.descriptor_manifest,
                descriptor_root=args.descriptor_root,
                contribution_target_manifest=args.contribution_target_manifest,
                contribution_target_root=args.contribution_target_root,
            )
            if output_root.exists() and any(output_root.iterdir()):
                raise B2DLCMCLIError("B2_DLCM_DRY_RUN_WRITE", "dry-run must not create output files")
            payload = {
                "mode": "dry_run",
                "status": result["status"],
                "artifact_written": False,
                "run_directory_created": False,
                "real_training_started": False,
                "evaluation_unlocked": False,
                "teacher_forward_count": 0,
                "hermetic_records_validated": result["hermetic_records_validated"],
                "seed": seed,
                "contract_stage": stage,
                "output_root_exists_before": output_root.exists(),
            }
            _print_summary(payload)
            return 0

        # b2_05b official path
        if config.get("real_training_enabled") is not True:
            raise B2DLCMCLIError(
                "B2_DLCM_REAL_TRAINING_FLAG",
                "B2-05B requires real_training_enabled=true",
            )
        from rad.phase_b import b2_dlcm_official as official

        if args.dry_run:
            result = official.official_dry_run(
                config=config,
                descriptor_manifest=args.descriptor_manifest,
                descriptor_root=args.descriptor_root,
                contribution_target_manifest=args.contribution_target_manifest,
                contribution_target_root=args.contribution_target_root,
                output_root=args.output_root,
                seed=seed,
                expected_plan_sha256=args.expected_plan_sha256,
                repo_root=_REPO_ROOT,
            )
            out = Path(args.output_root)
            if out.exists() and any(out.iterdir()):
                raise B2DLCMCLIError("B2_DLCM_DRY_RUN_WRITE", "dry-run must not create output files")
            _print_summary(result)
            return 0

        # Official non-dry-run: single-seed training entry (collection driver separate).
        official.verify_repository_identity_gate(config=config, repo_root=_REPO_ROOT)
        from rad.phase_b import b2_dlcm_training as training

        verified = training.load_verified_b2_dlcm_training_inputs(
            descriptor_manifest=args.descriptor_manifest,
            descriptor_root=args.descriptor_root,
            contribution_target_manifest=args.contribution_target_manifest,
            contribution_target_root=args.contribution_target_root,
            accepted_upstream=dict(config["accepted_upstream"]),
            evaluation_unlocked=False,
        )
        plan_sha = official.compute_accepted_dlcm_training_plan_scientific_sha256(
            config=config, verified=verified
        )
        if not config.get("expected_accepted_training_plan_sha256"):
            raise B2DLCMCLIError(
                "B2_DLCM_PLAN_SHA_REQUIRED",
                "expected_accepted_training_plan_sha256 must be pinned before training",
            )
        if args.expected_plan_sha256 is None:
            raise B2DLCMCLIError(
                "B2_DLCM_PLAN_SHA_REQUIRED",
                "--expected-plan-sha256 required for official training",
            )
        official.require_plan_sha_agreement(
            config=config,
            recomputed=plan_sha,
            cli_expected=args.expected_plan_sha256,
        )
        output_root = Path(args.output_root)
        env = training.collect_environment_contract(allow_cpu_for_hermetic=False)
        train_ns = official.records_as_namespaces(verified.training_records)
        cal_ns = official.records_as_namespaces(verified.calibration_records)
        result = official.run_official_seed_training(
            output_root=output_root,
            seed=seed,
            training_records=train_ns,
            calibration_records=cal_ns,
            environment_contract=env,
            maximum_epochs=int(config["maximum_epochs"]),
            patience=int(config["patience"]),
            batch_size=int(config["batch_size"]),
            device="cuda",
        )
        payload = {
            "mode": "official_seed",
            "status": result["status"],
            "artifact_written": True,
            "run_directory_created": True,
            "real_training_started": True,
            "evaluation_unlocked": False,
            "teacher_forward_count": 0,
            "seed": seed,
            "best_epoch": result.get("best_epoch"),
            "accepted_dlcm_training_plan_scientific_sha256": plan_sha,
            "contract_stage": stage,
        }
        _print_summary(payload)
        return 0 if result["status"] != "failed" else 3
    except B2DLCMCLIError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface official module codes
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
