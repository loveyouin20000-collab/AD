#!/usr/bin/env python3
"""B2-05A DLCM training contract CLI (dry-run / contract validation only)."""

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
    parser = argparse.ArgumentParser(
        description="B2 DLCM training contract CLI (real training disabled in B2-05A)."
    )
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
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract_stage") != "b2_05a":
        raise B2DLCMCLIError("B2_DLCM_CONFIG_STAGE_INVALID", "contract_stage must be b2_05a")
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
    # output-root may not exist yet; dry-run must not create it.


def _validate_contract(config: Mapping[str, Any], seed: int) -> None:
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
    if seed not in config["seeds"] and seed not in {17, 29, 43}:
        raise B2DLCMCLIError("B2_DLCM_SEED_UNKNOWN", f"seed {seed} not in contract")
    # Import architecture helpers to ensure contract modules load.
    from rad.phase_b import b2_dlcm as dlcm
    from rad.phase_b import b2_dlcm_deployment as deployment
    from rad.phase_b import b2_dlcm_training as training

    _ = dlcm.B2DLCM
    _ = training.ExplicitLRSchedule
    _ = deployment.LOADER_CONTRACT_VERSION
    if config.get("real_training_enabled") is not False:
        raise B2DLCMCLIError("B2_DLCM_REAL_TRAINING_FLAG", "B2-05A requires real_training_enabled=false")


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
    _emit(payload)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = _load_config(Path(args.config))
        _validate_paths(args)
        seed = int(args.seed)
        _validate_contract(config, seed)
        if not args.dry_run:
            print("B2_DLCM_REAL_TRAINING_NOT_ENABLED", file=sys.stderr)
            print("B2_DLCM_REAL_TRAINING_NOT_ENABLED")
            return 2
        # Dry-run: complete hermetic contract validation, write nothing.
        from rad.phase_b import b2_dlcm_training as training

        output_root = Path(args.output_root)
        result = training.dry_run_complete_contract_validation(
            config=config,
            seed=seed,
            output_root=output_root,
        )
        if output_root.exists() and any(output_root.iterdir()) if output_root.exists() else False:
            # Dry-run must not create the output root.
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
            "contract_stage": config["contract_stage"],
            "output_root_exists_before": output_root.exists(),
        }
        _print_summary(payload)
        return 0
    except B2DLCMCLIError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
