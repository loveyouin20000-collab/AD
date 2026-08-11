#!/usr/bin/env python3
"""B2-05C4A V5 calibration CLI (dry-run contract validation by default)."""

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

from rad.phase_b import b2_dlcm_v5_calibration as calibration  # noqa: E402
from rad.phase_b import b2_dlcm_v5_protocol as protocol  # noqa: E402

_SUMMARY_FIELDS = (
    "real_training_started",
    "calibration_started",
    "development_evaluation_started",
    "final_content_resolved",
    "final_materialization_started",
    "final_evaluation_started",
    "artifact_written",
    "run_directory_created",
    "teacher_forward_count",
)


class B2DLCMV5CLIError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B2 DLCM V5 calibration CLI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--process-label", default="A", choices=["A", "B"])
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol.reject_bypass_flags(payload)
    if payload.get("contract_stage") != "b2_05c4a":
        raise B2DLCMV5CLIError("B2_DLCM_V5_CONTRACT_MISMATCH", "contract_stage invalid")
    required = (
        "real_training_enabled",
        "calibration_enabled",
        "beta_grid_size",
        "loo_depth",
        "canonical_seed",
        "adopted_final_roster_scientific_sha256",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise B2DLCMV5CLIError("B2_DLCM_V5_CONTRACT_MISMATCH", f"missing keys {missing}")
    return payload


def _print_summary(payload: Mapping[str, Any]) -> None:
    for field in _SUMMARY_FIELDS:
        value = payload[field]
        rendered = str(value).lower() if isinstance(value, bool) else value
        print(f"{field} = {rendered}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config(Path(args.config))
    if not args.dry_run:
        protocol.forbid_training(context="calibrate_cli_non_dry")
    if config.get("real_training_enabled") is True:
        protocol.forbid_training(context="config_real_training_enabled")
    result = calibration.dry_run_complete_v5_contract_validation(
        config=config,
        output_dir=args.output_dir,
    )
    _print_summary(result)
    print(f"process_label = {args.process_label}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        B2DLCMV5CLIError,
        protocol.B2DLCMV5ProtocolError,
        calibration.B2DLCMV5CalibrationError,
    ) as exc:
        code = getattr(exc, "code", "B2_DLCM_V5_CONTRACT_MISMATCH")
        print(f"ERROR {code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
