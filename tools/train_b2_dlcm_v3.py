#!/usr/bin/env python3
"""B2-05C2A V3 DLCM training CLI (dry-run contract validation only)."""

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

from rad.phase_b import b2_dlcm_v3_protocol as protocol  # noqa: E402
from rad.phase_b import b2_dlcm_v3_training as training  # noqa: E402

RESULT_PREFIX = "B2_DLCM_V3_TRAINING_RESULT="

_SUMMARY_FIELDS = (
    "real_training_started",
    "development_evaluation_started",
    "final_content_resolved",
    "final_materialization_started",
    "final_evaluation_started",
    "artifact_written",
    "run_directory_created",
    "teacher_forward_count",
)


class B2DLCMV3CLIError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B2 DLCM V3 training CLI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol.reject_bypass_flags(payload)
    if payload.get("contract_stage") != "b2_05c2a":
        raise B2DLCMV3CLIError("B2_DLCM_V3_CONTRACT_MISMATCH", "contract_stage invalid")
    required = (
        "real_training_enabled",
        "candidate_layers",
        "prediction_depths",
        "seeds",
        "smoothmax_tau",
        "authoritative_v2_contract_commit",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise B2DLCMV3CLIError("B2_DLCM_V3_CONTRACT_MISMATCH", f"missing keys {missing}")
    return payload


def _print_summary(payload: Mapping[str, Any]) -> None:
    for field in _SUMMARY_FIELDS:
        value = payload[field]
        rendered = str(value).lower() if isinstance(value, bool) else value
        print(f"{field} = {rendered}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config(Path(args.config))
    seed = int(args.seed)
    if seed not in set(config["seeds"]):
        raise B2DLCMV3CLIError("B2_DLCM_V3_CONTRACT_MISMATCH", f"seed {seed} not in contract")
    if config["authoritative_v2_contract_commit"] != (
        "e54f2b44eeb962b05cfb7cf74764e55905f1a8f6"
    ):
        raise B2DLCMV3CLIError("B2_DLCM_V3_CONTRACT_MISMATCH", "v2 contract commit mismatch")

    protocol.require_real_training_enabled(config, dry_run=bool(args.dry_run))
    if not args.dry_run:
        raise B2DLCMV3CLIError(
            "B2_DLCM_V3_REAL_TRAINING_NOT_ENABLED",
            "C2A only supports --dry-run while real_training_enabled=false",
        )

    summary = training.dry_run_complete_v3_contract_validation(
        config=config,
        seed=seed,
        output_dir=args.output_dir,
    )
    _print_summary(summary)
    print(RESULT_PREFIX + json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        B2DLCMV3CLIError,
        protocol.B2DLCMV3ProtocolError,
        training.B2DLCMV3TrainingError,
    ) as exc:
        code = getattr(exc, "code", "B2_DLCM_V3_CONTRACT_MISMATCH")
        print(f"ERROR {code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
