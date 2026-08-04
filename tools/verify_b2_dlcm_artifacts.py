#!/usr/bin/env python3
"""Verify B2-05A DLCM artifact schemas and identity bindings (contract stage)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify B2 DLCM contract artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--deployment-checkpoint", default=None)
    parser.add_argument("--seed-collection-manifest", default=None)
    parser.add_argument("--accepted-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("contract_stage") != "b2_05a":
        print("B2_DLCM_VERIFY_STAGE_INVALID", file=sys.stderr)
        return 1
    if config.get("real_training_enabled") is not False:
        print("B2_DLCM_VERIFY_REAL_TRAINING_FLAG", file=sys.stderr)
        return 1
    # Contract-stage verification: config pins only unless artifacts provided.
    required_upstream = {
        "accepted_input_contribution_plan_scientific_sha256",
        "contribution_target_collection_scientific_sha256",
        "descriptor_collection_scientific_sha256",
        "descriptor_normalization_scientific_sha256",
    }
    missing = required_upstream - set(config.get("accepted_upstream", {}))
    if missing:
        print(f"B2_DLCM_VERIFY_UPSTREAM_MISSING: {sorted(missing)}", file=sys.stderr)
        return 1
    if args.deployment_checkpoint:
        print("B2_DLCM_VERIFY_NO_REAL_CHECKPOINT_IN_05A", file=sys.stderr)
        return 1
    if args.accepted_manifest:
        print("B2_DLCM_VERIFY_NO_ACCEPTED_IN_05A", file=sys.stderr)
        return 1
    payload: dict[str, Any] = {
        "status": "contract_ok",
        "contract_stage": "b2_05a",
        "real_training_enabled": False,
        "evaluation_unlocked": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
