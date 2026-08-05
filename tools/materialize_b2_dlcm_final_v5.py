#!/usr/bin/env python3
"""V5 Final materialization CLI.

C4C supports dry-run plan validation only. Real Final content remains guarded
until a later stage supplies a valid unlock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v5_final_unlock as final_unlock  # noqa: E402
from rad.phase_b import b2_dlcm_v5_protocol as protocol  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B2 DLCM V5 Final materialization tooling.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-plan-sha256", default=None)
    return parser


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        protocol.forbid_final_content_access(unlocked=False, context="materialize_v5_no_unlock")
        raise AssertionError("unreachable after fail-closed guard")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.dry_run:
        protocol.forbid_final_content_access(unlocked=False, context="materialize_v5_requires_unlock")
    config = _load_config(args.config)
    config = {**config, "tooling_commit": config.get("tooling_commit", "dry-run-tooling-commit")}
    plan = final_unlock.build_final_execution_plan(
        config=config,
        repo_identity={"head": str(config["tooling_commit"])},
    )
    plan_sha = final_unlock.final_execution_plan_sha256(plan)
    expected_plan_sha = args.expected_plan_sha256 or config.get("expected_accepted_v5_final_execution_plan_sha256")
    if expected_plan_sha is not None and expected_plan_sha != plan_sha:
        raise final_unlock.B2DLCMV5FinalUnlockError(
            "B2_DLCM_FINAL_EXECUTION_PLAN_MISMATCH",
            "CLI SHA != recomputed Final execution plan SHA",
        )
    print(json.dumps(final_unlock.dry_run_status(plan_sha256=plan_sha), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (protocol.B2DLCMV5ProtocolError, final_unlock.B2DLCMV5FinalUnlockError) as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
