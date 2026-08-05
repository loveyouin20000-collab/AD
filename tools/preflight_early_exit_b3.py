#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.phase_b import b3_early_exit_gate as gate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B3 early-exit accepted LSE preflight")
    p.add_argument("--config", type=Path, default=Path("configs/rad/early_exit_b3_accepted_lse.yaml"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = gate.load_early_exit_preflight_config(args.config, repo_root=REPO_ROOT)
    report = gate.run_early_exit_preflight(cfg)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
