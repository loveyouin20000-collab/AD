#!/usr/bin/env python3
"""Evaluate B2 DLCM V2 final set (locked until unlock)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v2_protocol as protocol  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate B2 DLCM V2 final set.")
    parser.add_argument("--materialization-manifest", required=True)
    parser.add_argument("--unlock", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    unlock = None
    if args.unlock:
        unlock = json.loads(Path(args.unlock).read_text(encoding="utf-8"))
    protocol.require_evaluation_unlock(unlock)
    protocol.forbid_final_content_access(unlocked=False, context="evaluate_cli_c1a")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except protocol.B2DLCMV2ProtocolError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
