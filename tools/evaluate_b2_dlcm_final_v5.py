#!/usr/bin/env python3
"""V5 Final evaluation CLI guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v5_protocol as protocol  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B2 DLCM V5 Final evaluation tooling.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        print("evaluation_started = false")
        return 0
    protocol.forbid_final_content_access(unlocked=False, context="evaluate_v5_requires_unlock")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except protocol.B2DLCMV5ProtocolError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
