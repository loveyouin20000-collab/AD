#!/usr/bin/env python3
"""Evaluate B2 DLCM V3 final content (locked in C2A)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v3_protocol as protocol  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate B2 DLCM V3 final content.")
    parser.add_argument("--materialization", required=True)
    parser.add_argument("--unlock", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    _parser().parse_args(argv)
    protocol.forbid_final_content_access(unlocked=False, context="evaluate_final_v3")
    raise protocol.B2DLCMV3ProtocolError(
        "B2_DLCM_FINAL_EVALUATION_MISMATCH",
        "C2A forbids final evaluation",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except protocol.B2DLCMV3ProtocolError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
