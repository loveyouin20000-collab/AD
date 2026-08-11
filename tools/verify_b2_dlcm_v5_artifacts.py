#!/usr/bin/env python3
"""Verify V5 JSON artifacts against .sha256 receipts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v5_protocol as protocol  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify V5 artifact receipts.")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    for path in args.paths:
        protocol.verify_json_receipt(path)
        print(f"ok {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except protocol.B2DLCMV5ProtocolError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
