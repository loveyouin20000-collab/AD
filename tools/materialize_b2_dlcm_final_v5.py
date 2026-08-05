#!/usr/bin/env python3
"""Fail-closed Final materialization stub for V5 C4A."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v5_protocol as protocol  # noqa: E402


def main() -> int:
    protocol.forbid_final_content_access(unlocked=False, context="materialize_v5_c4a")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except protocol.B2DLCMV5ProtocolError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
