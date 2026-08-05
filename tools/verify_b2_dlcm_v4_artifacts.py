#!/usr/bin/env python3
"""Verify B2 DLCM V3 artifact receipts and schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v4_protocol as protocol  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify B2 DLCM V3 artifacts.")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--require-schema-prefix", default="b2_dlcm_v4_")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.artifact:
        raise protocol.B2DLCMV4ProtocolError(
            "B2_DLCM_V4_CONTRACT_MISMATCH",
            "at least one --artifact required",
        )
    results = []
    for raw in args.artifact:
        payload = protocol.verify_json_receipt(raw)
        schema = str(payload.get("schema_version", ""))
        if args.require_schema_prefix and not schema.startswith(args.require_schema_prefix):
            raise protocol.B2DLCMV4ProtocolError(
                "B2_DLCM_V4_CONTRACT_MISMATCH",
                f"schema_version {schema!r} missing prefix {args.require_schema_prefix!r}",
            )
        results.append({"path": raw, "schema_version": schema, "ok": True})
    print(json.dumps({"status": "ok", "artifacts": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except protocol.B2DLCMV4ProtocolError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
