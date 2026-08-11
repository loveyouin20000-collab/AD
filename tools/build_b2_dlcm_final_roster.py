#!/usr/bin/env python3
"""Build untouched B2 DLCM V2 final evaluation roster (identity-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v2_final_roster as roster  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze untouched B2 DLCM final roster.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = roster.build_roster_from_official_paths(
        source_root=args.source_root,
        split_manifest_path=args.split_manifest,
        implementation_commit=args.implementation_commit,
        repo_root=args.repo_root,
    )
    digest = roster.persist_roster(args.output, payload)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "roster_scientific_sha256": payload["roster_scientific_sha256"],
                "file_sha256": digest,
                "counts": payload["counts"],
                "paths_present": payload["paths_present"],
                "final_content_resolved": payload["final_content_resolved"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except roster.B2DLCMV2FinalRosterError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
