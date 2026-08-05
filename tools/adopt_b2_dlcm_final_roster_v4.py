#!/usr/bin/env python3
"""Adopt untouched C1 final roster for V3 (no reselection)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rad.phase_b import b2_dlcm_v4_protocol as protocol  # noqa: E402
from rad.phase_b import b2_dlcm_v4_roster_adoption as adoption  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adopt C1 final roster for V3.")
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "docs/phase_b/b2_05c3_final_roster_adoption_manifest.json"),
    )
    parser.add_argument("--implementation-commit", default=None)
    return parser


def _git_head(repo_root: Path) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True)
        .strip()
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(args.repo_root)
    commit = args.implementation_commit or _git_head(repo_root)
    manifest = adoption.build_adoption_manifest(
        repo_root=repo_root,
        implementation_commit=commit,
    )
    roster = adoption.load_and_verify_c1_roster(repo_root)
    adoption.assert_adoption_matches_roster(manifest, roster)
    digest = protocol.write_json_with_receipt(args.output, manifest)
    print(f"wrote {args.output}")
    print(f"sha256 = {digest}")
    print(f"source_roster_scientific_sha256 = {manifest['source_roster_scientific_sha256']}")
    print(f"selection_reused_without_change = {str(manifest['selection_reused_without_change']).lower()}")
    print(f"final_content_resolved = {str(manifest['final_content_resolved']).lower()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (adoption.B2DLCMV4RosterAdoptionError, protocol.B2DLCMV4ProtocolError) as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
