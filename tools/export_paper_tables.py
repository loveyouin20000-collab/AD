#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.evaluation.paper_tables import (  # noqa: E402
    PAPER_TABLE_IDS,
    export_paper_tables,
)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                text=True,
            )
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _abs(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export paper CSV/LaTeX tables and a release manifest"
    )
    p.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "artifacts" / "results",
        help="Directory of per-experiment summary.json trees",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSV/TeX/release_manifest.json",
    )
    p.add_argument("--seed", type=int, default=111)
    p.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help="Config path to record in the release manifest (repeatable)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Checkpoint path to record in the release manifest (repeatable)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results = _abs(args.results)
    output_dir = _abs(
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "artifacts" / "paper"
    )
    default_configs = [REPO_ROOT / "configs" / "rad" / "experiments.yaml"]
    configs = [_abs(p) for p in (args.config or default_configs)]
    checkpoints = [_abs(p) for p in (args.checkpoint or [])]
    sha = _git_sha()

    print("experiment_kind=paper_table_export")
    print(f"git_sha={sha}")
    print(f"seed={args.seed}")
    print(f"results={results}")
    print(f"output_dir={output_dir}")
    print(f"table_count={len(PAPER_TABLE_IDS)}")
    print(f"dry_run={str(bool(args.dry_run)).lower()}")
    if args.dry_run:
        for table_id in PAPER_TABLE_IDS:
            print(f"table={table_id}")
        print("tag_recommendation=cvpr-rad-visualad-v2")
        return 0

    try:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from validate_environment import collect_environment
    except Exception:  # pragma: no cover - fallback without torch
        environment = {"python": sys.version.split()[0], "torch": "unknown"}
    else:
        environment = collect_environment()

    artifacts = export_paper_tables(
        results_dir=results,
        output_dir=output_dir,
        git_sha=sha,
        seed=args.seed,
        config_paths=configs,
        checkpoint_paths=checkpoints,
        environment=environment,
    )
    for table_id, paths in artifacts.items():
        print(f"wrote {table_id}: csv={paths['csv']} tex={paths['tex']}")
    print(f"release_manifest={output_dir / 'release_manifest.json'}")
    print("tag_recommendation=cvpr-rad-visualad-v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
