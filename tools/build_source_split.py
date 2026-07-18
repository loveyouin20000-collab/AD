from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.data.split import (  # noqa: E402
    build_source_split,
    load_samples_from_meta,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe source train/calibration split",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. mvtec)")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing meta.json",
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL manifest path (default: artifacts/splits/<dataset>_seed<seed>.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for default manifest name",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Reserved for future config wiring",
    )
    return parser.parse_args()


def resolve_output(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = REPO_ROOT / "artifacts" / "splits"
    return out_dir / f"{args.dataset}_seed{args.seed}.jsonl"


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    samples = load_samples_from_meta(root, mode="train")
    rows = build_source_split(
        samples,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    output = resolve_output(args)
    split_counts = Counter(r["split"] for r in rows)
    stratum_counts = Counter((r["category"], r["label"], r["split"]) for r in rows)
    preview = {str(k): v for k, v in list(stratum_counts.items())[:8]}

    print(f"dataset: {args.dataset}")
    print(f"root: {root}")
    print(f"seed: {args.seed}")
    print(f"calibration_fraction: {args.calibration_fraction}")
    print(f"samples: {len(samples)}")
    print(f"split_counts: {dict(split_counts)}")
    print(f"output: {output}")
    print(f"stratum_preview: {json.dumps(preview)}")

    if args.dry_run:
        return 0

    write_manifest(output, rows, force=args.force)
    print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
