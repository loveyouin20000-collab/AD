from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.config import ExperimentConfig  # noqa: E402
from rad.models.descriptors import DescriptorNormalizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit descriptor median/IQR stats on source-train teacher cache only",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cache", type=Path, required=True, help="Teacher cache directory")
    parser.add_argument("--output", type=Path, required=True, help="Stats JSON path")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config is not None:
        cfg = ExperimentConfig.from_yaml(args.config)
        seed = args.seed if args.seed is not None else cfg.seed
    else:
        seed = args.seed if args.seed is not None else 111

    output = Path(args.output)
    if args.output_dir is not None:
        output = Path(args.output_dir) / output.name

    print(f"cache: {args.cache}")
    print(f"output: {output}")
    print(f"seed: {seed}")
    print(f"max_samples: {args.max_samples}")

    if args.dry_run:
        return 0

    if output.exists() and not args.force:
        raise SystemExit(f"output exists: {output} (pass --force to overwrite)")

    normalizer = DescriptorNormalizer(clamp=(-8.0, 8.0))
    normalizer.fit_from_cache(args.cache, max_samples=args.max_samples)
    normalizer.save(output)
    assert normalizer.median is not None and normalizer.iqr is not None
    print(f"wrote: {output}")
    print(f"feature_dim: {int(normalizer.median.numel())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
