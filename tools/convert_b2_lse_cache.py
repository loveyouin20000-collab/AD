from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.phase_b import b2_lse_prerequisites as prereq  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert B2 production cache into LSE split caches.")
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--calibration-cache", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = prereq.convert_b2_cache_for_lse(
            source_cache=args.source_cache,
            train_cache=args.train_cache,
            calibration_cache=args.calibration_cache,
            data_path=args.data_path,
        )
    except prereq.B2LSEPrerequisiteError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
