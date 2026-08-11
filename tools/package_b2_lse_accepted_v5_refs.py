from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.phase_b import b2_lse_accepted_refs as refs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package accepted V5 checkpoint references for B2 LSE.")
    parser.add_argument("--accepted-manifest", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-calibration-manifest", required=True, type=Path)
    parser.add_argument("--expected-accepted-identity", required=True)
    parser.add_argument("--expected-v5-deployment-identity", required=True)
    parser.add_argument("--expected-calibration-ab-identity", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = refs.package_accepted_checkpoint_reference(
            accepted_manifest=args.accepted_manifest,
            source_checkpoint=args.source_checkpoint,
            source_calibration_manifest=args.source_calibration_manifest,
            expected_accepted_identity=args.expected_accepted_identity,
            expected_v5_deployment_identity=args.expected_v5_deployment_identity,
            expected_calibration_ab_identity=args.expected_calibration_ab_identity,
        )
    except refs.B2AcceptedRefsError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
