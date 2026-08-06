#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.phase_b import b3_early_exit_prerequisites as prereq  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize B3 early-exit prerequisites")
    p.add_argument("--config", type=Path, default=Path("configs/rad/early_exit_b3_accepted_lse.yaml"))
    p.add_argument(
        "--calibration-predictions",
        type=Path,
        default=Path(
            "/root/autodl-tmp/AD-phase-b2-06d-lse-training-unlock/"
            "artifacts/checkpoints/lse/b2_06d_first_controlled_run/cal_predictions.jsonl"
        ),
    )
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("artifacts/targets/early_exit/b3_02_materialization_manifest.json"),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _write_json_with_sha(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        prereq.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.dry_run:
        # Validate inputs without writing the three prerequisite files.
        cfg = args.config if args.config.is_absolute() else REPO_ROOT / args.config
        if not args.calibration_predictions.is_file():
            raise SystemExit("B3_EXIT_PREREQ_CALIBRATION_PREDICTIONS_REQUIRED")
        print(json.dumps({"ready_to_materialize": True, "training_started": False, "evaluation_started": False}, indent=2))
        return 0
    manifest = prereq.materialize_exit_prerequisites(
        config_path=args.config,
        calibration_predictions_path=args.calibration_predictions,
        repo_root=REPO_ROOT,
    )
    manifest_out = args.manifest_out if args.manifest_out.is_absolute() else REPO_ROOT / args.manifest_out
    _write_json_with_sha(manifest_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
