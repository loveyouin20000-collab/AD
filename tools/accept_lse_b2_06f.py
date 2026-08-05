from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.phase_b import b2_lse_accepted_closure as closure  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Close accepted B2 LSE artifact")
    p.add_argument(
        "--decision",
        type=Path,
        default=Path("docs/phase_b/b2_06e_lse_qualification_decision_manifest.json"),
    )
    p.add_argument(
        "--training-receipt",
        type=Path,
        default=Path(
            "artifacts/checkpoints/lse/b2_06d_first_controlled_run/"
            "b2_06d_lse_training_receipt.json"
        ),
    )
    p.add_argument(
        "--lse-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/lse/b2_06d_first_controlled_run/lse_best.pt"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase_b/b2_06f_accepted_lse_artifact"),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path.with_suffix(path.suffix + ".sha256")).write_text(
        closure.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    decision_path = _resolve(args.decision)
    receipt_path = _resolve(args.training_receipt)
    source_checkpoint = _resolve(args.lse_checkpoint)
    output_dir = _resolve(args.output_dir)
    accepted_refs = output_dir / "accepted_refs"
    accepted_checkpoint = accepted_refs / "lse_best.pt"

    decision = closure.load_json(decision_path)
    receipt = closure.load_json(receipt_path)
    checkpoint_sha = closure.sha256_file(source_checkpoint)
    manifest = closure.build_accepted_lse_manifest(
        decision=decision,
        training_receipt=receipt,
        lse_checkpoint_sha256=checkpoint_sha,
        accepted_checkpoint_path=str(accepted_checkpoint),
        source_checkpoint_path=str(source_checkpoint),
        closure_git_sha=_git_sha(),
    )
    dry_run_report = {
        "schema_version": "b2_06f_lse_accepted_closure_dry_run_v1",
        "ready": True,
        "accepted_artifact_generated": False,
        "training_started": False,
        "evaluation_started": False,
        "accepted_lse_identity": manifest["accepted_lse_identity"],
        "lse_checkpoint_sha256": checkpoint_sha,
        "output_dir": str(output_dir),
    }
    if args.dry_run:
        print(json.dumps(dry_run_report, indent=2, sort_keys=True))
        return 0

    if (output_dir / "accepted_lse_manifest.json").exists():
        raise SystemExit("B2_LSE_ACCEPTED_CLOSURE_ALREADY_EXISTS")
    accepted_refs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, accepted_checkpoint)
    copied_sha = closure.sha256_file(accepted_checkpoint)
    if copied_sha != checkpoint_sha:
        raise SystemExit("B2_LSE_ACCEPTED_CLOSURE_COPY_SHA_MISMATCH")
    manifest["accepted_lse_checkpoint_sha256"] = copied_sha
    manifest["accepted_lse_identity"] = closure.canonical_json_sha256(
        {k: v for k, v in manifest.items() if k != "accepted_lse_identity"}
    )
    receipt_payload = {
        "schema_version": "b2_06f_lse_accepted_artifact_closure_receipt_v1",
        "accepted_lse_identity": manifest["accepted_lse_identity"],
        "accepted_lse_checkpoint": str(accepted_checkpoint),
        "accepted_lse_checkpoint_sha256": copied_sha,
        "accepted_manifest": str(output_dir / "accepted_lse_manifest.json"),
        "training_started": False,
        "evaluation_started": False,
        "accepted_artifact_generated": True,
    }
    receipt_payload["receipt_identity"] = closure.canonical_json_sha256(receipt_payload)
    _write_json(output_dir / "accepted_lse_manifest.json", manifest)
    _write_json(output_dir / "accepted_lse_closure_receipt.json", receipt_payload)
    print(json.dumps({"manifest": manifest, "receipt": receipt_payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
