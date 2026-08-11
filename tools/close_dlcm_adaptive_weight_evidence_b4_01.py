#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.phase_b import b2_dlcm_training as training  # noqa: E402
from rad.phase_b import b2_dlcm_v5 as v5  # noqa: E402
from rad.phase_b import b2_dlcm_v5_official as official  # noqa: E402
from rad.phase_b import b4_dlcm_weight_evidence as evidence  # noqa: E402


DEFAULT_DESCRIPTOR_RUN = Path(
    "/root/autodl-tmp/AD-phase-b2-descriptor-real-extraction/"
    "artifacts/phase_b/b2_descriptor_artifacts/authoritative-run-a-20260729-013956"
)
DEFAULT_TARGET_RUN = Path(
    "/root/autodl-tmp/AD-phase-b2-contribution-target-materialization/"
    "artifacts/phase_b/b2_contribution_targets/authoritative-run-a-20260804-030431"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Close B4-01 DLCM adaptive weight evidence")
    p.add_argument("--config", type=Path, default=Path("configs/phase_b/b2_dlcm_uniform_anchored_official_v5.json"))
    p.add_argument(
        "--accepted-reference-evidence",
        type=Path,
        default=Path("docs/phase_b/b2_06b_accepted_v5_reference_packaging_evidence.json"),
    )
    p.add_argument("--descriptor-manifest", type=Path, default=DEFAULT_DESCRIPTOR_RUN / "final_manifest.json")
    p.add_argument("--descriptor-root", type=Path, default=DEFAULT_DESCRIPTOR_RUN)
    p.add_argument("--contribution-target-manifest", type=Path, default=DEFAULT_TARGET_RUN / "final_manifest.json")
    p.add_argument("--contribution-target-root", type=Path, default=DEFAULT_TARGET_RUN)
    p.add_argument("--output-dir", type=Path, default=Path("docs/phase_b"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _tracked_pt_count() -> int:
    out = subprocess.check_output(["git", "ls-files", "*.pt"], cwd=REPO_ROOT).decode()
    return len([line for line in out.splitlines() if line.strip()])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".sha256").write_text(
        evidence.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        evidence.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stable_sample_id",
                "category",
                "split",
                "depth",
                "player_layer_ids",
                "dynamic_weights",
                "deployment_weights",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "player_layer_ids": json.dumps(row["player_layer_ids"], separators=(",", ":")),
                    "dynamic_weights": json.dumps(row["dynamic_weights"], separators=(",", ":")),
                    "deployment_weights": json.dumps(row["deployment_weights"], separators=(",", ":")),
                }
            )
    path.with_suffix(path.suffix + ".sha256").write_text(
        evidence.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def _checkpoint_sha256(path: Path) -> str:
    return evidence.sha256_file(path)


def _materialize_rows(args: argparse.Namespace, accepted: dict[str, Any]) -> list[dict[str, Any]]:
    config = evidence.load_json(_resolve(args.config))
    verified = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=args.descriptor_manifest,
        descriptor_root=args.descriptor_root,
        contribution_target_manifest=args.contribution_target_manifest,
        contribution_target_root=args.contribution_target_root,
        accepted_upstream=dict(config["accepted_upstream"]),
        evaluation_unlocked=False,
    )
    ckpt_path = Path(str(accepted["packaged_checkpoint"]))
    expected_sha = str(accepted["checkpoint_sha256"])
    got_sha = _checkpoint_sha256(ckpt_path)
    if got_sha != expected_sha:
        raise evidence.B4DLCMWeightEvidenceError(
            "B4_DLCM_WEIGHT_EVIDENCE_CHECKPOINT_SHA_MISMATCH",
            f"{got_sha} != {expected_sha}",
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if str(ckpt.get("H_deploy")) != evidence.EXPECTED_V5_DEPLOYMENT_IDENTITY:
        raise evidence.B4DLCMWeightEvidenceError(
            "B4_DLCM_WEIGHT_EVIDENCE_IDENTITY_MISMATCH",
            "checkpoint H_deploy does not match accepted V5 deployment identity",
        )
    beta = float(ckpt["beta"])
    trunk = official.load_c3_deployment_trunk(ckpt)
    rows: list[dict[str, Any]] = []
    trunk.eval()
    with torch.no_grad():
        for record in verified.calibration_records:
            depth = 24
            desc = record["descriptors"][depth]
            if desc.ndim == 3:
                desc = desc.reshape(desc.shape[-2], desc.shape[-1])
            x = desc.unsqueeze(0).to(dtype=torch.float32)
            _logits, dynamic_weights = trunk.forward(x, prediction_depth=depth)
            dynamic = dynamic_weights.reshape(-1).detach().cpu().to(dtype=torch.float32)
            deployment = v5.mix_uniform_anchored_weights(dynamic, beta).detach().cpu()
            rows.append(
                {
                    "stable_sample_id": str(record["stable_sample_id"]),
                    "category": str(record["category"]),
                    "split": "calibration",
                    "depth": depth,
                    "player_layer_ids": list(trunk.players_for_depth(depth)),
                    "dynamic_weights": [float(v) for v in dynamic.tolist()],
                    "deployment_weights": [float(v) for v in deployment.reshape(-1).tolist()],
                }
            )
    return rows


def _summary_markdown(payload: dict[str, Any]) -> str:
    dep = payload["deployment_weight_summary"]
    dyn = payload["dynamic_weight_summary"]
    return f"""# B4-01 DLCM Adaptive Weight Evidence

Status: {payload["status"]}

## Result

Accepted V5 deployment weights were exported on the Calibration split without
training, Final access, or model artifact generation.

```text
sample_adaptive_variation_observed = {str(payload["sample_adaptive_variation_observed"]).lower()}
uniform_equivalent_at_tolerance = {str(payload["uniform_equivalent_at_tolerance"]).lower()}
calibration_records = {payload["calibration_records"]}
beta_star_decimal = {payload["beta_star_decimal"]}
```

## Deployment Weight Statistics

```text
layer_means = {dep["layer_means"]}
layer_stds = {dep["layer_stds"]}
max_layer_std = {dep["max_layer_std"]}
mean_sample_linf_delta_from_uniform = {dep["mean_sample_linf_delta_from_uniform"]}
max_sample_linf_delta_from_uniform = {dep["max_sample_linf_delta_from_uniform"]}
rows_non_uniform_at_tolerance = {dep["rows_non_uniform_at_tolerance"]}
```

## Dynamic Head Statistics

```text
layer_means = {dyn["layer_means"]}
layer_stds = {dyn["layer_stds"]}
max_layer_std = {dyn["max_layer_std"]}
mean_sample_linf_delta_from_uniform = {dyn["mean_sample_linf_delta_from_uniform"]}
max_sample_linf_delta_from_uniform = {dyn["max_sample_linf_delta_from_uniform"]}
rows_non_uniform_at_tolerance = {dyn["rows_non_uniform_at_tolerance"]}
```

## Paper Interpretation

The accepted V5 deployment is not merely a fixed equal-weight fusion at the
selected tolerance. Its deployment weights retain sample-level variation after
the uniform anchor beta is applied. This supports the paper claim that DLCM is
a sample-adaptive layer fusion mechanism, while keeping the claim limited to
the accepted Calibration-split evidence.

## Boundary

```text
training_started = false
evaluation_started = false
final_content_accessed = false
model_artifact_generated = false
tracked_pt_files = {payload["boundary"]["tracked_pt_files"]}
```

Weight evidence identity:

```text
{payload["weight_evidence_identity"]}
```
"""


def main() -> int:
    args = parse_args()
    accepted = evidence.load_json(_resolve(args.accepted_reference_evidence))
    rows = _materialize_rows(args, accepted)
    payload = evidence.build_weight_evidence_manifest(
        rows=rows,
        accepted_reference_evidence=accepted,
        tracked_pt_count=_tracked_pt_count(),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    output_dir = _resolve(args.output_dir)
    _write_json(output_dir / "b4_01_dlcm_adaptive_weight_evidence_manifest.json", payload)
    _write_text(output_dir / "b4_01_dlcm_adaptive_weight_evidence.md", _summary_markdown(payload))
    _write_csv(output_dir / "b4_01_dlcm_adaptive_weight_trace.csv", rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
