from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.artifacts import atomic_write_json, refuse_existing_run  # noqa: E402
from rad.config import ExperimentConfig  # noqa: E402
from rad.data.adapters import build_preprocess, get_adapter  # noqa: E402
from rad.errors import (  # noqa: E402
    ARTIFACT_INTEGRITY_EXIT_CODE,
    ArtifactIntegrityError,
    OutputProtectionError,
    RADContractError,
)
from rad.evaluation.dataset_evaluator import evaluate_dataset  # noqa: E402
from rad.evaluation.export import (  # noqa: E402
    TransferSamplePrediction,
    export_transfer_predictions,
)
from rad.evaluation.paper_metrics import compute_paper_metrics  # noqa: E402
from rad.evaluation.zero_shot import (  # noqa: E402
    TargetAccessError,
    assert_policy_eligible_for_evaluation,
    assert_policy_unchanged,
    boundary_complexity,
    forbid_target_access_during_calibration,
    load_frozen_policy_profile,
)
from tools.smoke_adaptive_engine import build_engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot transfer eval with frozen source-calibrated policy"
    )
    p.add_argument("--config", type=str, default="configs/rad/zero_shot_transfer.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--profile", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--target-dataset", type=str, default=None)
    p.add_argument("--target-data-path", type=Path, default=None)
    p.add_argument(
        "--calibration-policy",
        type=Path,
        default=None,
        help="Override frozen source-calibrated policy profiles JSON",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _resolve_calibration_policy_path(
    args: argparse.Namespace,
    transfer: dict[str, Any],
    adaptive: dict[str, Any],
) -> Path:
    """Precedence: --calibration-policy → configured policy path → hard failure."""
    if args.calibration_policy is not None:
        return _resolve(args.calibration_policy)
    configured = transfer.get("calibration_policy") or adaptive.get("policy_profiles")
    if not configured:
        raise ArtifactIntegrityError("calibration policy path not configured")
    return _resolve(configured)


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = ExperimentConfig.from_yaml(args.config)
    if cfg.zero_shot.target_tuning:
        raise SystemExit("target_tuning must be false")

    transfer = dict(raw.get("transfer", {}))
    adaptive = dict(raw.get("adaptive", {}))
    seed = args.seed if args.seed is not None else cfg.seed
    torch.manual_seed(seed)
    device = torch.device(args.device or raw.get("device", cfg.device))
    profile_name = args.profile or str(
        transfer.get("policy_profile", adaptive.get("profile_name", "balanced"))
    )
    output_dir = args.output_dir or Path(
        transfer.get("output_dir", "artifacts/eval/zero_shot_transfer")
    )
    output_dir = _resolve(output_dir)
    limit = args.limit if args.limit is not None else transfer.get("limit")
    epsilon = float(transfer.get("epsilon_gain", 0.05))
    full_depth = int(adaptive.get("full_depth", cfg.backbone.depth))

    policy_path = _resolve_calibration_policy_path(args, transfer, adaptive)
    target_dataset = str(
        args.target_dataset
        or transfer.get("target_dataset")
        or (cfg.zero_shot.target_datasets[0] if cfg.zero_shot.target_datasets else "visa")
    )
    configured_target_path = transfer.get("target_data_path")
    if args.target_data_path is not None:
        target_path = _resolve(args.target_data_path)
    elif configured_target_path:
        target_path = _resolve(configured_target_path)
    else:
        raise ArtifactIntegrityError(
            "target data path is required via transfer.target_data_path or --target-data-path"
        )
    backbone_name = str(raw.get("teacher", {}).get("backbone", "ViT-L/14@336px"))
    image_size = int(raw.get("image_size", 518))

    config_hash = sha256_file(Path(args.config))
    sha = git_sha()
    print(f"config: {args.config}")
    print(f"config_hash: {config_hash}")
    print(f"git_sha: {sha}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"source_dataset: {cfg.zero_shot.source_dataset}")
    print(f"target_datasets: {cfg.zero_shot.target_datasets}")
    print(f"target_dataset: {target_dataset}")
    print(f"target_tuning: {cfg.zero_shot.target_tuning}")
    print(f"policy_path: {policy_path}")
    print(f"profile: {profile_name}")
    print(f"target_data_path: {target_path}")
    print(f"adapter: {target_dataset}")
    print(f"output_dir: {output_dir}")

    # Source-only calibration gate: refuse touching target during policy load.
    with forbid_target_access_during_calibration(
        source_dataset=cfg.zero_shot.source_dataset,
        target_datasets=cfg.zero_shot.target_datasets,
    ) as guard:
        guard.check_path(policy_path)
        profile, policy_digest = load_frozen_policy_profile(policy_path, profile_name)
        assert_policy_unchanged(policy_path, profile_name, policy_digest)

    if args.dry_run:
        print(f"policy_digest: {policy_digest}")
        print("dry-run ok")
        return 0

    assert_policy_eligible_for_evaluation(policy_path)

    try:
        refuse_existing_run(output_dir)
    except OutputProtectionError as exc:
        raise SystemExit(str(exc)) from exc

    if not target_path.is_dir():
        raise SystemExit(f"missing target data path: {target_path}")

    engine = build_engine(raw=raw, cfg=cfg, device=device, profile=profile)
    assert_policy_unchanged(policy_path, profile_name, policy_digest)

    adapter = get_adapter(target_dataset, target_path)
    preprocess = build_preprocess(backbone_name, image_size)
    outputs = evaluate_dataset(
        adapter=adapter,
        engine=engine,
        preprocess=preprocess,
        device=device,
        split="test",
        limit=None if limit is None else int(limit),
        force_full_depth=False,
        compute_full_depth_reference=True,
    )
    assert_policy_unchanged(policy_path, profile_name, policy_digest)

    paper = compute_paper_metrics(
        image_labels=outputs.image_labels,
        image_scores=outputs.image_scores,
        masks=outputs.masks,
        anomaly_maps=outputs.anomaly_maps,
    )

    rows: list[TransferSamplePrediction] = []
    for i, pred in enumerate(outputs.sample_predictions):
        mask = outputs.masks[i]
        rows.append(
            TransferSamplePrediction(
                sample_id=pred.sample_id,
                dataset=pred.dataset,
                selected_depth=int(pred.selected_depth),
                image_label=int(pred.image_label),
                residual_gain=float(
                    0.0 if pred.residual_gain is None else pred.residual_gain
                ),
                anomaly_area=float(mask.mean()),
                contrast_proxy=float(mask.std()),
                boundary_complexity=float(boundary_complexity(mask)),
            )
        )

    summary = export_transfer_predictions(
        rows,
        output_dir=output_dir,
        full_depth=full_depth,
        epsilon=epsilon,
        adaptive_maps=outputs.anomaly_maps,
        full_depth_maps=None,
        masks=outputs.masks,
        images=None,
        paper_metrics=paper,
    )
    # When full-depth maps are not separately stored, export uses paper metrics
    # plus policy aggregates from residual gains / depths.
    meta: dict[str, Any] = {
        "config_hash": config_hash,
        "git_sha": sha,
        "seed": seed,
        "policy_path": str(policy_path),
        "policy_profile": profile_name,
        "policy_digest": policy_digest,
        "target_tuning": False,
        "target_dataset": target_dataset,
        "target_data_path": str(target_path),
        "adapter": target_dataset,
        "n_samples": len(rows),
        "paper_metrics": paper.as_dict(),
        "summary": summary,
    }
    atomic_write_json(output_dir / "run_meta.json", meta)
    atomic_write_json(output_dir / "metrics.json", paper.as_dict())
    print(
        json.dumps(
            {"n": len(rows), "paper_metrics": paper.as_dict(), "summary_keys": list(summary.keys())},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TargetAccessError as exc:
        raise SystemExit(f"target access violation: {exc}") from exc
    except ArtifactIntegrityError as exc:
        print(f"artifact integrity error: {exc}", file=sys.stderr)
        raise SystemExit(ARTIFACT_INTEGRITY_EXIT_CODE) from exc
    except RADContractError as exc:
        raise SystemExit(f"RAD contract error: {exc}") from exc
