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

from rad.artifacts import (  # noqa: E402
    assert_json_artifact_eligible_for_evaluation,
    atomic_write_json,
    refuse_existing_run,
)
from rad.config import ExperimentConfig  # noqa: E402
from rad.data.adapters import build_preprocess, get_adapter  # noqa: E402
from rad.errors import OutputProtectionError, RADContractError  # noqa: E402
from rad.evaluation.dataset_evaluator import evaluate_dataset  # noqa: E402
from rad.evaluation.effective_config import (  # noqa: E402
    adaptive_config_identity,
    deep_merge_config,
)
from rad.evaluation.paper_metrics import compute_paper_metrics  # noqa: E402
from tools.smoke_adaptive_engine import build_engine, load_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-dataset adaptive evaluation (paper metrics path)"
    )
    p.add_argument("--config", type=str, default="configs/rad/adaptive.yaml")
    p.add_argument(
        "--overlay",
        type=str,
        default=None,
        help="Optional YAML overlay deep-merged into --config (matrix rows)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--profile", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force-full-depth", action="store_true")
    p.add_argument("--compute-full-depth-reference", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


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


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    overlay_path: Path | None = None
    if args.overlay:
        overlay_path = Path(args.overlay)
        if not overlay_path.is_absolute():
            overlay_path = REPO_ROOT / overlay_path
    config_identity = adaptive_config_identity(cfg_path, overlay_path=overlay_path)
    raw = yaml.safe_load(cfg_path.read_text())
    if overlay_path is not None:
        overlay = yaml.safe_load(overlay_path.read_text()) or {}
        if not isinstance(overlay, dict):
            raise SystemExit("--overlay must be a YAML mapping")
        raw = deep_merge_config(raw, overlay)
    cfg = ExperimentConfig.from_yaml(args.config)
    adaptive = dict(raw.get("adaptive", {}))
    data = dict(raw.get("data", {}))

    seed = args.seed if args.seed is not None else cfg.seed
    torch.manual_seed(seed)
    device = torch.device(args.device or raw.get("device", cfg.device))
    profile_name = args.profile or str(adaptive.get("profile_name", "balanced"))
    output_dir = args.output_dir or Path(
        adaptive.get("dataset_output_dir", "artifacts/eval/adaptive_dataset")
    )
    output_dir = _resolve(output_dir)

    dataset_name = args.dataset or str(data.get("dataset", "mvtec"))
    data_path = _resolve(args.data_path or data.get("data_path", ""))
    backbone_name = str(raw.get("teacher", {}).get("backbone", "ViT-L/14@336px"))
    image_size = int(raw.get("image_size", 518))
    limit = args.limit if args.limit is not None else adaptive.get("limit")
    force_full_depth = bool(args.force_full_depth)
    compute_ref = bool(
        args.compute_full_depth_reference
        or adaptive.get("compute_full_depth_reference", False)
    )

    config_hash_fields = config_identity.as_manifest_fields()
    sha = git_sha()
    profiles_path = _resolve(adaptive["policy_profiles"])
    lse_path = _resolve(adaptive["lse_checkpoint"])
    dlcm_path = _resolve(adaptive["dlcm_checkpoint"])

    print(f"config: {args.config}")
    if overlay_path is not None:
        print(f"overlay: {overlay_path.relative_to(REPO_ROOT)}")
    print(f"base_config_sha256: {config_identity.base_config_sha256}")
    if config_identity.overlay_sha256 is not None:
        print(f"overlay_sha256: {config_identity.overlay_sha256}")
    print(f"effective_config_sha256: {config_identity.effective_config_sha256}")
    print(f"config_sha256: {config_hash_fields['config_sha256']}")
    print(f"git_sha: {sha}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"dataset: {dataset_name}")
    print(f"data_path: {data_path}")
    print(f"adapter: {dataset_name}")
    print(f"backbone: {backbone_name}")
    print(f"profile: {profile_name}")
    print(f"output_dir: {output_dir}")
    print(f"limit: {limit}")
    print(f"force_full_depth: {force_full_depth}")
    print(f"compute_full_depth_reference: {compute_ref}")
    if raw.get("selector"):
        print(f"selector_signals: {json.dumps(raw['selector'].get('signals', {}))}")
    if raw.get("method"):
        print(f"method: {json.dumps(raw['method'])}")

    if args.dry_run:
        print("dry-run ok")
        return 0

    try:
        refuse_existing_run(output_dir)
    except OutputProtectionError as exc:
        raise SystemExit(str(exc)) from exc

    if not data_path.is_dir():
        raise SystemExit(f"missing data path: {data_path}")
    if not lse_path.is_file():
        raise SystemExit(f"missing LSE checkpoint: {lse_path}")
    if not dlcm_path.is_file():
        raise SystemExit(f"missing DLCM checkpoint: {dlcm_path}")

    assert_json_artifact_eligible_for_evaluation(
        profiles_path, kind="policy profiles"
    )
    assert_json_artifact_eligible_for_evaluation(
        _resolve(adaptive["descriptor_stats"]), kind="descriptor statistics"
    )

    profile = load_profile(profiles_path, profile_name)
    adapter = get_adapter(dataset_name, data_path)
    preprocess = build_preprocess(backbone_name, image_size)
    engine = build_engine(raw=raw, cfg=cfg, device=device, profile=profile)

    outputs = evaluate_dataset(
        adapter=adapter,
        engine=engine,
        preprocess=preprocess,
        device=device,
        split="test",
        limit=None if limit is None else int(limit),
        force_full_depth=force_full_depth,
        compute_full_depth_reference=compute_ref,
    )
    metrics = compute_paper_metrics(
        image_labels=outputs.image_labels,
        image_scores=outputs.image_scores,
        masks=outputs.masks,
        anomaly_maps=outputs.anomaly_maps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for pred in outputs.sample_predictions:
            handle.write(
                json.dumps(
                    {
                        "sample_id": pred.sample_id,
                        "dataset": pred.dataset,
                        "category": pred.category,
                        "image_label": pred.image_label,
                        "image_score": pred.image_score,
                        "selected_depth": pred.selected_depth,
                        "residual_gain": pred.residual_gain,
                    }
                )
                + "\n"
            )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "command": list(sys.argv),
        "config_path": str(args.config),
        **config_hash_fields,
        "overlay_path": str(overlay_path.relative_to(REPO_ROOT)) if overlay_path else None,
        "git_sha": sha,
        "seed": seed,
        "dataset": dataset_name,
        "adapter": dataset_name,
        "backbone": backbone_name,
        "candidate_layers": list(cfg.backbone.candidate_layers),
        "preprocess": {
            "image_size": preprocess.image_size,
            "mean": list(preprocess.mean),
            "std": list(preprocess.std),
        },
        "checkpoint_paths": {
            "lse": str(lse_path),
            "dlcm": str(dlcm_path),
            "policy_profiles": str(profiles_path),
        },
        "checkpoint_sha256": {
            "lse": sha256_file(lse_path),
            "dlcm": sha256_file(dlcm_path),
        },
        "n_samples": int(len(outputs.records)),
        "force_full_depth": force_full_depth,
        "compute_full_depth_reference": compute_ref,
        "metric_config": {
            "aupro_max_fpr": 0.3,
            "aupro_steps": 200,
            "boundary_tolerance_ratio": 0.005,
        },
        **engine.selector_provenance(),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(output_dir / "metrics.json", metrics.as_dict())
    print(json.dumps({"n": len(outputs.records), "metrics": metrics.as_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RADContractError as exc:
        raise SystemExit(f"RAD contract error: {exc}") from exc
