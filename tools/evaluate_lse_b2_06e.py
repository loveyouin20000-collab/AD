from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.train_lse as train_lse  # noqa: E402
from rad.config import ExperimentConfig  # noqa: E402
from rad.data.cache_dataset import TeacherCacheDataset  # noqa: E402
from rad.models.descriptors import (  # noqa: E402
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.lse import LSE  # noqa: E402
from rad.models.selector_signals import (  # noqa: E402
    apply_selector_signal_mask,
    build_default_selector_signal_layout,
    parse_enabled_signals,
    selector_signal_provenance,
)
from rad.phase_b import b2_lse_accepted_gate as accepted_gate  # noqa: E402
from rad.phase_b import b2_lse_prerequisites as prereq  # noqa: E402
from rad.phase_b import b2_lse_qualification as qual  # noqa: E402
from rad.trainers.lse_trainer import LSETrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and qualify B2-06D LSE checkpoint")
    p.add_argument("--config", type=Path, default=Path("configs/rad/lse_b2_accepted_v5.yaml"))
    p.add_argument(
        "--lse-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/lse/b2_06d_first_controlled_run/lse_best.pt"),
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase_b/b2_06e_lse_evaluation_qualification"),
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-calibration-nll", type=float, default=0.5)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path.with_suffix(path.suffix + ".sha256")).write_text(
        qual.sha256_file(path) + "  " + path.name + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig.from_yaml(str(config_path))
    lse_cfg = dict(raw.get("lse", {}))
    fusion_cfg = raw.get("fusion", {})
    preflight_cfg = accepted_gate.load_lse_preflight_config(config_path, repo_root=REPO_ROOT)
    preflight = accepted_gate.run_lse_preflight(preflight_cfg)
    if preflight["ready"] is not True:
        raise SystemExit("B2_LSE_QUALIFICATION_PREFLIGHT_NOT_READY")

    lse_checkpoint = _resolve(args.lse_checkpoint)
    training_receipt_path = _resolve(args.training_receipt)
    output_dir = _resolve(args.output_dir)
    config_hash = qual.sha256_file(config_path)
    checkpoint_sha = qual.sha256_file(lse_checkpoint)
    receipt = qual.load_json(training_receipt_path)
    if checkpoint_sha != receipt.get("best_checkpoint_sha256"):
        raise SystemExit("B2_LSE_QUALIFICATION_CHECKPOINT_SHA_MISMATCH")

    expected = {
        "accepted_identity": preflight["accepted_identity"],
        "v5_deployment_identity": preflight["v5_deployment_identity"],
        "H_decision": preflight["H_decision"],
        "H_evidence": preflight["H_evidence"],
        "config_hash": config_hash,
        "best_checkpoint_sha256": checkpoint_sha,
    }
    early_depths = tuple(int(x) for x in lse_cfg.get("early_depths", [12, 18]))
    if args.dry_run:
        dry_run_report = {
            "schema_version": "b2_06e_lse_qualification_dry_run_v1",
            "ready": True,
            "evaluation_started": False,
            "accepted_artifact_generated": False,
            "lse_checkpoint": str(lse_checkpoint),
            "lse_checkpoint_sha256": checkpoint_sha,
            "training_receipt": str(training_receipt_path),
            "config_sha256": config_hash,
            "required_depths": list(early_depths),
            "max_calibration_nll": float(args.max_calibration_nll),
        }
        print(json.dumps(dry_run_report, indent=2, sort_keys=True))
        return 0

    device = torch.device(args.device or raw.get("device", cfg.device))
    seed = int(receipt.get("seed", cfg.seed))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    candidate_layers = tuple(cfg.backbone.candidate_layers)
    cal_cache_dir = Path(lse_cfg.get("calibration_cache", fusion_cfg.get("calibration_cache")))
    cal_gains_path = Path(lse_cfg.get("calibration_gain_targets"))
    stats_path = Path(lse_cfg.get("descriptor_stats", fusion_cfg.get("descriptor_stats", "")))
    dlcm_ckpt_path = Path(lse_cfg.get("dlcm_checkpoint"))
    cal_cache_dir = _resolve(cal_cache_dir)
    cal_gains_path = _resolve(cal_gains_path)
    stats_path = _resolve(stats_path)
    dlcm_ckpt_path = _resolve(dlcm_ckpt_path)

    cal_cache = TeacherCacheDataset(cal_cache_dir)
    cal_gains = train_lse.load_gain_index(cal_gains_path)
    normalizer = DescriptorNormalizer.load(stats_path) if stats_path.is_file() else None
    dlcm_ckpt = torch.load(dlcm_ckpt_path, map_location="cpu")
    dlcm = prereq.load_lse_dlcm_adapter_from_checkpoint(
        dlcm_ckpt,
        device=device,
        candidate_layers=candidate_layers,
    )
    selector_cfg = dict(raw.get("selector", {}))
    selector_layout = build_default_selector_signal_layout()
    enabled_signals = parse_enabled_signals(selector_cfg.get("signals"))
    selector_prov = selector_signal_provenance(
        enabled_signals=enabled_signals,
        layout=selector_layout,
        mask_applied=True,
    )
    layer_extractor = LayerDescriptorExtractor()
    context_extractor = CheckpointContextExtractor(backbone_depth=cfg.backbone.depth)
    cal_rows = train_lse.materialize_rows(
        cache=cal_cache,
        gains=cal_gains,
        dlcm=dlcm,
        layer_extractor=layer_extractor,
        context_extractor=context_extractor,
        normalizer=normalizer,
        candidate_layers=candidate_layers,
        early_depths=early_depths,
        device=device,
        limit=None,
        enabled_signals=enabled_signals,
        selector_layout=selector_layout,
    )
    ckpt = torch.load(lse_checkpoint, map_location=device)
    lse_model = LSE(
        state_dim=int(ckpt["state_dim"]),
        early_depths=tuple(int(x) for x in ckpt["early_depths"]),
    )
    lse_model.load_state_dict(ckpt["lse"], strict=True)
    lse_model.to(device)
    trainer = LSETrainer(
        model=lse_model,
        early_depths=early_depths,
        epsilon_gain=float(lse_cfg.get("epsilon_gain", 0.05)),
        sufficiency_weight=float(lse_cfg.get("sufficiency_weight", 0.5)),
    ).to(device)
    cal_loader = DataLoader(
        train_lse.LSEFeatureDataset(cal_rows),
        batch_size=int(lse_cfg.get("batch_size", 32)),
        shuffle=False,
        collate_fn=train_lse.collate_lse,
        num_workers=0,
    )
    cal_batches = [train_lse.move_batch(batch, device) for batch in cal_loader]
    metrics = trainer.evaluate(cal_batches)
    decision = qual.qualify_lse_evaluation(
        receipt=receipt,
        metrics=metrics,
        expected=expected,
        max_calibration_nll=float(args.max_calibration_nll),
        required_depths=early_depths,
    )
    report = {
        "schema_version": "b2_06e_lse_evaluation_report_v1",
        "evaluation_started": True,
        "accepted_artifact_generated": False,
        "lse_checkpoint": str(lse_checkpoint),
        "lse_checkpoint_sha256": checkpoint_sha,
        "training_receipt": str(training_receipt_path),
        "training_receipt_sha256": qual.sha256_file(training_receipt_path),
        "config_sha256": config_hash,
        "calibration_rows": len(cal_rows),
        "selector_signal_layout_hash": selector_prov["selector_signal_layout_hash"],
        "metrics": metrics,
        "decision": decision,
    }
    _write_json(output_dir / "lse_evaluation_report.json", report)
    _write_json(output_dir / "lse_qualification_decision_manifest.json", decision)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
