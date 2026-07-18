#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sized
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import train_fusion as fusion_helpers  # noqa: E402

from rad.checkpoints.manifest_v1 import (  # noqa: E402
    SCHEMA_VERSION,
    CheckpointManifestV1,
    sha256_file,
    write_checkpoint_with_manifest,
)
from rad.config import ExperimentConfig  # noqa: E402
from rad.data.cache_dataset import TeacherCacheDataset  # noqa: E402
from rad.evaluation.zero_shot import pixel_average_precision, pro_score_proxy  # noqa: E402
from rad.losses.localization import sample_localization_error  # noqa: E402
from rad.models.descriptors import (  # noqa: E402
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM, sum_preserving_fusion  # noqa: E402
from rad.models.lse import LSE  # noqa: E402
from rad.trainers.fusion_trainer import FusionLossWeights, compute_fusion_objective  # noqa: E402
from rad.trainers.joint_trainer import (  # noqa: E402
    JointTrainer,
    NoRegressionThresholds,
    StagedCheckpointInfo,
    evaluate_no_regression,
    soft_expected_depth_ratio,
    validate_staged_checkpoint_pair,
)
from rad.trainers.lse_trainer import compute_lse_objective  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint DLCM+LSE fine-tuning ablation (non-primary pipeline)"
    )
    p.add_argument("--config", type=str, default="configs/rad/joint.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--fusion-checkpoint", type=Path, required=True)
    p.add_argument("--lse-checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-joint", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def git_sha() -> str:
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def config_hash(cfg: ExperimentConfig) -> str:
    payload = {
        "seed": cfg.seed,
        "backbone": {
            "depth": cfg.backbone.depth,
            "candidate_layers": list(cfg.backbone.candidate_layers),
        },
        "training": (
            {
                "mode": cfg.training.mode,
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "learning_rate": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
            }
            if cfg.training
            else None
        ),
        "joint": (
            {
                "enabled": cfg.joint.enabled,
                "primary_pipeline": cfg.joint.primary_pipeline,
                "trainable_modules": list(cfg.joint.trainable_modules),
                "fusion_loss_weight": cfg.joint.fusion_loss_weight,
                "lse_loss_weight": cfg.joint.lse_loss_weight,
                "compute_final_weight": cfg.joint.compute_final_weight,
                "compute_ramp_fraction": cfg.joint.compute_ramp_fraction,
                "compute_target_depth_ratio": cfg.joint.compute_target_depth_ratio,
            }
            if cfg.joint
            else None
        ),
    }
    return sha256_text(yaml.safe_dump(payload, sort_keys=True))


def _abs(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_launch_allowed(args: argparse.Namespace, cfg: ExperimentConfig) -> None:
    if not args.allow_joint:
        raise SystemExit(
            "Refusing joint training without explicit --allow-joint opt-in flag"
        )
    if cfg.joint is None or not cfg.joint.enabled:
        raise SystemExit("joint.enabled must be true in config")
    if cfg.joint.primary_pipeline:
        raise SystemExit("joint.primary_pipeline must be false for this ablation")
    if cfg.training is None or cfg.training.mode != "joint":
        raise SystemExit("training.mode must be joint")


def ensure_output_dir_available(output_dir: Path) -> None:
    for name in ("manifest.json", "summary.json"):
        path = output_dir / name
        if path.is_file():
            raise SystemExit(f"output directory already contains {name}: {path}")


def print_dry_run_plan(
    *,
    cfg: ExperimentConfig,
    staged: StagedCheckpointInfo,
    output_dir: Path,
    seed: int,
    smoke_test: bool,
) -> None:
    assert cfg.training is not None
    assert cfg.joint is not None
    print("experiment_kind=training_ablation")
    print("primary_pipeline=false")
    print(f"config_hash={config_hash(cfg)}")
    print(f"git_sha={git_sha()}")
    print(f"seed={seed}")
    print(f"split_manifest_hash={staged.split_manifest_hash}")
    print(f"fusion_checkpoint_sha256={staged.fusion_sha256}")
    print(f"lse_checkpoint_sha256={staged.lse_sha256}")
    print(f"trainable_modules={list(cfg.joint.trainable_modules)}")
    lr_text = f"{cfg.training.learning_rate:.1e}".replace("e-0", "e-")
    print(f"learning_rate={lr_text}")
    print(f"compute_ramp_fraction={cfg.joint.compute_ramp_fraction}")
    print(f"compute_final_weight={cfg.joint.compute_final_weight}")
    nr = cfg.joint.no_regression
    print(
        "no_regression="
        + json.dumps(
            {
                "max_pixel_ap_drop": nr.max_pixel_ap_drop,
                "max_pro_drop": nr.max_pro_drop,
                "max_mean_error_relative_increase": nr.max_mean_error_relative_increase,
            }
        )
    )
    print(f"output_dir={output_dir}")
    print(f"smoke_test={smoke_test}")
    if smoke_test:
        print("eligible_for_evaluation=false")
        print("train_samples=16")
        print("calibration_samples=8")
        print("epochs=1")


def assert_trainable_only_dlcm_lse(trainer: JointTrainer) -> None:
    names = trainer.trainable_parameter_names()
    bad = [n for n in names if not (n.startswith("dlcm.") or n.startswith("lse."))]
    if bad:
        raise RuntimeError(f"unexpected trainable parameters: {bad}")


def load_joint_trainer(
    *,
    cfg: ExperimentConfig,
    staged: StagedCheckpointInfo,
    fusion_checkpoint: Path,
    lse_checkpoint: Path,
    device: torch.device,
) -> JointTrainer:
    assert cfg.joint is not None
    assert cfg.data is not None

    fusion_payload = torch.load(fusion_checkpoint, map_location="cpu")
    lse_payload = torch.load(lse_checkpoint, map_location="cpu")
    layers = cfg.backbone.candidate_layers
    train_depths = tuple(sorted(set(cfg.joint.early_depths + (cfg.joint.full_depth,))))

    stats_path = _abs(cfg.data.descriptor_stats) if cfg.data.descriptor_stats else None
    normalizer = (
        DescriptorNormalizer.load(stats_path)
        if stats_path is not None and stats_path.is_file()
        else None
    )

    dlcm = DLCM(max_layer_id=max(layers), alpha=0.0)
    dlcm.load_state_dict(fusion_payload["dlcm"])
    state_dim = int(lse_payload.get("state_dim", 26))
    lse = LSE(state_dim=state_dim, early_depths=cfg.joint.early_depths)
    lse.load_state_dict(lse_payload["lse"])

    trainer = JointTrainer(
        dlcm=dlcm,
        lse=lse,
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=cfg.backbone.depth),
        train_depths=train_depths,
        candidate_layers=layers,
        early_depths=cfg.joint.early_depths,
        full_depth=cfg.joint.full_depth,
        fusion_loss_weight=cfg.joint.fusion_loss_weight,
        lse_loss_weight=cfg.joint.lse_loss_weight,
        compute_final_weight=cfg.joint.compute_final_weight,
        compute_ramp_fraction=cfg.joint.compute_ramp_fraction,
        compute_target_depth_ratio=cfg.joint.compute_target_depth_ratio,
        soft_exit_temperature=cfg.joint.soft_exit_temperature,
        epsilon_gain=cfg.joint.epsilon_gain,
        epsilon_absolute=cfg.joint.epsilon_absolute,
        sufficiency_weight=cfg.joint.sufficiency_weight,
        fusion_loss_weights=FusionLossWeights(),
        normalizer=normalizer,
    ).to(device)

    assert_trainable_only_dlcm_lse(trainer)
    return trainer


def build_loaders(
    *,
    cfg: ExperimentConfig,
    train_depths: tuple[int, ...],
    batch_size: int,
    limit_train: int | None,
    limit_cal: int | None,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, TeacherCacheDataset, TeacherCacheDataset]:
    assert cfg.data is not None
    if cfg.data.train_cache is None or cfg.data.calibration_cache is None:
        raise ValueError("joint training requires data.train_cache and data.calibration_cache")
    train_cache = TeacherCacheDataset(_abs(cfg.data.train_cache))
    cal_cache = TeacherCacheDataset(_abs(cfg.data.calibration_cache))
    shapley_path = _abs(cfg.data.shapley_targets) if cfg.data.shapley_targets else None
    shapley = (
        fusion_helpers.load_shapley_index(shapley_path)
        if shapley_path is not None and shapley_path.is_file()
        else {}
    )
    data_root = _abs(cfg.data.data_path)
    layers = cfg.backbone.candidate_layers

    train_ds = fusion_helpers.FusionCacheDataset(
        train_cache,
        shapley,
        data_root,
        cfg.image_size,
        layers,
        train_depths,
        limit=limit_train,
    )
    cal_ds = fusion_helpers.FusionCacheDataset(
        cal_cache,
        shapley,
        data_root,
        cfg.image_size,
        layers,
        train_depths,
        limit=limit_cal,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=fusion_helpers.collate_fusion,
        num_workers=num_workers,
    )
    cal_loader = DataLoader(
        cal_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=fusion_helpers.collate_fusion,
        num_workers=0,
    )
    return train_loader, cal_loader, train_cache, cal_cache


@torch.no_grad()
def evaluate_full_depth_metrics(
    trainer: JointTrainer,
    cal_loader: DataLoader,
    device: torch.device,
    *,
    full_depth: int,
) -> dict[str, float]:
    trainer.eval()
    aps: list[float] = []
    pros: list[float] = []
    errors: list[float] = []
    for batch in cal_loader:
        batch = fusion_helpers.move_batch(batch, device)
        maps = batch["maps_by_depth"][full_depth]
        layer_ids = batch["layer_ids_by_depth"][full_depth]
        b, n_layers = maps.shape[:2]
        valid = torch.ones(b, n_layers, dtype=torch.bool, device=device)
        maps_4d = maps.squeeze(2)
        layer_desc = trainer.layer_extractor(maps_4d, valid_mask=valid)
        if trainer.normalizer is not None:
            flat = layer_desc.reshape(b * n_layers, -1)
            flat = trainer.normalizer.transform(flat)
            layer_desc = flat.view(b, n_layers, -1)
        ctx = trainer.context_extractor(
            maps_4d,
            valid_mask=valid,
            layer_ids=layer_ids,
            prev_fused=None,
        )
        weights = trainer.dlcm(layer_desc, ctx, layer_ids, valid)
        fused = sum_preserving_fusion(maps, weights, valid)
        err = sample_localization_error(
            fused, batch["mask"], batch["image_label"]
        ).mean()
        for i in range(fused.shape[0]):
            pred = torch.sigmoid(fused[i]).squeeze().detach().cpu().numpy()
            gt = batch["mask"][i].squeeze().detach().cpu().numpy()
            aps.append(pixel_average_precision(pred, gt))
            pros.append(pro_score_proxy(pred, gt))
        errors.append(float(err.item()))
    return {
        "pixel_ap": float(sum(aps) / max(len(aps), 1)),
        "pro": float(sum(pros) / max(len(pros), 1)),
        "mean_sample_error": float(sum(errors) / max(len(errors), 1)),
    }


@torch.no_grad()
def evaluate_calibration_losses(
    trainer: JointTrainer,
    cal_loader: DataLoader,
    device: torch.device,
    *,
    progress: float,
) -> dict[str, float]:
    trainer.eval()
    fusion_total = 0.0
    lse_total = 0.0
    compute_total = 0.0
    n_batches = 0
    for batch in cal_loader:
        batch = fusion_helpers.move_batch(batch, device)
        fusion_result = compute_fusion_objective(
            trainer.dlcm,
            batch,
            trainer._fusion_config(),
            training_fraction=progress,
        )
        target_by_depth, _ = trainer._online_targets(fusion_result.sample_errors)
        state_by_depth = trainer._build_lse_states(batch)
        lse_result = compute_lse_objective(
            trainer.lse,
            state_by_depth,
            target_by_depth,
            trainer._lse_config(),
        )
        depth_ratio = soft_expected_depth_ratio(
            lse_result.sufficiency_logits,
            trainer.early_depths,
            trainer.full_depth,
            trainer.soft_exit_temperature,
        )
        compute_loss = torch.relu(
            depth_ratio.mean() - trainer.compute_target_depth_ratio
        )
        fusion_total += float(fusion_result.total_loss.item())
        lse_total += float(lse_result.total_loss.item())
        compute_total += float(compute_loss.item())
        n_batches += 1
    denom = max(n_batches, 1)
    return {
        "fusion_loss": fusion_total / denom,
        "lse_loss": lse_total / denom,
        "compute_loss": compute_total / denom,
    }


def save_diagnostic_checkpoint(
    *,
    path: Path,
    trainer: JointTrainer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    eligible_for_evaluation: bool,
    smoke_test: bool,
    gate: Any,
    cal_metrics: dict[str, float],
    cal_losses: dict[str, float],
    selection_score: float,
    cfg_hash: str,
    seed: int,
    staged: StagedCheckpointInfo,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "eligible_for_evaluation": eligible_for_evaluation,
        "smoke_test": smoke_test,
        "primary_pipeline": False,
        "experiment_kind": "training_ablation",
        "dlcm": trainer.dlcm.state_dict(),
        "lse": trainer.lse.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config_hash": cfg_hash,
        "git_sha": git_sha(),
        "seed": seed,
        "fusion_checkpoint_sha256": staged.fusion_sha256,
        "lse_checkpoint_sha256": staged.lse_sha256,
        "gate": {
            "passed": gate.passed,
            "pixel_ap_drop": gate.pixel_ap_drop,
            "pro_drop": gate.pro_drop,
            "mean_error_relative_increase": gate.mean_error_relative_increase,
            "reasons": list(gate.reasons),
        },
        "cal_metrics": cal_metrics,
        "cal_losses": cal_losses,
        "selection_score": selection_score,
        "trainable_parameter_names": trainer.trainable_parameter_names(),
    }
    torch.save(payload, path)


def run_joint_training(
    *,
    cfg: ExperimentConfig,
    staged: StagedCheckpointInfo,
    fusion_checkpoint: Path,
    lse_checkpoint: Path,
    output_dir: Path,
    seed: int,
    device: torch.device,
    smoke_test: bool,
) -> int:
    assert cfg.training is not None
    assert cfg.joint is not None
    assert cfg.data is not None

    cfg_hash = config_hash(cfg)
    sha = git_sha()
    epochs = 1 if smoke_test else cfg.training.epochs
    limit_train = 16 if smoke_test else None
    limit_cal = 8 if smoke_test else None
    batch_size = cfg.training.batch_size
    train_depths = tuple(sorted(set(cfg.joint.early_depths + (cfg.joint.full_depth,))))

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    trainer = load_joint_trainer(
        cfg=cfg,
        staged=staged,
        fusion_checkpoint=fusion_checkpoint,
        lse_checkpoint=lse_checkpoint,
        device=device,
    )
    train_loader, cal_loader, train_cache, _cal_cache = build_loaders(
        cfg=cfg,
        train_depths=train_depths,
        batch_size=batch_size,
        limit_train=limit_train,
        limit_cal=limit_cal,
        num_workers=0 if smoke_test else cfg.training.num_workers,
    )

    print(f"config_hash={cfg_hash}")
    print(f"git_sha={sha}")
    print(f"seed={seed}")
    print(f"split_manifest_hash={train_cache.meta.get('split_hash')}")
    print(f"fusion_checkpoint_sha256={staged.fusion_sha256}")
    print(f"lse_checkpoint_sha256={staged.lse_sha256}")
    print(f"trainable_modules={list(cfg.joint.trainable_modules)}")
    n_train = len(cast(Sized, train_loader.dataset))
    n_cal = len(cast(Sized, cal_loader.dataset))
    print(f"train_samples={n_train} cal_samples={n_cal}")
    print(f"epochs={epochs} batch_size={batch_size}")

    optimizer = torch.optim.AdamW(
        trainer.trainable_parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    resolved_cfg_path = output_dir / "resolved_config.yaml"
    resolved_cfg_path.write_text(
        yaml.safe_dump(
            {
                "seed": seed,
                "device": str(device),
                "smoke_test": smoke_test,
                "fusion_checkpoint": str(fusion_checkpoint),
                "lse_checkpoint": str(lse_checkpoint),
                "train_samples": n_train,
                "cal_samples": n_cal,
            },
            sort_keys=True,
        )
        + "\n"
    )

    thresholds = NoRegressionThresholds(
        max_pixel_ap_drop=cfg.joint.no_regression.max_pixel_ap_drop,
        max_pro_drop=cfg.joint.no_regression.max_pro_drop,
        max_mean_error_relative_increase=cfg.joint.no_regression.max_mean_error_relative_increase,
    )
    final_compute_weight = cfg.joint.compute_final_weight

    total_steps = max(1, epochs * len(train_loader))
    global_step = 0
    best_score = float("inf")
    gate_passed_any = False
    best_epoch: int | None = None
    start_ts = datetime.now(timezone.utc).isoformat()
    first_batch_shapes: dict[str, Any] | None = None

    for epoch in range(epochs):
        trainer.train()
        epoch_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"joint-train-ep{epoch}"):
            progress = global_step / float(total_steps)
            batch = fusion_helpers.move_batch(batch, device)
            batch["source_dataset"] = cfg.zero_shot.source_dataset
            if first_batch_shapes is None:
                first_batch_shapes = {
                    "maps_by_depth": {
                        str(d): list(batch["maps_by_depth"][d].shape)
                        for d in batch["maps_by_depth"]
                    }
                }
            optimizer.zero_grad(set_to_none=True)
            step_out = trainer.train_step(batch, progress, optimizer)
            epoch_losses.append(float(step_out.total_loss.item()))
            global_step += 1

        train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        val_progress = min(1.0, global_step / float(total_steps))
        cal_metrics = evaluate_full_depth_metrics(
            trainer, cal_loader, device, full_depth=cfg.joint.full_depth
        )
        gate = evaluate_no_regression(
            cal_metrics, staged.reference_full_depth_metrics, thresholds
        )
        cal_losses = evaluate_calibration_losses(
            trainer, cal_loader, device, progress=val_progress
        )
        selection_score = (
            cal_losses["fusion_loss"]
            + 0.5 * cal_losses["lse_loss"]
            + final_compute_weight * cal_losses["compute_loss"]
        )
        eligible = bool(gate.passed and not smoke_test)

        metric_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "cal_metrics": cal_metrics,
            "gate_passed": gate.passed,
            "gate_reasons": list(gate.reasons),
            "pixel_ap_drop": gate.pixel_ap_drop,
            "pro_drop": gate.pro_drop,
            "mean_error_relative_increase": gate.mean_error_relative_increase,
            "cal_fusion_loss": cal_losses["fusion_loss"],
            "cal_lse_loss": cal_losses["lse_loss"],
            "cal_compute_loss": cal_losses["compute_loss"],
            "selection_score": selection_score,
            "eligible_for_evaluation": eligible,
            "smoke_test": smoke_test,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(metric_row) + "\n")
        print(json.dumps(metric_row, indent=2))

        save_diagnostic_checkpoint(
            path=ckpt_dir / "last_diagnostic.pt",
            trainer=trainer,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            eligible_for_evaluation=False,
            smoke_test=smoke_test,
            gate=gate,
            cal_metrics=cal_metrics,
            cal_losses=cal_losses,
            selection_score=selection_score,
            cfg_hash=cfg_hash,
            seed=seed,
            staged=staged,
        )

        if smoke_test:
            continue

        if gate.passed and selection_score + 1e-12 < best_score:
            best_score = selection_score
            best_epoch = epoch
            gate_passed_any = True
            best_path = ckpt_dir / "best_gate_passed.pt"
            save_diagnostic_checkpoint(
                path=best_path,
                trainer=trainer,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                eligible_for_evaluation=True,
                smoke_test=False,
                gate=gate,
                cal_metrics=cal_metrics,
                cal_losses=cal_losses,
                selection_score=selection_score,
                cfg_hash=cfg_hash,
                seed=seed,
                staged=staged,
            )
            digest = sha256_file(best_path)
            manifest = CheckpointManifestV1(
                schema_version=SCHEMA_VERSION,
                stage="joint",
                status="passed",
                checkpoint_sha256=digest,
                candidate_layers=cfg.backbone.candidate_layers,
                source_dataset=staged.source_dataset,
                split_manifest_hash=staged.split_manifest_hash,
                preprocessing_hash=staged.preprocessing_hash,
                teacher_checkpoint_hash=staged.teacher_checkpoint_hash,
                descriptor_stats_hash=staged.descriptor_stats_hash,
                upstream_fusion_checkpoint_hash=staged.fusion_sha256,
                gates={"staged_training": True, "source_only_selection": True},
                reference_full_depth_metrics=cal_metrics,
            )
            write_checkpoint_with_manifest(best_path, manifest)

    end_ts = datetime.now(timezone.utc).isoformat()
    if smoke_test:
        status = "smoke_test"
        exit_code = 0
        eligible_for_evaluation = False
    elif gate_passed_any:
        status = "passed"
        exit_code = 0
        eligible_for_evaluation = True
    else:
        status = "gate_failed"
        exit_code = 2
        eligible_for_evaluation = False

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": "training_ablation",
        "primary_pipeline": False,
        "config_hash": cfg_hash,
        "git_sha": sha,
        "seed": seed,
        "split_manifest_hash": staged.split_manifest_hash,
        "cache_manifest_hash": str(train_cache.meta.get("split_hash")),
        "teacher_checkpoint_hash": staged.teacher_checkpoint_hash,
        "fusion_checkpoint_sha256": staged.fusion_sha256,
        "lse_checkpoint_sha256": staged.lse_sha256,
        "candidate_layers": list(cfg.backbone.candidate_layers),
        "early_depths": list(cfg.joint.early_depths),
        "preprocessing_hash": staged.preprocessing_hash,
        "descriptor_stats_hash": staged.descriptor_stats_hash,
        "trainable_parameter_names": trainer.trainable_parameter_names(),
        "trainable_parameter_count": len(trainer.trainable_parameter_names()),
        "tensor_shapes": first_batch_shapes or {},
        "no_regression_thresholds": {
            "max_pixel_ap_drop": thresholds.max_pixel_ap_drop,
            "max_pro_drop": thresholds.max_pro_drop,
            "max_mean_error_relative_increase": thresholds.max_mean_error_relative_increase,
        },
        "smoke_test": smoke_test,
        "eligible_for_evaluation": eligible_for_evaluation,
        "started_at": start_ts,
        "finished_at": end_ts,
        "status": status,
        "best_epoch": best_epoch,
        "best_selection_score": best_score if gate_passed_any else None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    summary = {
        "status": status,
        "epochs_ran": epochs,
        "global_steps": global_step,
        "gate_passed_any": gate_passed_any,
        "best_epoch": best_epoch,
        "best_selection_score": best_score if gate_passed_any else None,
        "smoke_test": smoke_test,
        "eligible_for_evaluation": eligible_for_evaluation,
        "config_hash": cfg_hash,
        "git_sha": sha,
        "seed": seed,
        "fusion_checkpoint_sha256": staged.fusion_sha256,
        "lse_checkpoint_sha256": staged.lse_sha256,
    }
    if status == "gate_failed":
        summary["reasons"] = ["no_epoch_passed_no_regression_gate"]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return exit_code


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(_abs(args.config))
    seed = int(args.seed if args.seed is not None else cfg.seed)
    output_dir = args.output_dir or (REPO_ROOT / "artifacts" / "joint" / f"mvtec_seed{seed}")
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    try:
        ensure_launch_allowed(args, cfg)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    fusion_checkpoint = _abs(args.fusion_checkpoint)
    lse_checkpoint = _abs(args.lse_checkpoint)
    staged = validate_staged_checkpoint_pair(
        fusion_checkpoint,
        lse_checkpoint,
        expected_candidate_layers=cfg.backbone.candidate_layers,
    )

    if args.dry_run:
        print_dry_run_plan(
            cfg=cfg,
            staged=staged,
            output_dir=output_dir,
            seed=seed,
            smoke_test=args.smoke_test,
        )
        return 0

    if args.smoke_test:
        if output_dir.is_dir() and (output_dir / "manifest.json").is_file():
            pass
    else:
        ensure_output_dir_available(output_dir)

    device = torch.device(args.device or cfg.device)
    return run_joint_training(
        cfg=cfg,
        staged=staged,
        fusion_checkpoint=fusion_checkpoint,
        lse_checkpoint=lse_checkpoint,
        output_dir=output_dir,
        seed=seed,
        device=device,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    raise SystemExit(main())
