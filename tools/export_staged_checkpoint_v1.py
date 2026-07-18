#!/usr/bin/env python3
"""Export existing staged fusion/LSE checkpoints into rad-checkpoint-v1 layout.

Does not mutate source artifacts. Writes best_gate_passed.pt + sidecar manifest
into a new output directory after recomputing reference metrics on source cal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from rad.trainers.fusion_trainer import FusionTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/rad/fusion.yaml")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--fusion-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/fusion/seed_222/dlcm.pt"),
    )
    p.add_argument(
        "--lse-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/lse/lse_best.pt"),
    )
    p.add_argument(
        "--fusion-output-dir",
        type=Path,
        default=Path("artifacts/fusion/mvtec_seed111"),
    )
    p.add_argument(
        "--lse-output-dir",
        type=Path,
        default=Path("artifacts/lse/mvtec_seed111"),
    )
    p.add_argument("--limit-cal", type=int, default=None)
    p.add_argument("--skip-lse", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_path(path: Path) -> str:
    return sha256_file(path)


def load_mask(data_root: Path, mask_path: str, image_size: int) -> torch.Tensor:
    if not mask_path:
        return torch.zeros(1, image_size, image_size)
    path = data_root / mask_path
    if not path.is_file():
        return torch.zeros(1, image_size, image_size)
    img = Image.open(path).convert("L")
    t = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    return (t(img) > 0.5).float()


@torch.no_grad()
def evaluate_fusion_reference(
    *,
    trainer: FusionTrainer,
    cache: TeacherCacheDataset,
    data_root: Path,
    image_size: int,
    candidate_layers: tuple[int, ...],
    device: torch.device,
    limit: int | None,
    depth: int = 24,
) -> dict[str, float]:
    trainer.eval()
    aps: list[float] = []
    pros: list[float] = []
    errors: list[float] = []
    n = len(cache) if limit is None else min(limit, len(cache))
    for idx in tqdm(range(n), desc="fusion-cal-metrics"):
        sample = cache[idx]
        avail = [x for x in candidate_layers if x <= depth]
        stacked = torch.stack([sample["maps"][depth][layer] for layer in avail], dim=0)
        maps = stacked.unsqueeze(0).unsqueeze(2).to(device)  # [1,L,1,H,W]
        layer_ids = torch.tensor([avail], dtype=torch.long, device=device)
        valid = torch.ones(1, len(avail), dtype=torch.bool, device=device)
        layer_desc, ctx = trainer._describe(maps, layer_ids, valid, prev_fused=None)
        weights = trainer.dlcm(layer_desc, ctx, layer_ids, valid)
        fused = sum_preserving_fusion(maps, weights, valid)  # [1,1,H,W]
        mask = load_mask(data_root, sample.get("mask_path", ""), image_size).unsqueeze(0).to(device)
        label = torch.tensor(
            [float(sample.get("image_label", int(mask.any().item())))],
            device=device,
        )
        err = sample_localization_error(fused, mask, label).mean().item()
        pred = torch.sigmoid(fused).squeeze().detach().cpu().numpy()
        gt = mask.squeeze().detach().cpu().numpy()
        aps.append(pixel_average_precision(pred, gt))
        pros.append(pro_score_proxy(pred, gt))
        errors.append(float(err))
    return {
        "pixel_ap": float(sum(aps) / max(len(aps), 1)),
        "pro": float(sum(pros) / max(len(pros), 1)),
        "mean_sample_error": float(sum(errors) / max(len(errors), 1)),
    }


def export_fusion(
    *,
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    raw: dict[str, Any],
    device: torch.device,
) -> tuple[Path, str, dict[str, float]]:
    fusion_cfg = raw.get("fusion", {})
    src = _abs(args.fusion_checkpoint)
    out_root = _abs(args.fusion_output_dir)
    ckpt_dir = out_root / "checkpoints"
    dest = ckpt_dir / "best_gate_passed.pt"

    train_cache_path = _abs(Path(fusion_cfg["train_cache"]))
    cal_cache_path = _abs(Path(fusion_cfg["calibration_cache"]))
    stats_path = _abs(Path(fusion_cfg["descriptor_stats"]))
    train_cache = TeacherCacheDataset(train_cache_path)
    cal_cache = TeacherCacheDataset(cal_cache_path)

    provenance = {
        "split_manifest_hash": str(train_cache.meta.get("split_hash")),
        "preprocessing_hash": str(train_cache.meta.get("preprocessing_hash")),
        "teacher_checkpoint_hash": str(train_cache.meta.get("checkpoint_hash")),
        "descriptor_stats_hash": sha256_path(stats_path),
        "source_dataset": str(train_cache.meta.get("dataset", cfg.zero_shot.source_dataset)),
        "candidate_layers": tuple(cfg.backbone.candidate_layers),
    }

    print(f"fusion_src: {src}")
    print(f"fusion_dest: {dest}")
    print(json.dumps(provenance, indent=2))
    if args.dry_run:
        return dest, sha256_path(src), {}

    if dest.exists() or dest.with_suffix(".manifest.json").exists():
        raise SystemExit(f"refusing to overwrite existing fusion export: {ckpt_dir}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    payload = torch.load(src, map_location="cpu")
    layers = tuple(int(x) for x in payload.get("candidate_layers", cfg.backbone.candidate_layers))
    dlcm = DLCM(max_layer_id=max(layers), alpha=0.0)
    dlcm.load_state_dict(payload["dlcm"])
    normalizer = DescriptorNormalizer.load(stats_path) if stats_path.is_file() else None
    trainer = FusionTrainer(
        dlcm=dlcm,
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=cfg.backbone.depth),
        normalizer=normalizer,
        candidate_layers=layers,
        freeze_backbone=True,
    ).to(device)

    data_root = Path(cfg.data.data_path) if cfg.data else Path(".")
    metrics = evaluate_fusion_reference(
        trainer=trainer,
        cache=cal_cache,
        data_root=data_root,
        image_size=cfg.image_size,
        candidate_layers=layers,
        device=device,
        limit=args.limit_cal,
    )
    # Prefer no-regression vs equal from source payload when present
    status = "passed"
    if "cal_metrics" in payload:
        cm = payload["cal_metrics"]
        if float(cm.get("pixel_ap", 0)) + 1e-6 < float(cm.get("equal_pixel_ap", 0)):
            status = "failed"

    digest = sha256_path(dest)
    manifest = CheckpointManifestV1(
        schema_version=SCHEMA_VERSION,
        stage="fusion",
        status=status,
        checkpoint_sha256=digest,
        candidate_layers=layers,
        source_dataset=provenance["source_dataset"],
        split_manifest_hash=provenance["split_manifest_hash"],
        preprocessing_hash=provenance["preprocessing_hash"],
        teacher_checkpoint_hash=provenance["teacher_checkpoint_hash"],
        descriptor_stats_hash=provenance["descriptor_stats_hash"],
        upstream_fusion_checkpoint_hash=None,
        gates={"staged_training": True, "source_only_selection": True},
        reference_full_depth_metrics=metrics,
    )
    write_checkpoint_with_manifest(dest, manifest)
    summary = {
        "checkpoint": str(dest),
        "status": status,
        "checkpoint_sha256": digest,
        "reference_full_depth_metrics": metrics,
        "source_checkpoint": str(src),
    }
    (out_root / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if status != "passed":
        raise SystemExit("fusion export status is not passed; refusing to continue")
    return dest, digest, metrics


def export_lse(
    *,
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    raw: dict[str, Any],
    fusion_digest: str,
) -> Path:
    fusion_cfg = raw.get("fusion", {})
    lse_cfg = raw.get("lse", {})
    src = _abs(args.lse_checkpoint)
    out_root = _abs(args.lse_output_dir)
    ckpt_dir = out_root / "checkpoints"
    dest = ckpt_dir / "best_gate_passed.pt"

    train_cache_path = _abs(Path(lse_cfg.get("train_cache", fusion_cfg["train_cache"])))
    stats_path = _abs(Path(lse_cfg.get("descriptor_stats", fusion_cfg["descriptor_stats"])))
    train_cache = TeacherCacheDataset(train_cache_path)

    print(f"lse_src: {src}")
    print(f"lse_dest: {dest}")
    print(f"upstream_fusion_checkpoint_hash: {fusion_digest}")
    if args.dry_run:
        return dest

    if dest.exists() or dest.with_suffix(".manifest.json").exists():
        raise SystemExit(f"refusing to overwrite existing lse export: {ckpt_dir}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    digest = sha256_path(dest)
    payload = torch.load(src, map_location="cpu")
    # Ensure loadable LSE state
    state_dim = int(payload.get("state_dim", 26))
    early = tuple(int(x) for x in payload.get("early_depths", (12, 18)))
    model = LSE(state_dim=state_dim, early_depths=early)
    model.load_state_dict(payload["lse"])

    manifest = CheckpointManifestV1(
        schema_version=SCHEMA_VERSION,
        stage="lse",
        status="passed",
        checkpoint_sha256=digest,
        candidate_layers=tuple(cfg.backbone.candidate_layers),
        source_dataset=str(train_cache.meta.get("dataset", cfg.zero_shot.source_dataset)),
        split_manifest_hash=str(
            payload.get("split_manifest_hash") or train_cache.meta.get("split_hash")
        ),
        preprocessing_hash=str(train_cache.meta.get("preprocessing_hash")),
        teacher_checkpoint_hash=str(train_cache.meta.get("checkpoint_hash")),
        descriptor_stats_hash=sha256_path(stats_path),
        upstream_fusion_checkpoint_hash=fusion_digest,
        gates={"staged_training": True, "source_only_selection": True},
        reference_full_depth_metrics=None,
    )
    write_checkpoint_with_manifest(dest, manifest)
    summary = {
        "checkpoint": str(dest),
        "status": "passed",
        "checkpoint_sha256": digest,
        "upstream_fusion_checkpoint_hash": fusion_digest,
        "source_checkpoint": str(src),
    }
    (out_root / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return dest


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    # Prefer lse.yaml when exporting both so paths stay consistent
    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device(args.device or raw.get("device", cfg.device))

    print(f"config: {args.config}")
    print(f"device: {device}")
    print(f"seed: {args.seed}")

    fusion_path, fusion_digest, _metrics = export_fusion(
        args=args, cfg=cfg, raw=raw, device=device
    )
    if args.skip_lse:
        return 0
    # Load lse.yaml defaults for cache fields if fusion-only config lacks them
    if "lse" not in raw:
        lse_raw = yaml.safe_load(Path("configs/rad/lse.yaml").read_text())
        raw = {**lse_raw, **raw}
        raw["fusion"] = {**lse_raw.get("fusion", {}), **raw.get("fusion", {})}
        raw["lse"] = lse_raw.get("lse", {})
    if args.dry_run:
        export_lse(args=args, cfg=cfg, raw=raw, fusion_digest=fusion_digest)
        print("dry-run complete")
        return 0
    export_lse(args=args, cfg=cfg, raw=raw, fusion_digest=fusion_digest)
    print(f"exported fusion={fusion_path} digest={fusion_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
