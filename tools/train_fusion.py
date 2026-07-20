from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.config import ExperimentConfig  # noqa: E402
from rad.data.cache_dataset import TeacherCacheDataset  # noqa: E402
from rad.errors import ArtifactIntegrityError  # noqa: E402
from rad.evaluation.paper_metrics import compute_paper_metrics  # noqa: E402
from rad.models.descriptors import (  # noqa: E402
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM, sum_preserving_fusion  # noqa: E402
from rad.trainers.fusion_trainer import (  # noqa: E402
    FusionLossWeights,
    FusionTrainer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DLCM fusion on cached teacher maps")
    p.add_argument("--config", type=str, default="configs/rad/fusion.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-cal", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def require_shapley_targets(
    path: Path, *, allow_missing: bool
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    if not path.is_file():
        if allow_missing:
            return {}
        raise ArtifactIntegrityError(f"Missing Shapley targets: {path}")
    return load_shapley_index(path)


def require_descriptor_stats(
    path: Path, *, require: bool
) -> DescriptorNormalizer | None:
    if not path.is_file():
        if require:
            raise ArtifactIntegrityError(f"Missing descriptor statistics: {path}")
        return None
    return DescriptorNormalizer.load(path)


def preflight_fusion_artifacts(
    fusion_cfg: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, dict[int, dict[str, torch.Tensor]]], DescriptorNormalizer | None]:
    """Fail-closed artifact checks before any optimizer / epoch work."""
    allow_missing = bool(fusion_cfg.get("allow_missing_shapley", False))
    require_stats = bool(fusion_cfg.get("require_descriptor_stats", True))
    shapley = require_shapley_targets(
        repo_root / fusion_cfg["shapley_targets"],
        allow_missing=allow_missing,
    )
    normalizer = require_descriptor_stats(
        repo_root / fusion_cfg["descriptor_stats"],
        require=require_stats,
    )
    return shapley, normalizer


def validate_shapley_coverage(
    sample_ids: list[str],
    shapley_by_id: dict[str, dict[int, dict[str, torch.Tensor]]],
    *,
    allow_missing: bool,
) -> None:
    missing = [sid for sid in sample_ids if sid not in shapley_by_id]
    if missing and not allow_missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ArtifactIntegrityError(
            f"Missing Shapley targets for {len(missing)} sample(s): {preview}{more}"
        )


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


class FusionCacheDataset(Dataset):
    def __init__(
        self,
        cache: TeacherCacheDataset,
        shapley_by_id: dict[str, dict[int, dict[str, torch.Tensor]]],
        data_root: Path,
        image_size: int,
        candidate_layers: tuple[int, ...],
        train_depths: tuple[int, ...],
        limit: int | None = None,
        *,
        allow_missing_shapley: bool = False,
    ) -> None:
        self.cache = cache
        self.shapley_by_id = shapley_by_id
        self.data_root = data_root
        self.image_size = image_size
        self.candidate_layers = candidate_layers
        self.train_depths = train_depths
        self.allow_missing_shapley = allow_missing_shapley
        self.indices = list(range(len(cache)))
        if limit is not None:
            self.indices = self.indices[:limit]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.cache[self.indices[idx]]
        sid = sample["sample_id"]
        maps_by_depth: dict[int, torch.Tensor] = {}
        layer_ids_by_depth: dict[int, torch.Tensor] = {}
        for depth in self.train_depths:
            avail = [x for x in self.candidate_layers if x <= depth]
            stacked = torch.stack(
                [sample["maps"][depth][layer] for layer in avail], dim=0
            )  # [L,H,W]
            maps_by_depth[depth] = stacked.unsqueeze(1)  # [L,1,H,W]
            layer_ids_by_depth[depth] = torch.tensor(avail, dtype=torch.long)

        shapley = self.shapley_by_id.get(sid)
        if shapley is None:
            if not self.allow_missing_shapley:
                raise ArtifactIntegrityError(
                    f"Missing Shapley target for sample: {sid}"
                )
            shapley = {}
            for depth in self.train_depths:
                l = sum(1 for x in self.candidate_layers if x <= depth)
                shapley[depth] = {
                    "distribution": torch.full((l,), 1.0 / l),
                    "phi": torch.zeros(l),
                }

        return {
            "sample_id": sid,
            "maps_by_depth": maps_by_depth,
            "layer_ids_by_depth": layer_ids_by_depth,
            "mask": load_mask(
                self.data_root, str(sample.get("mask_path") or ""), self.image_size
            ),
            "image_label": torch.tensor(float(sample["label"])),
            "teacher_logits": sample["teacher_logits"].unsqueeze(0)
            if sample["teacher_logits"].ndim == 2
            else sample["teacher_logits"],
            "shapley_by_depth": shapley,
        }


def collate_fusion(batch: list[dict[str, Any]]) -> dict[str, Any]:
    depths = list(batch[0]["maps_by_depth"].keys())
    maps_by_depth = {
        d: torch.stack([b["maps_by_depth"][d] for b in batch], dim=0) for d in depths
    }
    layer_ids_by_depth = {
        d: torch.stack([b["layer_ids_by_depth"][d] for b in batch], dim=0) for d in depths
    }
    shapley_by_depth = {}
    for d in depths:
        shapley_by_depth[d] = {
            "distribution": torch.stack(
                [b["shapley_by_depth"][d]["distribution"] for b in batch], dim=0
            ),
            "phi": torch.stack([b["shapley_by_depth"][d]["phi"] for b in batch], dim=0),
        }
    teacher = torch.stack([b["teacher_logits"] for b in batch], dim=0)
    if teacher.ndim == 3:
        teacher = teacher.unsqueeze(1)
    return {
        "maps_by_depth": maps_by_depth,
        "layer_ids_by_depth": layer_ids_by_depth,
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "image_label": torch.stack([b["image_label"] for b in batch], dim=0),
        "teacher_logits": teacher,
        "shapley_by_depth": shapley_by_depth,
        "sample_ids": [b["sample_id"] for b in batch],
    }


def load_shapley_index(path: Path) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    payload = torch.load(path, map_location="cpu")
    out: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for rec in payload["records"]:
        out[rec["sample_id"]] = {
            int(d): {
                "distribution": v["distribution"],
                "phi": v["phi"],
            }
            for d, v in rec["depths"].items()
        }
    return out


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {"sample_ids": batch.get("sample_ids")}
    out["maps_by_depth"] = {
        d: t.to(device) for d, t in batch["maps_by_depth"].items()
    }
    out["layer_ids_by_depth"] = {
        d: t.to(device) for d, t in batch["layer_ids_by_depth"].items()
    }
    out["mask"] = batch["mask"].to(device)
    out["image_label"] = batch["image_label"].to(device)
    out["teacher_logits"] = batch["teacher_logits"].to(device)
    out["shapley_by_depth"] = {
        d: {k: v.to(device) for k, v in kv.items()}
        for d, kv in batch["shapley_by_depth"].items()
    }
    return out


def detach_maps_to_cpu_numpy(maps: list[torch.Tensor]) -> list[np.ndarray]:
    """Convert maps to detached CPU ndarrays (no autograd / GPU retention)."""
    out: list[np.ndarray] = []
    for t in maps:
        x = t.detach().float().cpu().numpy()
        out.append(np.array(x, dtype=np.float64, copy=True))
    return out


def _logits_map_to_prob_hw(t: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(t, np.ndarray):
        x = np.asarray(t, dtype=np.float64)
    else:
        x = t.detach().float().cpu().numpy().astype(np.float64)
    while x.ndim > 2 and x.shape[0] == 1:
        x = np.squeeze(x, axis=0)
    if x.ndim == 3 and x.shape[0] == 1:
        x = np.squeeze(x, axis=0)
    if x.ndim != 2:
        raise ArtifactIntegrityError(f"expected HxW anomaly map, got shape {tuple(x.shape)}")
    x = np.clip(x, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float64)


def dataset_level_calibration_metrics(
    *,
    fused_maps: list[torch.Tensor] | list[np.ndarray],
    equal_maps: list[torch.Tensor] | list[np.ndarray],
    masks: list[torch.Tensor] | list[np.ndarray],
    image_labels: list[float],
    aupro_max_fpr: float = 0.3,
    aupro_steps: int = 200,
) -> dict[str, Any]:
    """Accumulate all calibration samples and compute dataset-level PaperMetrics."""
    if not (len(fused_maps) == len(equal_maps) == len(masks) == len(image_labels)):
        raise ArtifactIntegrityError("calibration accumulation length mismatch")
    fused_a = np.stack([_logits_map_to_prob_hw(m) for m in fused_maps], axis=0)
    equal_a = np.stack([_logits_map_to_prob_hw(m) for m in equal_maps], axis=0)
    h, w = int(fused_a.shape[1]), int(fused_a.shape[2])
    masks_a = np.stack([_mask_to_hw(m, h, w) for m in masks], axis=0)
    labels_a = np.asarray(image_labels, dtype=np.float64)
    scores_f = fused_a.reshape(fused_a.shape[0], -1).max(axis=1)
    scores_e = equal_a.reshape(equal_a.shape[0], -1).max(axis=1)
    paper_f = compute_paper_metrics(
        image_labels=labels_a,
        image_scores=scores_f,
        masks=masks_a,
        anomaly_maps=fused_a,
        aupro_max_fpr=aupro_max_fpr,
        aupro_steps=aupro_steps,
        boundary_enabled=False,
    )
    paper_e = compute_paper_metrics(
        image_labels=labels_a,
        image_scores=scores_e,
        masks=masks_a,
        anomaly_maps=equal_a,
        aupro_max_fpr=aupro_max_fpr,
        aupro_steps=aupro_steps,
        boundary_enabled=False,
    )
    return {
        "pixel_ap": float(paper_f.pixel_ap),
        "equal_pixel_ap": float(paper_e.pixel_ap),
        "paper_metrics": paper_f.as_dict(),
        "equal_paper_metrics": paper_e.as_dict(),
    }


def _mask_to_hw(mask: torch.Tensor | np.ndarray, h: int, w: int) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        x = np.asarray(mask, dtype=np.float64).reshape(-1)
    else:
        x = mask.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
    if x.size != h * w:
        raise ArtifactIntegrityError(f"mask size {x.size} != {h * w}")
    return x.reshape(h, w)


def _seed_checkpoint_eligible(result: dict[str, Any]) -> bool:
    if result.get("legacy_mode"):
        return False
    if result.get("eligible_for_evaluation") is False:
        return False
    return bool(result.get("no_regression_vs_equal"))


@torch.no_grad()
def evaluate_calibration_paper_metrics(
    trainer: FusionTrainer,
    loader: DataLoader,
    device: torch.device,
    depth: int = 24,
) -> dict[str, Any]:
    trainer.eval()
    fused_maps: list[np.ndarray] = []
    equal_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        maps = batch["maps_by_depth"][depth]
        layer_ids = batch["layer_ids_by_depth"][depth]
        b, l = maps.shape[:2]
        valid = torch.ones(b, l, dtype=torch.bool, device=device)
        layer_desc, ctx = trainer._describe(  # type: ignore[operator]
            maps, layer_ids, valid, prev_fused=None
        )
        weights = trainer.dlcm(layer_desc, ctx, layer_ids, valid)
        fused = sum_preserving_fusion(maps, weights, valid)
        equal = maps.sum(dim=1)
        # Detach to CPU numpy immediately — do not retain GPU graphs.
        fused_cpu = detach_maps_to_cpu_numpy([fused[i] for i in range(b)])
        equal_cpu = detach_maps_to_cpu_numpy([equal[i] for i in range(b)])
        mask_cpu = detach_maps_to_cpu_numpy([batch["mask"][i] for i in range(b)])
        fused_maps.extend(fused_cpu)
        equal_maps.extend(equal_cpu)
        masks.extend(mask_cpu)
        for i in range(b):
            labels.append(float(batch["image_label"][i].item()))
    return dataset_level_calibration_metrics(
        fused_maps=fused_maps,
        equal_maps=equal_maps,
        masks=masks,
        image_labels=labels,
    )


def finalize_fusion_run(
    results: list[dict[str, Any]],
    output_dir: Path,
    *,
    fail_if_no_gate_passes: bool,
    legacy_mode: bool,
) -> tuple[dict[str, Any], int]:
    """Select best passing seed; fail closed when no seed passes the gate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Legacy run-mode and per-seed legacy/ineligible rows never contribute a best ckpt.
    eligible = [
        r
        for r in results
        if (not legacy_mode) and _seed_checkpoint_eligible(r)
    ]
    summary: dict[str, Any] = {
        "results": results,
        "legacy_mode": bool(legacy_mode),
        "fail_if_no_gate_passes": bool(fail_if_no_gate_passes),
    }

    if legacy_mode:
        summary["best"] = None
        summary["status"] = "completed"
        summary["eligible_for_evaluation"] = False
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary, 0

    if not eligible:
        summary["best"] = None
        summary["status"] = "gate_failed" if fail_if_no_gate_passes else "completed"
        summary["eligible_for_evaluation"] = False
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        code = 5 if fail_if_no_gate_passes else 0
        return summary, code

    best = max(eligible, key=lambda r: float(r["cal_metrics"]["pixel_ap"]))
    summary["best"] = best
    summary["status"] = "completed"
    summary["eligible_for_evaluation"] = True

    best_src = Path(best.get("gate_checkpoint") or best["checkpoint"])
    dest = output_dir / "best_gate_passed.pt"
    if best_src.is_file():
        shutil.copy2(best_src, dest)
    else:
        raise ArtifactIntegrityError(
            f"passing seed missing gate checkpoint: {best_src}"
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, 0


def train_one_seed(
    *,
    seed: int,
    cfg: ExperimentConfig,
    fusion_cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    max_steps: int | None,
    epochs: int,
    limit_train: int | None,
    limit_cal: int | None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    allow_missing_shapley = bool(fusion_cfg.get("allow_missing_shapley", False))
    # Fail closed on missing Shapley / descriptor stats before any training work.
    shapley, normalizer = preflight_fusion_artifacts(fusion_cfg, repo_root=REPO_ROOT)

    layers = cfg.backbone.candidate_layers
    train_depths = tuple(int(x) for x in fusion_cfg.get("train_depths", [12, 18, 24]))
    train_cache = TeacherCacheDataset(REPO_ROOT / fusion_cfg["train_cache"])
    cal_cache = TeacherCacheDataset(REPO_ROOT / fusion_cfg["calibration_cache"])

    data_root = Path(cfg.data.data_path) if cfg.data else Path(".")
    image_size = cfg.image_size

    train_ds = FusionCacheDataset(
        train_cache,
        shapley,
        data_root,
        image_size,
        layers,
        train_depths,
        limit=limit_train,
        allow_missing_shapley=allow_missing_shapley,
    )
    if not allow_missing_shapley:
        train_ids = [
            train_cache[train_ds.indices[i]]["sample_id"] for i in range(len(train_ds))
        ]
        validate_shapley_coverage(
            train_ids, shapley, allow_missing=allow_missing_shapley
        )

    cal_ds = FusionCacheDataset(
        cal_cache,
        shapley if allow_missing_shapley else {},
        data_root,
        image_size,
        layers,
        train_depths,
        limit=limit_cal,
        allow_missing_shapley=True,  # calibration only needs maps/masks
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(fusion_cfg.get("batch_size", 1)),
        shuffle=True,
        collate_fn=collate_fusion,
        num_workers=0,
    )
    cal_loader = DataLoader(
        cal_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fusion,
        num_workers=0,
    )

    dlcm = DLCM(
        max_layer_id=max(layers),
        alpha=0.0,
        alpha_init=float(fusion_cfg.get("alpha_init", 0.1)),
        alpha_warmup_fraction=float(fusion_cfg.get("alpha_warmup_fraction", 0.2)),
    ).to(device)
    trainer = FusionTrainer(
        dlcm=dlcm,
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=cfg.backbone.depth),
        normalizer=normalizer,
        loss_weights=FusionLossWeights(
            lambda_loc={int(k): float(v) for k, v in fusion_cfg.get("lambda_loc", {}).items()}
            or None,
            map_kd=float(fusion_cfg.get("map_kd", 0.5)),
            boundary_kd=float(fusion_cfg.get("boundary_kd", 0.2)),
            contribution=float(fusion_cfg.get("contribution", 0.5)),
        ),
        train_depths=train_depths,
        candidate_layers=layers,
        freeze_backbone=True,
    ).to(device)

    opt = torch.optim.Adam(trainer.trainable_parameters(), lr=float(fusion_cfg.get("lr", 3e-4)))
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    total_steps = max(1, epochs * len(train_loader))
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"seed{seed}-ep{epoch}"):
            progress = global_step / float(total_steps)
            trainer.dlcm.set_progress(progress)
            batch = move_batch(batch, device)
            metrics = trainer.training_step(batch, opt)
            metrics["step"] = float(global_step)
            history.append(metrics)
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is not None and global_step >= max_steps:
            break

    cal_metrics = evaluate_calibration_paper_metrics(
        trainer, cal_loader, device, depth=24
    )
    last_path = seed_dir / "last.pt"
    payload = {
        "seed": seed,
        "dlcm": trainer.dlcm.state_dict(),
        "cal_metrics": {
            k: v
            for k, v in cal_metrics.items()
            if k in ("pixel_ap", "equal_pixel_ap")
        },
        "paper_metrics": cal_metrics.get("paper_metrics"),
        "candidate_layers": list(layers),
        "train_depths": list(train_depths),
        "freeze_backbone": True,
        "legacy_mode": allow_missing_shapley,
        "eligible_for_evaluation": False,
    }
    torch.save(payload, last_path)
    (seed_dir / "history.json").write_text(json.dumps(history[-50:], indent=2) + "\n")
    (seed_dir / "calibration_metrics.json").write_text(
        json.dumps(cal_metrics, indent=2) + "\n"
    )

    no_regression = (
        float(cal_metrics["pixel_ap"]) + 1e-6
        >= float(cal_metrics["equal_pixel_ap"])
        - float(fusion_cfg.get("max_pixel_ap_drop", 0.0))
    )
    gate_path: Path | None = None
    if no_regression:
        gate_path = seed_dir / "gate_passed.pt"
        gate_payload = {
            **payload,
            "eligible_for_evaluation": not allow_missing_shapley,
            "no_regression_vs_equal": True,
        }
        torch.save(gate_payload, gate_path)
    (seed_dir / "gate_result.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "no_regression_vs_equal": no_regression,
                "pixel_ap": cal_metrics["pixel_ap"],
                "equal_pixel_ap": cal_metrics["equal_pixel_ap"],
                "legacy_mode": allow_missing_shapley,
            },
            indent=2,
        )
        + "\n"
    )

    result = {
        "seed": seed,
        "checkpoint": str(last_path),
        "gate_checkpoint": str(gate_path) if gate_path is not None else None,
        "cal_metrics": {
            "pixel_ap": float(cal_metrics["pixel_ap"]),
            "equal_pixel_ap": float(cal_metrics["equal_pixel_ap"]),
        },
        "paper_metrics": cal_metrics.get("paper_metrics"),
        "no_regression_vs_equal": no_regression,
        "steps": global_step,
        "legacy_mode": allow_missing_shapley,
        "eligible_for_evaluation": bool(no_regression and not allow_missing_shapley),
    }
    return result


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = ExperimentConfig.from_yaml(args.config)
    fusion_cfg = raw.get("fusion", {})
    device = torch.device(args.device or raw.get("device", cfg.device))
    output_dir = Path(args.output_dir or fusion_cfg.get("output_dir", "artifacts/checkpoints/fusion"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    seeds = args.seeds or fusion_cfg.get("seeds", [cfg.seed])
    if args.seed is not None:
        seeds = [args.seed]
    epochs = args.epochs if args.epochs is not None else int(fusion_cfg.get("epochs", 1))
    allow_missing_shapley = bool(fusion_cfg.get("allow_missing_shapley", False))
    fail_if_no_gate_passes = bool(fusion_cfg.get("fail_if_no_gate_passes", True))

    print(f"config: {args.config}")
    print(f"device: {device}")
    print(f"seeds: {seeds}")
    print(f"epochs: {epochs}")
    print(f"output_dir: {output_dir}")
    print(f"max_steps: {args.max_steps}")
    print(f"freeze_backbone: True")
    print(f"allow_missing_shapley: {allow_missing_shapley}")
    print(f"require_descriptor_stats: {fusion_cfg.get('require_descriptor_stats', True)}")
    print(f"fail_if_no_gate_passes: {fail_if_no_gate_passes}")

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        results.append(
            train_one_seed(
                seed=int(seed),
                cfg=cfg,
                fusion_cfg=fusion_cfg,
                device=device,
                output_dir=output_dir,
                max_steps=args.max_steps,
                epochs=epochs,
                limit_train=args.limit_train,
                limit_cal=args.limit_cal,
            )
        )

    summary, code = finalize_fusion_run(
        results,
        output_dir,
        fail_if_no_gate_passes=fail_if_no_gate_passes,
        legacy_mode=allow_missing_shapley,
    )
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
