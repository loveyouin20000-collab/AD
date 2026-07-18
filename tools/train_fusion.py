from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.config import ExperimentConfig  # noqa: E402
from rad.data.cache_dataset import TeacherCacheDataset  # noqa: E402
from rad.models.descriptors import (  # noqa: E402
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM, sum_preserving_fusion  # noqa: E402
from rad.trainers.fusion_trainer import (  # noqa: E402
    FusionLossWeights,
    FusionTrainer,
    pixel_average_precision,
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
    ) -> None:
        self.cache = cache
        self.shapley_by_id = shapley_by_id
        self.data_root = data_root
        self.image_size = image_size
        self.candidate_layers = candidate_layers
        self.train_depths = train_depths
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
            # Fallback equal distribution if missing
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


@torch.no_grad()
def evaluate_pixel_ap(
    trainer: FusionTrainer,
    loader: DataLoader,
    device: torch.device,
    depth: int = 24,
) -> dict[str, float]:
    trainer.eval()
    aps_dlcm: list[float] = []
    aps_equal: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        maps = batch["maps_by_depth"][depth]
        layer_ids = batch["layer_ids_by_depth"][depth]
        b, l = maps.shape[:2]
        valid = torch.ones(b, l, dtype=torch.bool, device=device)
        layer_desc, ctx = trainer._describe(maps, layer_ids, valid, prev_fused=None)
        weights = trainer.dlcm(layer_desc, ctx, layer_ids, valid)
        fused = sum_preserving_fusion(maps, weights, valid)
        equal = maps.sum(dim=1)
        aps_dlcm.append(pixel_average_precision(fused, batch["mask"]))
        aps_equal.append(pixel_average_precision(equal, batch["mask"]))
    return {
        "pixel_ap": float(sum(aps_dlcm) / max(len(aps_dlcm), 1)),
        "equal_pixel_ap": float(sum(aps_equal) / max(len(aps_equal), 1)),
    }


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

    layers = cfg.backbone.candidate_layers
    train_depths = tuple(int(x) for x in fusion_cfg.get("train_depths", [12, 18, 24]))
    train_cache = TeacherCacheDataset(REPO_ROOT / fusion_cfg["train_cache"])
    cal_cache = TeacherCacheDataset(REPO_ROOT / fusion_cfg["calibration_cache"])
    shapley = load_shapley_index(REPO_ROOT / fusion_cfg["shapley_targets"])
    stats_path = REPO_ROOT / fusion_cfg["descriptor_stats"]
    normalizer = DescriptorNormalizer.load(stats_path) if stats_path.is_file() else None

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
    )
    cal_ds = FusionCacheDataset(
        cal_cache,
        {},  # cal may not have shapley; eval only needs maps/masks
        data_root,
        image_size,
        layers,
        train_depths,
        limit=limit_cal,
    )
    # Provide dummy shapley for cal collate
    for i in range(len(cal_ds)):
        pass

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
    best = {"pixel_ap": -1.0, "path": None, "equal_pixel_ap": 0.0}

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

    cal_metrics = evaluate_pixel_ap(trainer, cal_loader, device, depth=24)
    ckpt_path = seed_dir / "dlcm.pt"
    torch.save(
        {
            "seed": seed,
            "dlcm": trainer.dlcm.state_dict(),
            "cal_metrics": cal_metrics,
            "candidate_layers": list(layers),
            "train_depths": list(train_depths),
            "freeze_backbone": True,
        },
        ckpt_path,
    )
    (seed_dir / "history.json").write_text(json.dumps(history[-50:], indent=2) + "\n")
    (seed_dir / "cal_metrics.json").write_text(json.dumps(cal_metrics, indent=2) + "\n")

    no_regression = cal_metrics["pixel_ap"] + 1e-6 >= cal_metrics["equal_pixel_ap"]
    result = {
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "cal_metrics": cal_metrics,
        "no_regression_vs_equal": no_regression,
        "steps": global_step,
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

    print(f"config: {args.config}")
    print(f"device: {device}")
    print(f"seeds: {seeds}")
    print(f"epochs: {epochs}")
    print(f"output_dir: {output_dir}")
    print(f"max_steps: {args.max_steps}")
    print(f"freeze_backbone: True")

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

    # Select best by calibration pixel AP subject to no regression at layer 24
    eligible = [r for r in results if r["no_regression_vs_equal"]]
    pool = eligible if eligible else results
    best = max(pool, key=lambda r: r["cal_metrics"]["pixel_ap"])
    summary = {"results": results, "best": best}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
