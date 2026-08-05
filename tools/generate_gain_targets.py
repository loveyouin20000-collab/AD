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
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.config import ExperimentConfig  # noqa: E402
from rad.data.cache_dataset import TeacherCacheDataset  # noqa: E402
from rad.losses.localization import sample_localization_error  # noqa: E402
from rad.models.descriptors import (  # noqa: E402
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import sum_preserving_fusion  # noqa: E402
from rad.phase_b import b2_lse_prerequisites as prereq  # noqa: E402
from rad.targets.residual_gain import build_gain_target_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate residual-gain / sufficiency targets")
    p.add_argument("--config", type=str, default="configs/rad/lse.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--dlcm-checkpoint", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epsilon-gain", type=float, default=None)
    p.add_argument("--epsilon-absolute", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force", action="store_true")
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
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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
def fuse_errors_for_sample(
    *,
    sample: dict[str, Any],
    dlcm: prereq.LSEDLCMAdapter,
    layer_extractor: LayerDescriptorExtractor,
    context_extractor: CheckpointContextExtractor,
    normalizer: DescriptorNormalizer | None,
    candidate_layers: tuple[int, ...],
    train_depths: tuple[int, ...],
    data_root: Path,
    image_size: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    mask = load_mask(data_root, str(sample.get("mask_path") or ""), image_size).to(device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    label = torch.tensor([float(sample["label"])], device=device)
    prev_fused: torch.Tensor | None = None
    errors: dict[int, torch.Tensor] = {}

    for depth in train_depths:
        avail = [x for x in candidate_layers if x <= depth]
        stacked = torch.stack([sample["maps"][depth][layer] for layer in avail], dim=0)
        maps = stacked.unsqueeze(0).unsqueeze(2).to(device)  # [1, L, 1, H, W]
        layer_ids = torch.tensor([avail], dtype=torch.long, device=device)
        b, l = maps.shape[:2]
        valid = torch.ones(b, l, dtype=torch.bool, device=device)

        maps_4d = maps.squeeze(2)
        layer_desc = layer_extractor(maps_4d, valid_mask=valid)
        if normalizer is not None:
            flat = layer_desc.reshape(b * l, -1)
            flat = normalizer.transform(flat)
            layer_desc = flat.view(b, l, -1)
        ctx = context_extractor(
            maps_4d,
            valid_mask=valid,
            layer_ids=layer_ids,
            prev_fused=prev_fused,
        )
        weights = dlcm.weights(
            layer_desc,
            prediction_depth=int(depth),
            player_layer_ids=tuple(avail),
            context=ctx,
            layer_ids=layer_ids,
            valid_mask=valid,
        )
        fused = sum_preserving_fusion(maps, weights, valid)
        err = sample_localization_error(fused, mask, label)
        errors[int(depth)] = err.detach().cpu()
        prev_fused = fused.detach()

    return errors


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = ExperimentConfig.from_yaml(args.config)
    lse_cfg = raw.get("lse", {})
    fusion_cfg = raw.get("fusion", {})

    seed = args.seed if args.seed is not None else cfg.seed
    torch.manual_seed(seed)
    device = torch.device(args.device or raw.get("device", cfg.device))

    cache_dir = args.cache or Path(lse_cfg.get("train_cache", fusion_cfg.get("train_cache", "")))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir

    ckpt_path = args.dlcm_checkpoint or Path(
        lse_cfg.get("dlcm_checkpoint", "artifacts/checkpoints/fusion/seed_222/dlcm.pt")
    )
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path

    out_dir = args.output_dir or Path(lse_cfg.get("output_dir", "artifacts/targets/gain"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    output = args.output or (out_dir / "mvtec_train.pt")
    if not output.is_absolute():
        output = REPO_ROOT / output

    candidate_layers = tuple(cfg.backbone.candidate_layers)
    train_depths = tuple(
        int(x) for x in lse_cfg.get("train_depths", fusion_cfg.get("train_depths", [12, 18, 24]))
    )
    early_depths = tuple(int(x) for x in lse_cfg.get("early_depths", [12, 18]))
    full_depth = int(lse_cfg.get("full_depth", max(train_depths)))
    epsilon_gain = float(
        args.epsilon_gain if args.epsilon_gain is not None else lse_cfg.get("epsilon_gain", 0.05)
    )
    epsilon_absolute = float(
        args.epsilon_absolute
        if args.epsilon_absolute is not None
        else lse_cfg.get("epsilon_absolute", 0.5)
    )

    stats_path = Path(lse_cfg.get("descriptor_stats", fusion_cfg.get("descriptor_stats", "")))
    if stats_path and not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path

    config_hash = sha256_file(Path(args.config))
    checkpoint_hash = sha256_file(ckpt_path) if ckpt_path.is_file() else "missing"
    sha = git_sha()

    print(f"config: {args.config}")
    print(f"config_hash: {config_hash}")
    print(f"git_sha: {sha}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"cache: {cache_dir}")
    print(f"dlcm_checkpoint: {ckpt_path}")
    print(f"checkpoint_hash: {checkpoint_hash}")
    print(f"output: {output}")
    print(f"train_depths: {train_depths}")
    print(f"early_depths: {early_depths}")
    print(f"epsilon_gain: {epsilon_gain} epsilon_absolute: {epsilon_absolute}")

    if args.dry_run:
        return 0

    if not ckpt_path.is_file():
        raise SystemExit(f"missing DLCM checkpoint: {ckpt_path}")
    if output.exists() and not args.force:
        raise SystemExit(f"output exists: {output} (pass --force)")

    dataset = TeacherCacheDataset(cache_dir)
    split_hash = dataset.meta.get("split_hash")
    print(f"split_manifest_hash: {split_hash}")
    print(f"n_cache: {len(dataset)}")

    normalizer = DescriptorNormalizer.load(stats_path) if stats_path.is_file() else None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dlcm = prereq.load_lse_dlcm_adapter_from_checkpoint(
        ckpt,
        device=device,
        candidate_layers=candidate_layers,
    )
    dlcm_beta = dlcm.beta

    layer_extractor = LayerDescriptorExtractor()
    context_extractor = CheckpointContextExtractor(backbone_depth=cfg.backbone.depth)
    data_root = Path(dataset.meta.get("data_path", cfg.data.data_path if cfg.data else "."))
    image_size = int(dataset.meta.get("image_size", cfg.image_size))

    n = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    records: list[dict[str, Any]] = []
    logged_shapes = False

    for i in tqdm(range(n), desc="gain_targets"):
        sample = dataset[i]
        with torch.no_grad():
            errors = fuse_errors_for_sample(
                sample=sample,
                dlcm=dlcm,
                layer_extractor=layer_extractor,
                context_extractor=context_extractor,
                normalizer=normalizer,
                candidate_layers=candidate_layers,
                train_depths=train_depths,
                data_root=data_root,
                image_size=image_size,
                device=device,
            )
        if not logged_shapes:
            for d, e in errors.items():
                print(f"tensor_shapes depth={d} error={tuple(e.shape)}")
            logged_shapes = True

        rec = build_gain_target_record(
            errors,
            epsilon_gain=epsilon_gain,
            epsilon_absolute=epsilon_absolute,
            early_depths=early_depths,
            full_depth=full_depth,
            stop_gradient=True,
        )
        records.append(
            {
                "sample_id": sample["sample_id"],
                "label": int(sample["label"]),
                "category": sample["category"],
                **rec,
            }
        )

    payload = {
        "schema_version": 1,
        "seed": seed,
        "config_hash": config_hash,
        "git_sha": sha,
        "checkpoint_hash": checkpoint_hash,
        "split_manifest_hash": split_hash,
        "candidate_layers": list(candidate_layers),
        "train_depths": list(train_depths),
        "early_depths": list(early_depths),
        "full_depth": full_depth,
        "epsilon_gain": epsilon_gain,
        "epsilon_absolute": epsilon_absolute,
        "dlcm_checkpoint": str(ckpt_path),
        "dlcm_beta": dlcm_beta,
        "cache_meta": {
            "split_hash": split_hash,
            "checkpoint_hash": dataset.meta.get("checkpoint_hash"),
            "preprocessing_hash": dataset.meta.get("preprocessing_hash"),
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    meta = {
        "n_records": len(records),
        "early_depths": list(early_depths),
        "epsilon_gain": epsilon_gain,
        "epsilon_absolute": epsilon_absolute,
        "config_hash": config_hash,
        "git_sha": sha,
        "checkpoint_hash": checkpoint_hash,
        "split_manifest_hash": split_hash,
        "output": str(output),
        "best_note": "raw errors/gains stored for threshold recalibration",
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote: {output} ({len(records)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
