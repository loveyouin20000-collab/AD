from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.config import ExperimentConfig  # noqa: E402
from rad.evaluation.export import TransferSamplePrediction, export_transfer_predictions  # noqa: E402
from rad.evaluation.zero_shot import (  # noqa: E402
    TargetAccessError,
    assert_policy_unchanged,
    boundary_complexity,
    boundary_f_score,
    forbid_target_access_during_calibration,
    load_frozen_policy_profile,
    pixel_average_precision,
    pro_score_proxy,
)
from tools.evaluate_adaptive import build_engine  # noqa: E402


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


def _load_visa_index(data_path: Path, limit: int | None) -> list[dict[str, Any]]:
    meta = json.loads((data_path / "meta.json").read_text())
    # VisualAD meta.json is typically {split: {cls: [records...]}}
    records: list[dict[str, Any]] = []
    test = meta.get("test") or meta.get("Test") or meta
    if isinstance(test, dict):
        for cls, items in test.items():
            if not isinstance(items, list):
                continue
            for it in items:
                rec = dict(it)
                rec.setdefault("cls_name", cls)
                records.append(rec)
    if limit is not None:
        records = records[: int(limit)]
    return records


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

    policy_path = _resolve(
        transfer.get("calibration_policy", adaptive.get("policy_profiles"))
    )
    target_path = _resolve(transfer.get("target_data_path", "/root/autodl-tmp/data/Visa"))

    config_hash = sha256_file(Path(args.config))
    sha = git_sha()
    print(f"config: {args.config}")
    print(f"config_hash: {config_hash}")
    print(f"git_sha: {sha}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"source_dataset: {cfg.zero_shot.source_dataset}")
    print(f"target_datasets: {cfg.zero_shot.target_datasets}")
    print(f"target_tuning: {cfg.zero_shot.target_tuning}")
    print(f"policy_path: {policy_path}")
    print(f"profile: {profile_name}")
    print(f"target_data_path: {target_path}")
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

    if not target_path.is_dir():
        raise SystemExit(f"missing target data path: {target_path}")

    engine = build_engine(raw=raw, cfg=cfg, device=device, profile=profile)
    # Freeze policy digest again after engine build
    assert_policy_unchanged(policy_path, profile_name, policy_digest)

    records = _load_visa_index(target_path, None if limit is None else int(limit))
    print(f"n_target_samples: {len(records)}")

    from PIL import Image
    import torchvision.transforms as T

    image_size = int(raw.get("image_size", 518))
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    rows: list[TransferSamplePrediction] = []
    adaptive_maps: list[np.ndarray] = []
    full_maps: list[np.ndarray] = []
    masks_np: list[np.ndarray] = []
    images_np: list[np.ndarray] = []

    for rec in tqdm(records, desc="zero_shot_transfer"):
        # Relative image path conventions in VisA meta
        img_rel = rec.get("img_path") or rec.get("image_path") or rec.get("img")
        mask_rel = rec.get("mask_path") or rec.get("mask")
        if img_rel is None:
            continue
        img_path = target_path / str(img_rel)
        if not img_path.is_file():
            # Sometimes paths already include dataset root segment
            img_path = Path(str(img_rel))
        label = int(rec.get("anomaly", rec.get("label", 0)))
        sample_id = str(rec.get("sample_id") or f"{rec.get('cls_name', 'unk')}/{img_rel}")

        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            adaptive = engine.infer(tensor, force_full_depth=False)
            full = engine.infer(tensor, force_full_depth=True)

        amap = adaptive.final_map[0].detach().float().cpu().numpy()
        fmap = full.final_map[0].detach().float().cpu().numpy()
        # sigmoid for metric space
        amap_p = 1.0 / (1.0 + np.exp(-amap))
        fmap_p = 1.0 / (1.0 + np.exp(-fmap))

        if label > 0 and mask_rel:
            mask_path = target_path / str(mask_rel)
            if not mask_path.is_file():
                mask_path = Path(str(mask_rel))
            mask_img = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
            mask = (np.array(mask_img, dtype=np.float64) > 127).astype(np.float64)
        else:
            mask = np.zeros((image_size, image_size), dtype=np.float64)

        # Residual gain proxy: localization error reduction full vs adaptive (higher = more to gain)
        residual = float(max(0.0, pixel_average_precision(fmap_p, mask) - pixel_average_precision(amap_p, mask)))
        area = float(mask.mean())
        gray = np.array(image.resize((image_size, image_size))).astype(np.float64).mean(axis=2) / 255.0
        if mask.sum() > 0 and (1 - mask).sum() > 0:
            contrast = float(abs(gray[mask > 0.5].mean() - gray[mask < 0.5].mean()))
        else:
            contrast = float(gray.std())
        complexity = boundary_complexity(mask)

        rows.append(
            TransferSamplePrediction(
                sample_id=sample_id,
                dataset="visa",
                selected_depth=int(adaptive.selected_depth),
                image_label=label,
                residual_gain=residual,
                pixel_ap_adaptive=pixel_average_precision(amap_p, mask),
                pixel_ap_full=pixel_average_precision(fmap_p, mask),
                pro_adaptive=pro_score_proxy(amap_p, mask),
                pro_full=pro_score_proxy(fmap_p, mask),
                boundary_f_adaptive=boundary_f_score(amap_p, mask),
                boundary_f_full=boundary_f_score(fmap_p, mask),
                anomaly_area=area,
                contrast_proxy=contrast,
                boundary_complexity=complexity,
            )
        )
        adaptive_maps.append(amap_p)
        full_maps.append(fmap_p)
        masks_np.append(mask)
        images_np.append(gray)

    # Final policy integrity check (never retuned on target)
    assert_policy_unchanged(policy_path, profile_name, policy_digest)

    summary = export_transfer_predictions(
        rows,
        output_dir=output_dir,
        full_depth=full_depth,
        epsilon=epsilon,
        adaptive_maps=np.stack(adaptive_maps) if adaptive_maps else None,
        full_depth_maps=np.stack(full_maps) if full_maps else None,
        masks=np.stack(masks_np) if masks_np else None,
        images=np.stack(images_np) if images_np else None,
    )
    meta = {
        "config_hash": config_hash,
        "git_sha": sha,
        "seed": seed,
        "policy_path": str(policy_path),
        "policy_profile": profile_name,
        "policy_digest": policy_digest,
        "target_tuning": False,
        "target_data_path": str(target_path),
        "n_samples": len(rows),
        "summary": summary,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"n": len(rows), "summary_keys": list(summary.keys())}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TargetAccessError as exc:
        raise SystemExit(f"target access violation: {exc}") from exc
