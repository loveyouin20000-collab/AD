#!/usr/bin/env python3
"""Evaluation-only AUPRO diagnostics for a frozen VisualAD baseline checkpoint.

Does not train, does not modify the completed baseline run directory, and does
not change utils.metrics.compute_metrics. Recomputes AUPRO with both
cal_pro_score (original) and safe_aupro (PaperMetrics).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage import measure
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import VisualAD_lib  # noqa: E402
from dataset import Dataset  # noqa: E402
from rad.artifacts import atomic_write_json, refuse_existing_run  # noqa: E402
from rad.evaluation.paper_metrics import compute_paper_metrics, safe_aupro  # noqa: E402
from utils.anomaly_detection import generate_anomaly_map_from_tokens  # noqa: E402
from utils.feature_transform import create_feature_transform  # noqa: E402
from utils.metrics import cal_pro_score  # noqa: E402
from utils.transforms import get_transform  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aupro_diagnostics(
    masks: np.ndarray,
    amaps: np.ndarray,
    *,
    max_fpr: float = 0.3,
    steps: int = 200,
) -> dict[str, Any]:
    """Single-pass AUPRO + intermediates (avoids triple-loop cost).

    Computes the VisualAD-compatible trapezoidal AUPRO once, then reports both
    ``cal_pro_score`` and ``safe_aupro`` by evaluating each exactly once on the
    same arrays (they share the same algorithm on valid inputs).
    """
    gt = (masks > 0.5).astype(np.float64)
    n_regions = 0
    for i in range(len(gt)):
        if float(gt[i].sum()) <= 0.0:
            continue
        labeled = measure.label(gt[i].astype(np.uint8), connectivity=2)
        n_regions += int(labeled.max())

    min_th = float(np.min(amaps))
    max_th = float(np.max(amaps))
    base = {
        "n_images": int(len(masks)),
        "n_anomalous_images": int(sum(1 for i in range(len(gt)) if gt[i].sum() > 0)),
        "n_connected_regions": n_regions,
        "amap_min": min_th,
        "amap_max": max_th,
    }
    if max_th <= min_th or float(gt.sum()) <= 0.0:
        return {
            **base,
            "valid_fpr_points": 0,
            "normalized_fpr_span": 0.0,
            "cal_pro_score": 0.0,
            "safe_aupro": 0.0,
            "fallback": "constant_map_or_no_anomaly",
        }

    # Two authoritative calls on the same arrays (original vs paper-safe).
    original = float(cal_pro_score(masks, amaps, max_step=steps, expect_fpr=max_fpr))
    safe = float(safe_aupro(masks, amaps, max_fpr=max_fpr, steps=steps))
    return {
        **base,
        # FPR point/span audit is embedded inside the metric functions; when both
        # return >0 the curve had >=2 valid FPR points and nonzero span.
        "valid_fpr_points": None,
        "normalized_fpr_span": None,
        "cal_pro_score": original,
        "safe_aupro": safe,
        "abs_diff_cal_vs_safe": abs(original - safe),
        "nonzero_implies_valid_curve": bool(original > 0.0 and safe > 0.0),
        "fallback": None,
    }


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    image_size = int(checkpoint.get("image_size", 518))
    features_list = list(checkpoint.get("features_list", [6, 12, 18, 24]))

    class Args:
        pass

    args = Args()
    args.backbone = backbone
    args.image_size = image_size
    args.features_list = features_list

    preprocess, target_transform = get_transform(args)
    model, _ = VisualAD_lib.load(backbone, device=device)
    model.eval()
    model.to(device)
    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)

    feature_dim = model.visual.embed_dim
    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        for layer_name, state_dict in checkpoint["layer_transforms"].items():
            hidden_dim = state_dict["mlp.0.weight"].shape[0]
            module = create_feature_transform(
                transform_type="mlp",
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0,
            ).to(device)
            module.load_state_dict(state_dict)
            module.eval()
            layer_transforms[layer_name] = module

    cross_attn = None
    if "cross_attn" in checkpoint:
        from utils.spatial_cross_attention import build_layer_adaptive_cross_attention

        config = checkpoint.get("cross_attn_config", {})
        cross_attn = build_layer_adaptive_cross_attention(
            layers=features_list,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            res_scale_init=config.get("res_scale_init", 0.01),
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    return {
        "model": model,
        "layer_transforms": layer_transforms,
        "cross_attn": cross_attn,
        "features_list": features_list,
        "image_size": image_size,
        "backbone": backbone,
        "preprocess": preprocess,
        "target_transform": target_transform,
        "checkpoint": checkpoint,
    }


def run_inference(
    *,
    bundle: dict[str, Any],
    data_path: Path,
    dataset_name: str,
    device: torch.device,
    sigma: float,
) -> dict[str, list]:
    test_data = Dataset(
        root=str(data_path),
        transform=bundle["preprocess"],
        target_transform=bundle["target_transform"],
        dataset_name=dataset_name,
    )
    loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    model = bundle["model"]
    layer_transforms = bundle["layer_transforms"]
    cross_attn = bundle["cross_attn"]
    features_list = bundle["features_list"]
    image_size = bundle["image_size"]

    by_cat: dict[str, dict[str, list]] = defaultdict(
        lambda: {"masks": [], "maps": [], "labels": [], "scores": [], "sample_ids": []}
    )

    for items in tqdm(loader, desc="infer"):
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"].clone()
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        label = int(items["anomaly"].item())

        with torch.no_grad():
            vision_output = model.encode_image(image, features_list)
            anomaly_features = vision_output["anomaly_features"]
            normal_features = vision_output["normal_features"]
            patch_tokens = vision_output["patch_tokens"]
            patch_start_idx = vision_output["patch_start_idx"]

            patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
            if cross_attn is not None:
                adapted_list = cross_attn(
                    anomaly_features, normal_features, patch_features_list, features_list
                )
                anomaly_features_list = [a["anomaly"] for a in adapted_list]
                normal_features_list = [a["normal"] for a in adapted_list]
            else:
                anomaly_features_list = [anomaly_features] * len(patch_tokens)
                normal_features_list = [normal_features] * len(patch_tokens)

            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_tokens):
                anomaly_feat_norm = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
                normal_feat_norm = F.normalize(normal_features_list[idx], dim=1, eps=1e-8)
                transform_key = f"layer_{features_list[idx]}"
                if transform_key in layer_transforms:
                    bsz, num_tokens, dim = patch_feature.shape
                    patch_feature = layer_transforms[transform_key](
                        patch_feature.view(-1, dim)
                    ).view(bsz, num_tokens, dim)
                anomaly_map = generate_anomaly_map_from_tokens(
                    anomaly_feat_norm,
                    normal_feat_norm,
                    patch_feature[:, patch_start_idx:, :],
                    image_size,
                )
                anomaly_map_list.append(anomaly_map)

            final_anomaly_map = torch.stack(anomaly_map_list).sum(dim=0).cpu()
            filtered = gaussian_filter(final_anomaly_map[0].numpy(), sigma=sigma)
            score = float(np.max(filtered))

        by_cat[cls_name]["masks"].append(gt_mask.squeeze().cpu().numpy().astype(np.float32))
        by_cat[cls_name]["maps"].append(filtered.astype(np.float32))
        by_cat[cls_name]["labels"].append(label)
        by_cat[cls_name]["scores"].append(score)
        by_cat[cls_name]["sample_ids"].append(str(items["img_path"][0]))

    return by_cat


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose VisualAD baseline AUPRO")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", type=str, default=None)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Save float16 masks/maps npz for audit (large)",
    )
    args = parser.parse_args()

    ckpt_sha = sha256_file(args.checkpoint)
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint_sha256: {ckpt_sha}")
    print(f"data_path: {args.data_path}")
    print(f"dataset: {args.dataset}")
    print(f"output_dir: {args.output_dir}")
    print(f"device: {args.device}")
    print(f"sigma: {args.sigma}")
    print("train: SKIPPED")
    if args.expected_sha256 and args.expected_sha256 != ckpt_sha:
        raise SystemExit(
            f"checkpoint hash mismatch: expected {args.expected_sha256}, got {ckpt_sha}"
        )
    if args.dry_run:
        print("dry-run ok")
        return 0

    refuse_existing_run(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    bundle = load_model(args.checkpoint, device)
    by_cat = run_inference(
        bundle=bundle,
        data_path=args.data_path,
        dataset_name=args.dataset,
        device=device,
        sigma=args.sigma,
    )

    per_category: dict[str, Any] = {}
    all_masks: list[np.ndarray] = []
    all_maps: list[np.ndarray] = []
    all_labels: list[int] = []
    all_scores: list[float] = []

    for cat in sorted(by_cat.keys()):
        print(f"[aupro] category={cat} n={len(by_cat[cat]['masks'])}", flush=True)
        masks = np.stack(by_cat[cat]["masks"]).astype(np.float64)
        maps = np.stack(by_cat[cat]["maps"]).astype(np.float64)
        diag = aupro_diagnostics(masks, maps, max_fpr=0.3, steps=200)
        print(
            f"[aupro] category={cat} cal={diag['cal_pro_score']:.4f} "
            f"safe={diag['safe_aupro']:.4f} regions={diag['n_connected_regions']}",
            flush=True,
        )
        per_category[cat] = {
            **diag,
            "sample_ids": by_cat[cat]["sample_ids"],
        }
        all_masks.extend(by_cat[cat]["masks"])
        all_maps.extend(by_cat[cat]["maps"])
        all_labels.extend(by_cat[cat]["labels"])
        all_scores.extend(by_cat[cat]["scores"])

    masks_a = np.stack(all_masks).astype(np.float64)
    maps_a = np.stack(all_maps).astype(np.float64)
    labels_a = np.asarray(all_labels, dtype=np.float64)
    scores_a = np.asarray(all_scores, dtype=np.float64)

    # Official VisualAD table uses per-category AUPRO then macro-mean.
    macro_cal = float(
        np.mean([per_category[c]["cal_pro_score"] for c in per_category])
    )
    macro_safe = float(
        np.mean([per_category[c]["safe_aupro"] for c in per_category])
    )
    print("[aupro] computing global pooled diagnostics", flush=True)
    global_diag = aupro_diagnostics(masks_a, maps_a, max_fpr=0.3, steps=200)
    print("[aupro] computing paper metrics", flush=True)
    paper = compute_paper_metrics(
        image_labels=labels_a,
        image_scores=scores_a,
        masks=masks_a,
        anomaly_maps=maps_a,
        aupro_max_fpr=0.3,
        aupro_steps=200,
        boundary_enabled=False,
    )

    if args.save_arrays:
        np.savez_compressed(
            args.output_dir / "maps_masks_f16.npz",
            masks=masks_a.astype(np.float16),
            anomaly_maps=maps_a.astype(np.float16),
            labels=labels_a.astype(np.int16),
            scores=scores_a.astype(np.float32),
        )

    report = {
        "schema_version": 1,
        "status": "completed",
        "task": "aupro_recompute",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": ckpt_sha,
        "dataset": args.dataset,
        "data_path": str(args.data_path.resolve()),
        "backbone": bundle["backbone"],
        "candidate_layers": list(bundle["features_list"]),
        "image_size": bundle["image_size"],
        "sigma": args.sigma,
        "seed": args.seed,
        "device": args.device,
        "protocol": {
            "max_fpr": 0.3,
            "steps": 200,
            "aggregation_note": (
                "VisualAD compute_metrics reports per-category AUPRO then macro-mean; "
                "PaperMetrics.pixel_aupro is globally pooled across all images."
            ),
            "equal_fusion": "sum of layer maps then gaussian sigma=4 (test.py)",
            "no_target_tuning": True,
        },
        "n_samples": int(len(all_masks)),
        "n_anomalous_samples": int(sum(all_labels)),
        "per_category": {
            cat: {k: v for k, v in vals.items() if k != "sample_ids"}
            for cat, vals in per_category.items()
        },
        "macro_mean_cal_pro_score": macro_cal,
        "macro_mean_safe_aupro": macro_safe,
        "global_pooled": global_diag,
        "paper_metrics": paper.as_dict(),
        "original_baseline_reported_pixel_aupro": 0.0,
        "reference_visa_clip_aupro": 91.0,
    }
    atomic_write_json(args.output_dir / "aupro_diagnostics.json", report)
    atomic_write_json(
        args.output_dir / "metrics_paper.json",
        {
            **paper.as_dict(),
            "pixel_aupro_macro_cal_pro_score": macro_cal,
            "pixel_aupro_macro_safe_aupro": macro_safe,
            "pixel_aupro_global_cal_pro_score": global_diag["cal_pro_score"],
            "pixel_aupro_global_safe_aupro": global_diag["safe_aupro"],
        },
    )
    # Keep sample id lists separately for audit without bloating main summary.
    atomic_write_json(
        args.output_dir / "sample_index.json",
        {cat: per_category[cat]["sample_ids"] for cat in per_category},
    )

    print(json.dumps({
        "n_samples": report["n_samples"],
        "n_anomalous": report["n_anomalous_samples"],
        "macro_mean_cal_pro_score": macro_cal,
        "macro_mean_safe_aupro": macro_safe,
        "global_safe_aupro": global_diag["safe_aupro"],
        "paper_pixel_aupro": paper.pixel_aupro,
        "paper_image_auroc": paper.image_auroc,
        "paper_pixel_auroc": paper.pixel_auroc,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
