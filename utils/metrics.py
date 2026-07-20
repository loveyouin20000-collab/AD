
import numpy as np
from skimage import measure
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from tabulate import tabulate  # type: ignore[import-untyped]

# Official VisualAD AUPRO protocol (category-level then category-macro mean).
AUPRO_MAX_FPR = 0.3
AUPRO_STEPS = 200

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):  # type: ignore[no-untyped-def]
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = np.min(amaps), np.max(amaps)
    if (not np.isfinite(min_th)) or (not np.isfinite(max_th)) or max_th <= min_th:
        return 0.0
    delta = (max_th - min_th) / max_step
    if (not np.isfinite(delta)) or delta <= 0:
        return 0.0
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[:] = amaps > th
        pro = []
        for i in range(len(amaps)):
            mask = masks[i]
            amap = binary_amaps[i]
            if np.sum(mask) == 0:
                continue  # Skip if no anomaly region
            labeled_mask = measure.label(mask)
            regions = measure.regionprops(labeled_mask)
            for region in regions:
                coords = region.coords
                tp = np.sum(amap[coords[:, 0], coords[:, 1]])
                pro.append(tp / region.area)
        avg_pro = np.mean(pro) if pro else 0
        pros.append(avg_pro)
        fp = np.sum(binary_amaps * (1 - masks))
        total_fp_pixels = np.sum(1 - masks)
        fpr = fp / total_fp_pixels if total_fp_pixels != 0 else 0
        fprs.append(fpr)
    pros, fprs = np.array(pros), np.array(fprs)
    valid = fprs <= expect_fpr
    if not np.any(valid):
        return 0.0
    fpr_valid = fprs[valid]
    pro_valid = pros[valid]
    if len(fpr_valid) < 2:
        return 0.0
    span = fpr_valid.max() - fpr_valid.min()
    if span <= 0:
        return 0.0
    fpr_norm = (fpr_valid - fpr_valid.min()) / span
    return auc(fpr_norm, pro_valid)

def compute_metrics(results, obj_list, logger):  # type: ignore[no-untyped-def]
    """Compute per-category metrics and category-macro aggregates.

    Returns a JSON-serializable dict with metric values in ``[0, 1]`` plus AUPRO
    provenance fields. Still logs the VisualAD percentage table for humans.
    """
    table_ls = []
    per_category = []
    for obj in obj_list:
        obj_data = results[obj]
        # Pixel-level data
        gt_px, pr_px = [], []
        masks, amaps = [], []
        for mask_batch, anomaly_map_batch in zip(
            obj_data["imgs_masks"],
            obj_data["anomaly_maps"],
            strict=False,
        ):
            mask_np = mask_batch.squeeze().cpu().numpy()
            amap_np = anomaly_map_batch.squeeze().cpu().numpy()
            gt_px.extend(mask_np.flatten())
            pr_px.extend(amap_np.flatten())
            masks.append(mask_np)
            amaps.append(amap_np)
        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)
        masks = np.stack(masks)
        amaps = np.stack(amaps)
        # Sample-level data
        gt_sp = np.array(obj_data["gt_sp"])
        pr_sp = np.array(obj_data["pr_sp"])

        # Calculate metrics
        has_pixel_classes = bool(gt_px.size) and int(np.unique(gt_px).size) > 1
        pixel_auroc = roc_auc_score(gt_px, pr_px) if has_pixel_classes else 0
        pixel_ap = average_precision_score(gt_px, pr_px) if has_pixel_classes else 0
        pixel_aupro = (
            float(
                cal_pro_score(
                    masks,
                    amaps,
                    max_step=AUPRO_STEPS,
                    expect_fpr=AUPRO_MAX_FPR,
                )
            )
            if gt_px.size
            else 0.0
        )
        has_sample_classes = bool(gt_sp.size) and int(np.unique(gt_sp).size) > 1
        sample_auroc = roc_auc_score(gt_sp, pr_sp) if has_sample_classes else 0
        sample_ap = average_precision_score(gt_sp, pr_sp) if has_sample_classes else 0

        # F1 scores
        if has_pixel_classes:
            precisions, recalls, _ = precision_recall_curve(gt_px, pr_px)
            pixel_f1 = np.max(
                2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            )
        else:
            pixel_f1 = 0
        if has_sample_classes:
            precisions_sp, recalls_sp, _ = precision_recall_curve(gt_sp, pr_sp)
            sample_f1 = np.max(
                2
                * (precisions_sp * recalls_sp)
                / (precisions_sp + recalls_sp + 1e-8)
            )
        else:
            sample_f1 = 0

        per_category.append(
            {
                "class": obj,
                "pixel_auroc": float(pixel_auroc),
                "pixel_f1_max": float(pixel_f1),
                "pixel_ap": float(pixel_ap),
                "pixel_aupro": float(pixel_aupro),
                "image_auroc": float(sample_auroc),
                "image_f1_max": float(sample_f1),
                "image_ap": float(sample_ap),
            }
        )
        
        # Format table
        table = [
            obj,
            f"{pixel_auroc * 100:.1f}",
            f"{pixel_f1 * 100:.1f}",
            f"{pixel_ap * 100:.1f}",
            f"{pixel_aupro * 100:.1f}",
            f"{sample_auroc * 100:.1f}",
            f"{sample_f1 * 100:.1f}",
            f"{sample_ap * 100:.1f}"
        ]
        table_ls.append(table)
    
    # === New: Calculate and add mean row ===
    if len(table_ls) == 0:
        return None

    # Extract numeric part (skip first column class name)
    numeric_data = []
    for row in table_ls:
        numeric_values = [
            float(x.strip("%")) for x in row[1:]
        ]  # drop optional % then float
        numeric_data.append(numeric_values)

    # Calculate mean for each column (category-macro in percentage space, matching VisualAD)
    mean_values = np.array(numeric_data).mean(axis=0)
    mean_values_fmt = [f"{v:.1f}" for v in mean_values]

    # Add mean row
    mean_row = ['Mean'] + mean_values_fmt
    table_ls.append(mean_row)

    # === Generate table ===
    headers = ['Class', 'Pixel-AUROC', 'Pixel-F1', 'Pixel-AP', 'Pixel-AUPRO', 
              'Sample-AUROC', 'Sample-F1', 'Sample-AP']
    results_table = tabulate(table_ls, headers=headers, tablefmt='pipe')
    logger.info("\n%s", results_table)

    # Category-macro aggregates in [0, 1] for machine-readable export.
    macro = {
        "pixel_auroc": float(np.mean([c["pixel_auroc"] for c in per_category])),
        "pixel_f1_max": float(np.mean([c["pixel_f1_max"] for c in per_category])),
        "pixel_ap": float(np.mean([c["pixel_ap"] for c in per_category])),
        "pixel_aupro": float(np.mean([c["pixel_aupro"] for c in per_category])),
        "image_auroc": float(np.mean([c["image_auroc"] for c in per_category])),
        "image_f1_max": float(np.mean([c["image_f1_max"] for c in per_category])),
        "image_ap": float(np.mean([c["image_ap"] for c in per_category])),
    }
    return {
        **macro,
        "pixel_aupro_aggregation": "category_macro",
        "pixel_aupro_max_fpr": float(AUPRO_MAX_FPR),
        "pixel_aupro_steps": int(AUPRO_STEPS),
        "per_category": per_category,
    }
