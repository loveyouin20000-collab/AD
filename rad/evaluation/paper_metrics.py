"""Authoritative dataset-level paper metrics for RAD evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from skimage import measure
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from rad.errors import MetricComputationError

_EPS = 1e-8


@dataclass(frozen=True)
class PaperMetrics:
    image_auroc: float
    image_ap: float
    image_f1_max: float
    pixel_auroc: float
    pixel_ap: float
    pixel_f1_max: float
    pixel_aupro: float
    boundary_f_score: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_finite(name: str, array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise MetricComputationError(f"nonfinite values in {name}")
    return arr


def _f1_max(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        raise MetricComputationError("empty array for F1-max")
    if float(y_true.max()) < 0.5 or float(y_true.min()) >= 0.5:
        # Only one class present: F1-max is undefined for ranking; report 0.
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2.0 * precision * recall / (precision + recall + _EPS)
    value = float(np.max(f1))
    if not np.isfinite(value):
        raise MetricComputationError("nonfinite F1-max")
    return value


def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        raise MetricComputationError("empty array for AUROC")
    if len(np.unique(y_true)) < 2:
        return 0.0
    value = float(roc_auc_score(y_true, y_score))
    if not np.isfinite(value):
        raise MetricComputationError("nonfinite AUROC")
    return value


def _binary_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        raise MetricComputationError("empty array for AP")
    if float(y_true.max()) < 0.5:
        # No positive labels in the flattened set: do NOT invent AP=1.0.
        return 0.0
    value = float(average_precision_score(y_true, y_score))
    if not np.isfinite(value):
        raise MetricComputationError("nonfinite AP")
    return value


def safe_aupro(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    *,
    max_fpr: float = 0.3,
    steps: int = 200,
) -> float:
    """VisualAD-compatible PRO/AUPRO with fail-closed safeguards."""
    masks_arr = _require_finite("masks", masks)
    amaps = _require_finite("anomaly_maps", anomaly_maps)
    if masks_arr.shape != amaps.shape:
        raise MetricComputationError(
            f"AUPRO shape mismatch: masks {masks_arr.shape} vs maps {amaps.shape}"
        )
    if masks_arr.ndim != 3:
        raise MetricComputationError(f"AUPRO expects (N,H,W), got {masks_arr.shape}")
    if steps < 2:
        return 0.0

    gt = (masks_arr > 0.5).astype(np.float64)
    if float(gt.sum()) <= 0.0:
        return 0.0

    min_th = float(np.min(amaps))
    max_th = float(np.max(amaps))
    if not np.isfinite(min_th) or not np.isfinite(max_th):
        raise MetricComputationError("nonfinite AUPRO thresholds")
    if max_th <= min_th:
        # Constant anomaly map: no meaningful FPR span.
        return 0.0

    delta = (max_th - min_th) / float(steps)
    pros: list[float] = []
    fprs: list[float] = []
    binary = np.zeros_like(amaps, dtype=bool)
    for th in np.arange(min_th, max_th, delta):
        binary[:] = amaps > th
        region_pros: list[float] = []
        for i in range(len(amaps)):
            mask = gt[i]
            if float(mask.sum()) == 0.0:
                continue
            labeled = measure.label(mask.astype(np.uint8), connectivity=2)
            for region in measure.regionprops(labeled):
                coords = region.coords
                tp = float(np.sum(binary[i][coords[:, 0], coords[:, 1]]))
                region_pros.append(tp / float(region.area))
        avg_pro = float(np.mean(region_pros)) if region_pros else 0.0
        if not np.isfinite(avg_pro):
            raise MetricComputationError("nonfinite PRO intermediate value")
        pros.append(avg_pro)

        fp = float(np.sum(binary * (1.0 - gt)))
        total_neg = float(np.sum(1.0 - gt))
        fpr = fp / total_neg if total_neg > 0.0 else 0.0
        if not np.isfinite(fpr):
            raise MetricComputationError("nonfinite FPR intermediate value")
        fprs.append(fpr)

    pros_a = np.asarray(pros, dtype=np.float64)
    fprs_a = np.asarray(fprs, dtype=np.float64)
    valid = fprs_a <= float(max_fpr)
    if not np.any(valid):
        return 0.0
    fpr_valid = fprs_a[valid]
    pro_valid = pros_a[valid]
    if fpr_valid.size < 2:
        return 0.0
    span = float(fpr_valid.max() - fpr_valid.min())
    if span <= 0.0:
        return 0.0
    fpr_norm = (fpr_valid - fpr_valid.min()) / span
    # Trapezoidal AUC on normalized FPR axis (VisualAD cal_pro_score).
    order = np.argsort(fpr_norm)
    fpr_sorted = fpr_norm[order]
    pro_sorted = pro_valid[order]
    value = float(np.trapz(pro_sorted, fpr_sorted))
    if not np.isfinite(value):
        raise MetricComputationError("nonfinite AUPRO")
    return max(0.0, min(1.0, value))


def _extract_boundary(binary: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    edge = np.hypot(
        ndimage.convolve(binary, sx, mode="nearest"),
        ndimage.convolve(binary, sx.T, mode="nearest"),
    )
    return edge > 1e-6


def tolerance_boundary_f_score(
    pred: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    tolerance_ratio: float = 0.005,
    min_radius: int | None = 1,
) -> float:
    """Boundary F-score with morphological tolerance radius."""
    from scipy import ndimage

    pred_a = _require_finite("pred", pred)
    mask_a = _require_finite("mask", mask)
    if pred_a.shape != mask_a.shape:
        raise MetricComputationError("boundary F-score shape mismatch")
    if pred_a.ndim != 2:
        raise MetricComputationError("boundary F-score expects 2D maps")

    h, w = pred_a.shape
    diag = float(np.hypot(h, w))
    if min_radius is None:
        radius = max(1, int(round(float(tolerance_ratio) * diag)))
    elif float(tolerance_ratio) <= 0.0 and int(min_radius) == 0:
        radius = 0
    else:
        radius = max(int(min_radius), int(round(float(tolerance_ratio) * diag)))

    pred_bin = (pred_a >= float(threshold)).astype(np.float64)
    mask_bin = (mask_a > 0.5).astype(np.float64)
    pred_b = _extract_boundary(pred_bin)
    mask_b = _extract_boundary(mask_bin)

    if radius > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        pred_d = ndimage.binary_dilation(pred_b, structure=struct, iterations=radius)
        mask_d = ndimage.binary_dilation(mask_b, structure=struct, iterations=radius)
    else:
        pred_d = pred_b
        mask_d = mask_b

    # Match predicted boundary to dilated GT, and GT boundary to dilated prediction.
    tp = float(np.logical_and(pred_b, mask_d).sum())
    fp = float(np.logical_and(pred_b, np.logical_not(mask_d)).sum())
    fn = float(np.logical_and(mask_b, np.logical_not(pred_d)).sum())
    prec = tp / (tp + fp + _EPS)
    rec = tp / (tp + fn + _EPS)
    value = float(2.0 * prec * rec / (prec + rec + _EPS))
    if not np.isfinite(value):
        raise MetricComputationError("nonfinite boundary F-score")
    return value


def _mean_boundary_f_score(
    anomaly_maps: np.ndarray,
    masks: np.ndarray,
    *,
    tolerance_ratio: float,
    threshold: float,
) -> float:
    vals: list[float] = []
    for i in range(len(anomaly_maps)):
        if float(np.asarray(masks[i], dtype=np.float64).sum()) <= 0.0:
            # No GT anomaly region → boundary F undefined; exclude from mean.
            continue
        vals.append(
            tolerance_boundary_f_score(
                anomaly_maps[i],
                masks[i],
                threshold=threshold,
                tolerance_ratio=tolerance_ratio,
            )
        )
    if not vals:
        return 0.0
    value = float(np.mean(vals))
    if not np.isfinite(value):
        raise MetricComputationError("nonfinite mean boundary F-score")
    return value


def compute_paper_metrics(
    *,
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    aupro_max_fpr: float = 0.3,
    aupro_steps: int = 200,
    boundary_enabled: bool = True,
    boundary_tolerance_ratio: float = 0.005,
    boundary_threshold: float = 0.5,
) -> PaperMetrics:
    """Compute dataset-level paper metrics (no per-image pixel-AP averaging)."""
    labels = _require_finite("image_labels", image_labels).reshape(-1)
    scores = _require_finite("image_scores", image_scores).reshape(-1)
    masks_a = _require_finite("masks", masks)
    maps_a = _require_finite("anomaly_maps", anomaly_maps)
    if masks_a.shape != maps_a.shape:
        raise MetricComputationError(
            f"masks/maps shape mismatch: {masks_a.shape} vs {maps_a.shape}"
        )
    if labels.shape[0] != masks_a.shape[0] or scores.shape[0] != masks_a.shape[0]:
        raise MetricComputationError("image labels/scores length mismatch vs maps")

    gt_px = masks_a.reshape(-1)
    pr_px = maps_a.reshape(-1)
    # Keep normal-image pixels in the global flattened arrays intentionally.

    metrics = PaperMetrics(
        image_auroc=_binary_auroc((labels > 0.5).astype(np.float64), scores),
        image_ap=_binary_ap((labels > 0.5).astype(np.float64), scores),
        image_f1_max=_f1_max((labels > 0.5).astype(np.float64), scores),
        pixel_auroc=_binary_auroc((gt_px > 0.5).astype(np.float64), pr_px),
        pixel_ap=_binary_ap((gt_px > 0.5).astype(np.float64), pr_px),
        pixel_f1_max=_f1_max((gt_px > 0.5).astype(np.float64), pr_px),
        pixel_aupro=safe_aupro(
            masks_a,
            maps_a,
            max_fpr=aupro_max_fpr,
            steps=aupro_steps,
        ),
        boundary_f_score=(
            _mean_boundary_f_score(
                maps_a,
                masks_a,
                tolerance_ratio=boundary_tolerance_ratio,
                threshold=boundary_threshold,
            )
            if boundary_enabled
            else None
        ),
    )
    for key, value in metrics.as_dict().items():
        if value is None:
            continue
        if not np.isfinite(value):
            raise MetricComputationError(f"nonfinite paper metric: {key}")
        if not (0.0 <= float(value) <= 1.0 + 1e-6):
            raise MetricComputationError(f"metric out of range: {key}={value}")
    return metrics
