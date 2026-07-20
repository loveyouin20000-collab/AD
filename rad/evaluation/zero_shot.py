from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rad.artifacts import assert_json_artifact_eligible_for_evaluation
from rad.errors import ArtifactIntegrityError
from rad.models.policy import PolicyProfile


class TargetAccessError(RuntimeError):
    """Raised when target-domain samples are touched during source-only calibration."""


@dataclass
class CalibrationAccessGuard:
    source_dataset: str
    target_datasets: tuple[str, ...]

    def _mentions_target(self, text: str) -> str | None:
        lowered = text.replace("\\", "/").lower()
        source = self.source_dataset.lower()
        for tgt in self.target_datasets:
            name = str(tgt).lower()
            if not name or name == source:
                continue
            parts = [p for p in lowered.split("/") if p]
            if name in parts:
                return name
            if f"{name}_" in lowered or f"_{name}" in lowered or f"/{name}/" in f"/{lowered}/":
                return name
        return None

    def check_path(self, path: str | Path) -> None:
        hit = self._mentions_target(str(path))
        if hit is not None:
            raise TargetAccessError(
                f"target dataset {hit!r} accessed during source-only calibration: {path}"
            )

    def check_sample_id(self, sample_id: str) -> None:
        hit = self._mentions_target(sample_id)
        if hit is not None:
            raise TargetAccessError(
                f"target dataset {hit!r} sample accessed during calibration: {sample_id}"
            )


@contextmanager
def forbid_target_access_during_calibration(
    *,
    source_dataset: str,
    target_datasets: Sequence[str],
) -> Iterator[CalibrationAccessGuard]:
    guard = CalibrationAccessGuard(
        source_dataset=str(source_dataset),
        target_datasets=tuple(str(t) for t in target_datasets),
    )
    yield guard


def load_frozen_policy_profile(
    path: Path | str,
    name: str,
) -> tuple[PolicyProfile, str]:
    policy_path = Path(path)
    if not policy_path.is_file():
        raise ArtifactIntegrityError(f"missing calibration policy: {policy_path}")
    raw = json.loads(policy_path.read_text(encoding="utf-8"))
    payload = json.dumps(raw["profiles"][name], sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    p = raw["profiles"][name]
    profile = PolicyProfile(
        name=str(p["name"]),
        gain_threshold=float(p["gain_threshold"]),
        kappa=float(p["kappa"]),
        map_uncertainty_threshold=float(p["map_uncertainty_threshold"]),
        image_confidence_margin=float(p["image_confidence_margin"]),
        stability_threshold=float(p["stability_threshold"]),
        require_map_uncertainty=bool(p.get("require_map_uncertainty", False)),
        require_image_confidence=bool(p.get("require_image_confidence", False)),
        require_stability=bool(p.get("require_stability", False)),
    )
    return profile, digest


def assert_policy_unchanged(path: Path | str, name: str, expected_digest: str) -> None:
    _, digest = load_frozen_policy_profile(path, name)
    if digest != expected_digest:
        raise ValueError(
            f"source-calibrated policy {name!r} changed on disk "
            f"(expected {expected_digest[:12]}..., got {digest[:12]}...)"
        )


def assert_policy_eligible_for_evaluation(path: Path | str) -> None:
    """Reject test fixtures and diagnostics from real zero-shot evaluation."""
    assert_json_artifact_eligible_for_evaluation(path, kind="calibration policy")


def pixel_average_precision(pred: np.ndarray, mask: np.ndarray) -> float:
    """Deprecated per-map helper retained for legacy tooling.

    Paper tables must use ``rad.evaluation.paper_metrics.compute_paper_metrics``
    (dataset-level flattening). Do not invent AP=1.0 for empty masks in new code.
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(mask, dtype=np.float64).reshape(-1)
    y_score = np.asarray(pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    if float(y_true.max()) < 0.5:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _binarize(pred: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (np.asarray(pred, dtype=np.float64) >= thr).astype(np.float64)


def boundary_f_score(pred: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    """Legacy exact-edge boundary F-score.

    Prefer ``tolerance_boundary_f_score`` / ``PaperMetrics`` for paper reporting.
    """
    from rad.evaluation.paper_metrics import tolerance_boundary_f_score

    return tolerance_boundary_f_score(
        np.asarray(pred, dtype=np.float64),
        np.asarray(mask, dtype=np.float64),
        threshold=thr,
        tolerance_ratio=0.0,
        min_radius=0,
    )


def pro_score_proxy(pred: np.ndarray, mask: np.ndarray) -> float:
    """Deprecated single-threshold PRO proxy for legacy tooling only.

    Paper reporting must use ``PaperMetrics.pixel_aupro`` via ``safe_aupro``.
    """
    from scipy import ndimage

    m = (np.asarray(mask, dtype=np.float64) > 0.5).astype(np.uint8)
    p = np.asarray(pred, dtype=np.float64)
    if m.sum() == 0:
        return 0.0
    labeled, n = ndimage.label(m)
    if n == 0:
        return 0.0
    thr = float(np.quantile(p, 0.9))
    pb = (p >= thr).astype(np.float64)
    ious: list[float] = []
    for i in range(1, n + 1):
        region = labeled == i
        inter = float((pb[region] > 0).sum())
        union = float(region.sum())
        ious.append(inter / max(union, 1.0))
    return float(np.mean(ious))


def compute_transfer_metrics(
    *,
    adaptive_maps: np.ndarray,
    full_depth_maps: np.ndarray,
    masks: np.ndarray,
    selected_depths: np.ndarray,
    residual_gains: np.ndarray,
    image_labels: np.ndarray,
    epsilon: float,
    full_depth: int,
    image_scores_adaptive: np.ndarray | None = None,
    image_scores_full: np.ndarray | None = None,
) -> dict[str, Any]:
    from rad.evaluation.paper_metrics import compute_paper_metrics
    from rad.evaluation.policy_metrics import (
        expected_depth_and_histogram,
        false_safe_exit_rate,
    )

    labels = np.asarray(image_labels, dtype=np.float64).reshape(-1)
    if image_scores_adaptive is None:
        image_scores_adaptive = np.max(
            np.asarray(adaptive_maps, dtype=np.float64).reshape(len(adaptive_maps), -1),
            axis=1,
        )
    if image_scores_full is None:
        image_scores_full = np.max(
            np.asarray(full_depth_maps, dtype=np.float64).reshape(len(full_depth_maps), -1),
            axis=1,
        )

    adaptive = compute_paper_metrics(
        image_labels=labels,
        image_scores=image_scores_adaptive,
        masks=masks,
        anomaly_maps=adaptive_maps,
    )
    full = compute_paper_metrics(
        image_labels=labels,
        image_scores=image_scores_full,
        masks=masks,
        anomaly_maps=full_depth_maps,
    )
    expected, hist = expected_depth_and_histogram(selected_depths)
    fse = false_safe_exit_rate(
        selected_depths=selected_depths,
        residual_gains=residual_gains,
        epsilon=epsilon,
        full_depth=full_depth,
    )
    return {
        "n": int(len(selected_depths)),
        "adaptive": adaptive.as_dict(),
        "full": full.as_dict(),
        "pixel_ap_adaptive": adaptive.pixel_ap,
        "pixel_ap_full": full.pixel_ap,
        "pixel_ap_drop": float(full.pixel_ap - adaptive.pixel_ap),
        "pixel_aupro_adaptive": adaptive.pixel_aupro,
        "pixel_aupro_full": full.pixel_aupro,
        "pixel_aupro_drop": float(full.pixel_aupro - adaptive.pixel_aupro),
        "boundary_f_score_adaptive": adaptive.boundary_f_score,
        "boundary_f_score_full": full.boundary_f_score,
        "boundary_f_score_drop": float(
            (full.boundary_f_score or 0.0) - (adaptive.boundary_f_score or 0.0)
        ),
        "false_safe_exit_rate": fse,
        "expected_depth": expected,
        "exit_histogram": hist,
        "anomalous_fraction": float(np.mean(labels > 0.5)),
    }


def _anomaly_area(mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=np.float64)
    return float(m.mean())


def _contrast_proxy(image: np.ndarray, mask: np.ndarray) -> float:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=0)
    m = np.asarray(mask, dtype=np.float64) > 0.5
    if m.any() and (~m).any():
        return float(abs(img[m].mean() - img[~m].mean()))
    return float(img.std())


def boundary_complexity(mask: np.ndarray) -> float:
    from scipy import ndimage

    m = (np.asarray(mask, dtype=np.float64) > 0.5).astype(np.float64)
    if m.sum() == 0:
        return 0.0
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    edge = np.hypot(
        ndimage.convolve(m, sx, mode="nearest"),
        ndimage.convolve(m, sx.T, mode="nearest"),
    )
    return float((edge > 1e-6).mean())


def compute_stratified_metrics(
    *,
    adaptive_maps: np.ndarray,
    full_depth_maps: np.ndarray,
    masks: np.ndarray,
    selected_depths: np.ndarray,
    residual_gains: np.ndarray,
    image_labels: np.ndarray,
    images: np.ndarray,
    epsilon: float,
    full_depth: int,
) -> dict[str, Any]:
    labels = np.asarray(image_labels).reshape(-1)
    areas = np.array([_anomaly_area(masks[i]) for i in range(len(masks))])
    contrasts = np.array([_contrast_proxy(images[i], masks[i]) for i in range(len(masks))])
    complexities = np.array([boundary_complexity(masks[i]) for i in range(len(masks))])

    def subset(mask: np.ndarray) -> dict[str, Any]:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return {"n": 0}
        return compute_transfer_metrics(
            adaptive_maps=adaptive_maps[idx],
            full_depth_maps=full_depth_maps[idx],
            masks=masks[idx],
            selected_depths=selected_depths[idx],
            residual_gains=residual_gains[idx],
            image_labels=labels[idx],
            epsilon=epsilon,
            full_depth=full_depth,
        )

    area_med = float(np.median(areas[labels > 0.5])) if np.any(labels > 0.5) else 0.0
    contrast_med = float(np.median(contrasts))
    complex_med = float(np.median(complexities))

    return {
        "normal": subset(labels < 0.5),
        "anomalous": subset(labels > 0.5),
        "anomaly_area": {
            "low": subset((labels > 0.5) & (areas <= area_med)),
            "high": subset((labels > 0.5) & (areas > area_med)),
        },
        "contrast": {
            "low": subset(contrasts <= contrast_med),
            "high": subset(contrasts > contrast_med),
        },
        "boundary_complexity": {
            "low": subset(complexities <= complex_med),
            "high": subset(complexities > complex_med),
        },
    }
