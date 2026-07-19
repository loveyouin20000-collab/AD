from __future__ import annotations

import numpy as np
import pytest

from rad.errors import MetricComputationError
from rad.evaluation.paper_metrics import (
    PaperMetrics,
    compute_paper_metrics,
    safe_aupro,
    tolerance_boundary_f_score,
)


def _perfect_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # One normal + one anomalous; maps match masks exactly on a larger canvas
    # so boundary extraction is well-defined.
    labels = np.array([0, 1], dtype=np.int64)
    scores = np.array([0.1, 0.9], dtype=np.float64)
    masks = np.zeros((2, 16, 16), dtype=np.float64)
    masks[1, 6:10, 6:10] = 1.0
    maps = masks.copy()
    return labels, scores, masks, maps


def test_paper_metrics_perfect_prediction() -> None:
    labels, scores, masks, maps = _perfect_pair()
    metrics = compute_paper_metrics(
        image_labels=labels,
        image_scores=scores,
        masks=masks,
        anomaly_maps=maps,
    )
    assert isinstance(metrics, PaperMetrics)
    assert metrics.image_auroc == pytest.approx(1.0)
    assert metrics.image_ap == pytest.approx(1.0)
    assert metrics.pixel_auroc == pytest.approx(1.0)
    assert metrics.pixel_ap == pytest.approx(1.0)
    assert metrics.pixel_f1_max == pytest.approx(1.0)
    assert metrics.boundary_f_score is not None
    assert metrics.boundary_f_score == pytest.approx(1.0, abs=1e-6)


def test_paper_metrics_inverted_prediction() -> None:
    labels = np.array([0, 1], dtype=np.int64)
    scores = np.array([0.9, 0.1], dtype=np.float64)
    masks = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]], dtype=np.float64)
    maps = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]], dtype=np.float64)
    metrics = compute_paper_metrics(
        image_labels=labels,
        image_scores=scores,
        masks=masks,
        anomaly_maps=maps,
    )
    assert metrics.image_auroc == pytest.approx(0.0)
    assert metrics.pixel_auroc == pytest.approx(0.0)


def test_pixel_ap_is_dataset_level_and_includes_normal_pixels() -> None:
    # Normal image contributes high false-positive scores into the global pool.
    labels = np.array([0, 1], dtype=np.int64)
    scores = np.array([0.2, 0.8], dtype=np.float64)
    masks = np.array(
        [
            [[0, 0], [0, 0]],
            [[1, 0], [0, 0]],
        ],
        dtype=np.float64,
    )
    maps = np.array(
        [
            [[0.95, 0.95], [0.95, 0.95]],  # normal false positives outrank anomaly
            [[0.90, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    metrics = compute_paper_metrics(
        image_labels=labels,
        image_scores=scores,
        masks=masks,
        anomaly_maps=maps,
    )
    # Dataset-level AP must include normal-image pixels and stay < 1.
    assert metrics.pixel_ap < 1.0
    # Illegal per-image protocol would assign normal AP=1.0 then average → inflate.
    assert metrics.pixel_ap < 0.5


def test_known_image_f1_max() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    masks = np.zeros((4, 2, 2), dtype=np.float64)
    masks[2, 0, 0] = 1.0
    masks[3, 0, 0] = 1.0
    maps = masks.copy()
    metrics = compute_paper_metrics(
        image_labels=labels,
        image_scores=scores,
        masks=masks,
        anomaly_maps=maps,
    )
    # Separable scores → perfect image F1-max.
    assert metrics.image_f1_max == pytest.approx(1.0)


def test_safe_aupro_constant_map_is_zero() -> None:
    masks = np.array([[[1, 0], [0, 0]]], dtype=np.float64)
    maps = np.ones_like(masks)
    assert safe_aupro(masks, maps, max_fpr=0.3, steps=200) == 0.0


def test_safe_aupro_no_anomalous_regions_is_zero() -> None:
    masks = np.zeros((2, 2, 2), dtype=np.float64)
    maps = np.random.default_rng(0).random(masks.shape)
    assert safe_aupro(masks, maps, max_fpr=0.3, steps=50) == 0.0


def test_safe_aupro_rejects_nonfinite() -> None:
    masks = np.array([[[1, 0], [0, 0]]], dtype=np.float64)
    maps = np.array([[[np.nan, 0.1], [0.2, 0.3]]], dtype=np.float64)
    with pytest.raises(MetricComputationError, match="nonfinite|NaN|inf"):
        safe_aupro(masks, maps, max_fpr=0.3, steps=50)


def test_boundary_f_score_with_tolerance() -> None:
    # GT edge around a 1-pixel anomaly; prediction shifted by 1 pixel should
    # still match under generous tolerance on small maps.
    mask = np.zeros((16, 16), dtype=np.float64)
    mask[8, 8] = 1.0
    pred = np.zeros((16, 16), dtype=np.float64)
    pred[8, 9] = 1.0  # one-pixel shift
    # Exact match without tolerance is weak; with ratio large enough, F rises.
    f_tol = tolerance_boundary_f_score(
        pred,
        mask,
        threshold=0.5,
        tolerance_ratio=0.2,
    )
    f_exact = tolerance_boundary_f_score(
        pred,
        mask,
        threshold=0.5,
        tolerance_ratio=0.0,
        min_radius=0,
    )
    assert f_tol > f_exact
    assert 0.0 <= f_tol <= 1.0


def test_compute_paper_metrics_rejects_nonfinite_scores() -> None:
    labels = np.array([0, 1], dtype=np.int64)
    scores = np.array([0.1, np.inf], dtype=np.float64)
    masks = np.zeros((2, 2, 2), dtype=np.float64)
    masks[1, 0, 0] = 1.0
    maps = masks.copy()
    with pytest.raises(MetricComputationError):
        compute_paper_metrics(
            image_labels=labels,
            image_scores=scores,
            masks=masks,
            anomaly_maps=maps,
        )


def test_zero_shot_transfer_metrics_use_paper_metrics_not_proxies() -> None:
    from rad.evaluation import zero_shot

    source = zero_shot.__file__
    text = open(source, encoding="utf-8").read()
    # Reporting path must call compute_paper_metrics / PaperMetrics.
    assert "compute_paper_metrics" in text
    # compute_transfer_metrics body must not average per-image AP proxies.
    assert "pro_score_proxy" not in text.split("def compute_transfer_metrics")[1].split(
        "def compute_stratified_metrics"
    )[0]
