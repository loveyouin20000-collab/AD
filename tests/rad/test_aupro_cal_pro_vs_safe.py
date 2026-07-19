"""Diagnostic: compare original cal_pro_score with PaperMetrics safe_aupro."""

from __future__ import annotations

import numpy as np

from rad.evaluation.paper_metrics import safe_aupro
from utils.metrics import cal_pro_score


def _nondegenerate_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic maps with clear anomalous regions and non-constant scores."""
    rng = np.random.default_rng(0)
    n, h, w = 4, 32, 32
    masks = np.zeros((n, h, w), dtype=np.float64)
    maps = rng.normal(loc=0.1, scale=0.05, size=(n, h, w)).astype(np.float64)
    # Image 0: normal
    # Images 1-3: compact anomalous blobs with elevated map scores
    for i, (r0, c0) in enumerate([(8, 8), (16, 20), (22, 10)], start=1):
        masks[i, r0 : r0 + 6, c0 : c0 + 6] = 1.0
        maps[i, r0 : r0 + 6, c0 : c0 + 6] += 1.5
    return masks, maps


def test_cal_pro_score_and_safe_aupro_agree_on_valid_fixture() -> None:
    masks, maps = _nondegenerate_fixture()
    original = float(cal_pro_score(masks, maps, max_step=200, expect_fpr=0.3))
    safe = float(safe_aupro(masks, maps, max_fpr=0.3, steps=200))
    assert original > 0.0
    assert safe > 0.0
    assert abs(original - safe) < 1e-6


def test_compute_metrics_calls_cal_pro_score_not_placeholder() -> None:
    """Regression: export path must call cal_pro_score (no hardcoded zero)."""
    import inspect
    import re

    from utils import metrics as metrics_mod

    source = inspect.getsource(metrics_mod.compute_metrics)
    assert "cal_pro_score(" in source
    assert re.search(r"^\s*pixel_aupro\s*=\s*0\s*$", source, flags=re.M) is None
