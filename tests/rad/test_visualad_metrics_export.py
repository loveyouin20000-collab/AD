"""Tests for VisualAD compute_metrics AUPRO export (category-macro)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from rad.evaluation.paper_metrics import safe_aupro
from tests.rad.contracts.utils_metrics import load_metrics_module

_metrics = load_metrics_module()
cal_pro_score = _metrics.cal_pro_score
compute_metrics = _metrics.compute_metrics

AUPRO_MAX_FPR = 0.3
AUPRO_STEPS = 200


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _logger() -> tuple[logging.Logger, _ListHandler]:
    handler = _ListHandler()
    logger = logging.getLogger("test_visualad_metrics_export")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger, handler


def _nondegenerate_category(
    *,
    name: str = "toy",
    n: int = 4,
    h: int = 32,
    w: int = 32,
) -> dict:
    rng = np.random.default_rng(0)
    masks = np.zeros((n, h, w), dtype=np.float32)
    maps = rng.normal(loc=0.1, scale=0.05, size=(n, h, w)).astype(np.float32)
    labels = []
    scores = []
    for i, (r0, c0) in enumerate([(8, 8), (16, 20), (22, 10)], start=1):
        masks[i, r0 : r0 + 6, c0 : c0 + 6] = 1.0
        maps[i, r0 : r0 + 6, c0 : c0 + 6] += 1.5
        labels.append(1)
        scores.append(float(maps[i].max()))
    labels.insert(0, 0)
    scores.insert(0, float(maps[0].max()))
    return {
        "imgs_masks": [torch.from_numpy(m[None, ...]) for m in masks],
        "anomaly_maps": [torch.from_numpy(a[None, ...]) for a in maps],
        "gt_sp": labels,
        "pr_sp": scores,
        "_arrays": (masks.astype(np.float64), maps.astype(np.float64)),
        "_name": name,
    }


def test_compute_metrics_returns_finite_nonzero_aupro_on_valid_fixture() -> None:
    cat = _nondegenerate_category()
    logger, _ = _logger()
    export = compute_metrics({cat["_name"]: cat}, [cat["_name"]], logger)
    assert export is not None
    assert "pixel_aupro" in export
    value = float(export["pixel_aupro"])
    assert math_isfinite(value)
    assert value > 0.0
    assert 0.0 <= value <= 1.0


def test_compute_metrics_uses_official_aupro_protocol() -> None:
    cat = _nondegenerate_category()
    logger, _ = _logger()
    export = compute_metrics({cat["_name"]: cat}, [cat["_name"]], logger)
    assert export["pixel_aupro_max_fpr"] == pytest.approx(AUPRO_MAX_FPR)
    assert export["pixel_aupro_steps"] == AUPRO_STEPS
    assert export["pixel_aupro_aggregation"] == "category_macro"


def test_compute_metrics_aupro_agrees_with_safe_aupro() -> None:
    cat = _nondegenerate_category()
    masks, maps = cat["_arrays"]
    expected = float(
        cal_pro_score(masks, maps, max_step=AUPRO_STEPS, expect_fpr=AUPRO_MAX_FPR)
    )
    safe = float(safe_aupro(masks, maps, max_fpr=AUPRO_MAX_FPR, steps=AUPRO_STEPS))
    logger, _ = _logger()
    export = compute_metrics({cat["_name"]: cat}, [cat["_name"]], logger)
    assert expected > 0.0
    assert abs(expected - safe) < 1e-6
    assert float(export["pixel_aupro"]) == pytest.approx(expected, abs=1e-6)


def test_compute_metrics_constant_map_aupro_is_finite() -> None:
    n, h, w = 2, 16, 16
    masks = np.zeros((n, h, w), dtype=np.float32)
    masks[1, 4:8, 4:8] = 1.0
    maps = np.ones((n, h, w), dtype=np.float32)
    cat = {
        "imgs_masks": [torch.from_numpy(m[None, ...]) for m in masks],
        "anomaly_maps": [torch.from_numpy(a[None, ...]) for a in maps],
        "gt_sp": [0, 1],
        "pr_sp": [0.1, 0.9],
    }
    logger, _ = _logger()
    export = compute_metrics({"const": cat}, ["const"], logger)
    value = float(export["pixel_aupro"])
    assert math_isfinite(value)
    assert value == pytest.approx(0.0)


def test_compute_metrics_no_anomaly_aupro_is_finite() -> None:
    n, h, w = 2, 16, 16
    masks = np.zeros((n, h, w), dtype=np.float32)
    maps = np.linspace(0.0, 1.0, n * h * w, dtype=np.float32).reshape(n, h, w)
    cat = {
        "imgs_masks": [torch.from_numpy(m[None, ...]) for m in masks],
        "anomaly_maps": [torch.from_numpy(a[None, ...]) for a in maps],
        # Image labels keep two classes so sample AUROC remains defined;
        # pixel masks are all-normal for the AUPRO edge case under test.
        "gt_sp": [0, 1],
        "pr_sp": [0.1, 0.9],
    }
    logger, _ = _logger()
    export = compute_metrics({"normal": cat}, ["normal"], logger)
    value = float(export["pixel_aupro"])
    assert math_isfinite(value)
    assert value == pytest.approx(0.0)


def test_compute_metrics_does_not_replace_aupro_with_default_zero() -> None:
    import inspect
    import re

    from utils import metrics as metrics_mod

    source = inspect.getsource(metrics_mod.compute_metrics)
    assert "cal_pro_score(" in source
    assert re.search(r"^\s*pixel_aupro\s*=\s*0\s*$", source, flags=re.M) is None


def test_compute_metrics_category_macro_mean_of_two_classes() -> None:
    cat_a = _nondegenerate_category(name="a")
    cat_b = _nondegenerate_category(name="b", n=4)
    # Perturb class b so AUPRO differs slightly.
    masks_b, maps_b = cat_b["_arrays"]
    maps_b = maps_b.copy()
    maps_b[1] *= 0.5
    cat_b["anomaly_maps"] = [torch.from_numpy(a[None, ...].astype(np.float32)) for a in maps_b]
    cat_b["_arrays"] = (masks_b, maps_b.astype(np.float64))

    a_masks, a_maps = cat_a["_arrays"]
    b_masks, b_maps = cat_b["_arrays"]
    a_val = float(
        cal_pro_score(a_masks, a_maps, max_step=AUPRO_STEPS, expect_fpr=AUPRO_MAX_FPR)
    )
    b_val = float(
        cal_pro_score(b_masks, b_maps, max_step=AUPRO_STEPS, expect_fpr=AUPRO_MAX_FPR)
    )
    expected_macro = 0.5 * (a_val + b_val)

    logger, _ = _logger()
    export = compute_metrics(
        {"a": cat_a, "b": cat_b},
        ["a", "b"],
        logger,
    )
    assert export["pixel_aupro_aggregation"] == "category_macro"
    assert float(export["pixel_aupro"]) == pytest.approx(expected_macro, abs=1e-6)


def test_metrics_serialization_preserves_aupro_and_provenance(tmp_path: Path) -> None:
    cat = _nondegenerate_category()
    logger, _ = _logger()
    export = compute_metrics({cat["_name"]: cat}, [cat["_name"]], logger)
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["pixel_aupro"] == pytest.approx(export["pixel_aupro"])
    assert 0.0 <= float(loaded["pixel_aupro"]) <= 1.0
    assert loaded["pixel_aupro_aggregation"] == "category_macro"
    assert loaded["pixel_aupro_max_fpr"] == pytest.approx(0.3)
    assert loaded["pixel_aupro_steps"] == 200


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))
