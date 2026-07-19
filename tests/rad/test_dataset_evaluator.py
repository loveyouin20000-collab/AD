from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rad.data.adapters.mvtec import MVTecAdapter
from rad.data.adapters.preprocess import build_preprocess
from rad.data.adapters.visa import VisAAdapter
from rad.errors import MetricComputationError
from rad.evaluation.dataset_evaluator import (
    EvaluationOutputs,
    SamplePrediction,
    evaluate_dataset,
)
from rad.inference.adaptive_engine import AdaptiveResult


def _write_rgb(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _write_mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), value).save(path)


def _mvtec_fixture(root: Path) -> None:
    _write_rgb(root / "bottle" / "test" / "good" / "000.png")
    _write_rgb(root / "bottle" / "test" / "broken_large" / "000.png", (20, 20, 20))
    _write_mask(root / "bottle" / "ground_truth" / "broken_large" / "000_mask.png")
    _write_rgb(root / "cable" / "test" / "good" / "000.png", (30, 30, 30))


class FakeEngine:
    """Deterministic engine returning depth/map from a controllable table."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def infer(
        self,
        image: torch.Tensor,
        *,
        force_full_depth: bool = False,
        measure_timing: bool = False,
    ) -> AdaptiveResult:
        assert image.ndim == 4 and image.shape[0] == 1
        depth = 24 if force_full_depth else 12
        # Map intensity encodes force_full_depth for residual-gain tests.
        fill = 0.9 if force_full_depth else 0.4
        h = w = int(image.shape[-1])
        final_map = torch.full((1, h, w), fill, dtype=torch.float32, device=image.device)
        score = torch.tensor([fill], dtype=torch.float32, device=image.device)
        self.calls.append(
            {
                "force_full_depth": force_full_depth,
                "shape": tuple(image.shape),
                "device": str(image.device),
            }
        )
        return AdaptiveResult(
            final_map=final_map,
            image_score=score,
            selected_depth=depth,
            checkpoint_trace=[6, 12] if depth == 12 else [6, 12, 18, 24],
        )


def test_evaluate_dataset_shared_path_for_mvtec_and_visa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rad.evaluation.dataset_evaluator as mod

    # Ensure evaluator never imports/computes paper aggregates.
    monkeypatch.setattr(
        mod,
        "compute_paper_metrics",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no aggregate metrics")),
        raising=False,
    )

    mvtec_root = tmp_path / "mvtec"
    visa_root = tmp_path / "visa"
    _mvtec_fixture(mvtec_root)

    import json

    _write_rgb(visa_root / "candle" / "Data" / "Images" / "Normal" / "000.JPG")
    _write_rgb(visa_root / "candle" / "Data" / "Images" / "Anomaly" / "000.JPG", (9, 9, 9))
    _write_mask(visa_root / "candle" / "Data" / "Masks" / "Anomaly" / "000.png")
    (visa_root / "meta.json").write_text(
        json.dumps(
            {
                "test": {
                    "candle": [
                        {
                            "img_path": "candle/Data/Images/Anomaly/000.JPG",
                            "mask_path": "candle/Data/Masks/Anomaly/000.png",
                            "cls_name": "candle",
                            "anomaly": 1,
                        },
                        {
                            "img_path": "candle/Data/Images/Normal/000.JPG",
                            "mask_path": "",
                            "cls_name": "candle",
                            "anomaly": 0,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    preprocess = build_preprocess("ViT-L/14", image_size=16)
    device = torch.device("cpu")

    out_m = evaluate_dataset(
        adapter=MVTecAdapter(mvtec_root),
        engine=FakeEngine(),
        preprocess=preprocess,
        device=device,
        split="test",
        limit=2,
    )
    out_v = evaluate_dataset(
        adapter=VisAAdapter(visa_root),
        engine=FakeEngine(),
        preprocess=preprocess,
        device=device,
        split="test",
        limit=2,
    )

    assert isinstance(out_m, EvaluationOutputs)
    assert isinstance(out_v, EvaluationOutputs)
    assert out_m.anomaly_maps.shape[1:] == out_v.anomaly_maps.shape[1:] == (16, 16)
    assert out_m.masks.shape == out_m.anomaly_maps.shape
    assert out_m.image_scores.shape == (2,)
    assert out_v.image_scores.shape == (2,)


def test_limit_follows_deterministic_sorting(tmp_path: Path) -> None:
    _mvtec_fixture(tmp_path)
    out = evaluate_dataset(
        adapter=MVTecAdapter(tmp_path),
        engine=FakeEngine(),
        preprocess=build_preprocess("ViT-L/14", image_size=16),
        device=torch.device("cpu"),
        limit=2,
    )
    ids = [r.sample_id for r in out.records]
    assert ids == sorted(ids)
    assert len(ids) == 2
    # Full sorted list starts with broken_large then goods...
    assert ids[0] == "bottle/test/broken_large/000.png"


def test_normal_masks_become_zeros(tmp_path: Path) -> None:
    _mvtec_fixture(tmp_path)
    out = evaluate_dataset(
        adapter=MVTecAdapter(tmp_path),
        engine=FakeEngine(),
        preprocess=build_preprocess("ViT-L/14", image_size=16),
        device=torch.device("cpu"),
    )
    for i, record in enumerate(out.records):
        if record.image_label == 0:
            assert np.count_nonzero(out.masks[i]) == 0


def test_residual_gain_calls_sample_localization_error_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rad.evaluation.dataset_evaluator as mod

    calls: list[str] = []
    real = mod.sample_localization_error

    def tracked(logits, mask, image_label, **kwargs):  # type: ignore[no-untyped-def]
        tag = "full" if float(logits.mean()) > 0.6 else "adaptive"
        calls.append(tag)
        return real(logits, mask, image_label, **kwargs)

    monkeypatch.setattr(mod, "sample_localization_error", tracked)

    _mvtec_fixture(tmp_path)
    # Only one anomalous sample to keep residual defined.
    out = evaluate_dataset(
        adapter=MVTecAdapter(tmp_path),
        engine=FakeEngine(),
        preprocess=build_preprocess("ViT-L/14", image_size=16),
        device=torch.device("cpu"),
        limit=1,
        compute_full_depth_reference=True,
    )
    assert out.residual_gains is not None
    assert out.residual_gains.shape == (1,)
    assert calls.count("adaptive") == 1
    assert calls.count("full") == 1
    assert isinstance(out.sample_predictions[0], SamplePrediction)
    assert out.sample_predictions[0].residual_gain is not None


def test_residual_gain_source_forbids_metric_differences(tmp_path: Path) -> None:
    import inspect

    import rad.evaluation.dataset_evaluator as mod

    src = inspect.getsource(mod.evaluate_dataset)
    assert "sample_localization_error" in src
    for banned in (
        "pixel_average_precision",
        "average_precision_score",
        "roc_auc_score",
        "safe_aupro",
        "pro_score_proxy",
        "tolerance_boundary_f_score",
        "compute_paper_metrics",
    ):
        assert banned not in src


def test_nonfinite_engine_output_fails_with_sample_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mvtec_fixture(tmp_path)

    class NanEngine(FakeEngine):
        def infer(self, image, *, force_full_depth=False, measure_timing=False):  # type: ignore[no-untyped-def]
            result = super().infer(
                image, force_full_depth=force_full_depth, measure_timing=measure_timing
            )
            result.final_map = result.final_map.clone()
            result.final_map[0, 0, 0] = torch.nan
            return result

    with pytest.raises(MetricComputationError, match="bottle/test/broken_large/000.png"):
        evaluate_dataset(
            adapter=MVTecAdapter(tmp_path),
            engine=NanEngine(),
            preprocess=build_preprocess("ViT-L/14", image_size=16),
            device=torch.device("cpu"),
            limit=1,
        )


def test_memmap_and_memory_modes_match(tmp_path: Path) -> None:
    _mvtec_fixture(tmp_path)
    preprocess = build_preprocess("ViT-L/14", image_size=16)
    device = torch.device("cpu")
    adapter = MVTecAdapter(tmp_path)

    mem = evaluate_dataset(
        adapter=adapter,
        engine=FakeEngine(),
        preprocess=preprocess,
        device=device,
        use_memmap=False,
    )
    mm = evaluate_dataset(
        adapter=adapter,
        engine=FakeEngine(),
        preprocess=preprocess,
        device=device,
        use_memmap=True,
        memmap_dir=tmp_path / "mm",
    )
    assert np.allclose(mem.anomaly_maps, mm.anomaly_maps)
    assert np.allclose(mem.masks, mm.masks)
    assert np.allclose(mem.image_scores, mm.image_scores)
    assert np.array_equal(mem.selected_depths, mm.selected_depths)
