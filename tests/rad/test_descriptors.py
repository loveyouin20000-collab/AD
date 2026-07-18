from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rad.models.descriptors import (
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)


def test_layer_descriptors_finite_on_all_zero_and_constant_maps():
    extractor = LayerDescriptorExtractor(top_k_ratio=0.1)
    zeros = torch.zeros(2, 3, 16, 16)
    const = torch.ones(2, 3, 16, 16) * 0.5
    valid = torch.ones(2, 3, dtype=torch.bool)
    for maps in (zeros, const):
        out = extractor(maps, valid_mask=valid)
        assert out.shape == (2, 3, 18)
        assert torch.isfinite(out).all()


def test_layer_descriptors_invariant_to_batch_order():
    extractor = LayerDescriptorExtractor(top_k_ratio=0.1)
    maps = torch.randn(4, 3, 16, 16)
    valid = torch.tensor(
        [[True, True, False], [True, False, True], [True, True, True], [False, True, True]]
    )
    out = extractor(maps, valid_mask=valid)
    perm = torch.tensor([2, 0, 3, 1])
    out_perm = extractor(maps[perm], valid_mask=valid[perm])
    assert torch.allclose(out[perm], out_perm, atol=1e-5, rtol=1e-4)


def test_checkpoint_context_finite_and_shape():
    extractor = CheckpointContextExtractor(backbone_depth=24)
    maps = torch.randn(2, 2, 16, 16)
    valid = torch.ones(2, 2, dtype=torch.bool)
    layer_ids = torch.tensor([[6, 12], [6, 12]])
    out = extractor(
        maps,
        valid_mask=valid,
        layer_ids=layer_ids,
        prev_fused=torch.zeros(2, 1, 16, 16),
    )
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()
    # depth ratio for layer 12 / 24 = 0.5 when current depth is max valid layer
    assert torch.allclose(out[:, 0], torch.tensor([0.5, 0.5]), atol=1e-5)


def test_normalizer_fit_median_iqr_and_clamp(tmp_path: Path):
    # Tight IQR around 0 so large outliers clamp to ±8
    feats = torch.tensor([[-1000.0], [-0.1], [0.0], [0.1], [1000.0]])
    normalizer = DescriptorNormalizer(clamp=(-8.0, 8.0))
    normalizer.fit(feats)
    out = normalizer.transform(feats)
    assert out.shape == feats.shape
    assert out.min() >= -8.0
    assert out.max() <= 8.0
    assert out[0, 0] == -8.0
    assert out[-1, 0] == 8.0
    # near-median values stay small
    assert out[2, 0].abs() < 1.0


def test_normalizer_fit_from_cache_rejects_non_train_split(tmp_path: Path):
    meta = {
        "schema_version": 1,
        "split": "calibration",
        "candidate_layers": [6, 12],
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    # minimal empty index to pass dataset construction would fail; call helper directly
    normalizer = DescriptorNormalizer()
    with pytest.raises(ValueError, match="train"):
        normalizer.fit_from_cache(tmp_path)


def test_normalizer_roundtrip_save_load(tmp_path: Path):
    feats = torch.randn(32, 18)
    normalizer = DescriptorNormalizer(clamp=(-8.0, 8.0))
    normalizer.fit(feats)
    path = tmp_path / "stats.json"
    normalizer.save(path)
    loaded = DescriptorNormalizer.load(path)
    a = normalizer.transform(feats)
    b = loaded.transform(feats)
    assert torch.allclose(a, b)
