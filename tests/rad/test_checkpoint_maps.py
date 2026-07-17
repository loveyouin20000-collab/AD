from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from rad.models.checkpoint_maps import CheckpointMapGenerator, anomaly_map_from_tokens

_VISUALAD_PATH = Path(__file__).resolve().parents[2] / "VisualAD_lib" / "VisualAD.py"
_SPEC = importlib.util.spec_from_file_location("visualad_core_maps", _VISUALAD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
VisualAD = _MODULE.VisualAD

CANDIDATE_LAYERS = (6, 12, 18, 24)
IMAGE_SIZE = 32


@pytest.fixture
def visualad_model() -> VisualAD:
    model = VisualAD(
        embed_dim=64,
        image_resolution=IMAGE_SIZE,
        vision_layers=24,
        vision_width=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=2,
    )
    model.eval()
    return model


@pytest.fixture
def sample_image() -> torch.Tensor:
    return torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def staged_outputs(visualad_model: VisualAD, sample_image: torch.Tensor):
    with torch.no_grad():
        return visualad_model.visual.forward_staged(sample_image, list(CANDIDATE_LAYERS))


@pytest.fixture
def official_map_list(visualad_model: VisualAD, sample_image: torch.Tensor):
    with torch.no_grad():
        legacy = visualad_model.encode_image(sample_image, list(CANDIDATE_LAYERS))
    anomaly = legacy["anomaly_features"]
    normal = legacy["normal_features"]
    patch_start = legacy["patch_start_idx"]
    maps = []
    for patch_feature in legacy["patch_tokens"]:
        amap = anomaly_map_from_tokens(
            anomaly,
            normal,
            patch_feature[:, patch_start:, :],
            IMAGE_SIZE,
        )
        maps.append(amap.unsqueeze(1))
    return maps


@pytest.fixture
def generator() -> CheckpointMapGenerator:
    return CheckpointMapGenerator(image_size=IMAGE_SIZE)


def test_checkpoint_12_cannot_use_deeper_patch_tokens(generator, staged_outputs):
    maps = generator.build(depth=12, outputs=staged_outputs)
    assert set(maps) == {6, 12}


def test_checkpoint_24_matches_official_map_list(generator, official_map_list, staged_outputs):
    maps = generator.build(depth=24, outputs=staged_outputs)
    for depth, expected in zip(CANDIDATE_LAYERS, official_map_list, strict=True):
        assert torch.allclose(maps[depth], expected, atol=1e-5, rtol=1e-4)
