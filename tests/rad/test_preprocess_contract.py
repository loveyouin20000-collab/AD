from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode

from rad.data.adapters.preprocess import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessSpec,
    build_preprocess,
    preprocess_image,
    preprocess_mask,
)
from rad.errors import ConfigurationContractError


def test_preprocess_spec_is_frozen() -> None:
    spec = PreprocessSpec(
        image_size=32,
        mean=CLIP_MEAN,
        std=CLIP_STD,
        interpolation=InterpolationMode.BICUBIC,
    )
    with pytest.raises(AttributeError):
        spec.image_size = 64  # type: ignore[misc]


def test_build_preprocess_clip_uses_visualad_constants() -> None:
    spec = build_preprocess("ViT-L/14@336px", image_size=518)
    assert spec.image_size == 518
    assert spec.mean == CLIP_MEAN
    assert spec.std == CLIP_STD
    assert spec.interpolation == InterpolationMode.BICUBIC
    assert spec.mean == (0.48145466, 0.4578275, 0.40821073)
    assert spec.std == (0.26862954, 0.26130258, 0.27577711)


def test_build_preprocess_dinov2_uses_imagenet_constants() -> None:
    spec = build_preprocess("dinov2_vitl14", image_size=224)
    assert spec.mean == IMAGENET_MEAN
    assert spec.std == IMAGENET_STD
    assert spec.mean == (0.485, 0.456, 0.406)
    assert spec.std == (0.229, 0.224, 0.225)
    assert spec.interpolation == InterpolationMode.BICUBIC


def test_build_preprocess_unknown_backbone_fails() -> None:
    with pytest.raises(ConfigurationContractError, match="backbone|unknown|Unsupported"):
        build_preprocess("resnet50", image_size=224)


def test_preprocess_image_shape_and_clip_normalization() -> None:
    spec = build_preprocess("ViT-L/14", image_size=16)
    # Constant RGB=128 → after /255 and CLIP normalize.
    image = Image.new("RGB", (8, 8), (128, 128, 128))
    tensor = preprocess_image(image, spec)

    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32
    expected = torch.tensor(
        [((128 / 255.0) - m) / s for m, s in zip(CLIP_MEAN, CLIP_STD, strict=True)],
        dtype=torch.float32,
    ).view(3, 1, 1)
    assert torch.allclose(tensor, expected.expand_as(tensor), atol=1e-5)


def test_preprocess_image_uses_configured_interpolation() -> None:
    # Checkerboard so bicubic vs nearest differ after upsample.
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[0::2, 0::2] = 255
    arr[1::2, 1::2] = 255
    image = Image.fromarray(arr, mode="RGB")

    bicubic = PreprocessSpec(
        image_size=32,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        interpolation=InterpolationMode.BICUBIC,
    )
    nearest = PreprocessSpec(
        image_size=32,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        interpolation=InterpolationMode.NEAREST,
    )
    t_bi = preprocess_image(image, bicubic)
    t_nn = preprocess_image(image, nearest)
    assert not torch.allclose(t_bi, t_nn)


def test_preprocess_mask_nearest_binary_no_normalization() -> None:
    # Soft gray values must become binary {0,1}, never mean/std normalized.
    mask = Image.fromarray(
        np.array(
            [
                [0, 64, 128],
                [200, 255, 10],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        ),
        mode="L",
    )
    out = preprocess_mask(mask, image_size=6)
    assert out.shape == (6, 6)
    assert out.dtype == torch.float32
    unique = set(torch.unique(out).tolist())
    assert unique <= {0.0, 1.0}
    # Values were thresholded, not ImageNet-normalized into negative range.
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_preprocess_mask_none_becomes_zero_mask() -> None:
    out = preprocess_mask(None, image_size=8)
    assert out.shape == (8, 8)
    assert torch.count_nonzero(out) == 0
    assert out.dtype == torch.float32


def test_adapters_remain_preprocessing_free() -> None:
    import inspect

    from rad.data.adapters import mvtec, visa

    for module in (mvtec, visa):
        source = inspect.getsource(module)
        assert "PreprocessSpec" not in source
        assert "preprocess_image" not in source
        assert "preprocess_mask" not in source
        assert "Normalize" not in source
        assert "torchvision.transforms" not in source
