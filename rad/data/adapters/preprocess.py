"""Backbone-aware evaluation preprocessing (separate from dataset adapters)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from rad.errors import ConfigurationContractError

# VisualAD / OpenAI CLIP constants (VisualAD_lib.constants).
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# Standard ImageNet normalization for DINOv2.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessSpec:
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: InterpolationMode


def _classify_backbone(backbone_name: str) -> str:
    name = backbone_name.strip().lower()
    if not name:
        raise ConfigurationContractError("backbone_name must be non-empty")
    if "dino" in name:
        return "dinov2"
    if (
        "clip" in name
        or name.startswith("vit-")
        or name.startswith("vit_")
        or "/" in name  # e.g. ViT-L/14@336px
        or "@" in name
    ):
        return "clip"
    raise ConfigurationContractError(
        f"Unsupported backbone for evaluation preprocessing: {backbone_name!r}. "
        "Supported first-round families: CLIP (e.g. ViT-L/14@336px) and DINOv2."
    )


def build_preprocess(backbone_name: str, image_size: int) -> PreprocessSpec:
    """Build a preprocess spec from backbone family and target size."""
    if int(image_size) <= 0:
        raise ConfigurationContractError(f"image_size must be positive, got {image_size}")
    family = _classify_backbone(backbone_name)
    if family == "clip":
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    return PreprocessSpec(
        image_size=int(image_size),
        mean=mean,
        std=std,
        interpolation=InterpolationMode.BICUBIC,
    )


def preprocess_image(image: Image.Image, spec: PreprocessSpec) -> torch.Tensor:
    """Resize with configured interpolation, convert to float tensor, normalize."""
    rgb = image.convert("RGB")
    resized = TF.resize(
        rgb,
        size=[spec.image_size, spec.image_size],
        interpolation=spec.interpolation,
        antialias=True,
    )
    tensor = TF.to_tensor(resized)  # float32 in [0, 1], shape (3, H, W)
    return TF.normalize(tensor, mean=list(spec.mean), std=list(spec.std))


def preprocess_mask(
    mask: Image.Image | None,
    image_size: int,
) -> torch.Tensor:
    """Nearest-neighbor resize to a binary float mask in {0, 1}; never normalize."""
    size = int(image_size)
    if size <= 0:
        raise ConfigurationContractError(f"image_size must be positive, got {image_size}")
    if mask is None:
        return torch.zeros((size, size), dtype=torch.float32)

    gray = mask.convert("L")
    resized = gray.resize((size, size), resample=Image.Resampling.NEAREST)
    tensor = TF.to_tensor(resized).squeeze(0)  # (H, W) in [0, 1]
    # Binary: any positive intensity is foreground (matches VisualAD >0 thresholding).
    return (tensor > 0).to(dtype=torch.float32)
