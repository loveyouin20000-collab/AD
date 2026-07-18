from __future__ import annotations

import torch
import torch.nn.functional as F


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """x: [B, 1, H, W] -> [B, 1, H, W] edge magnitude."""
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def soft_dice_loss(probs: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample soft Dice loss. probs/mask: [B, 1, H, W] -> [B]."""
    dims = (1, 2, 3)
    intersection = (probs * mask).sum(dim=dims)
    denom = probs.sum(dim=dims) + mask.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice


def boundary_l1_loss(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sobel-boundary L1 between predicted probs and GT mask. -> [B]."""
    pred_b = _sobel_magnitude(probs)
    gt_b = _sobel_magnitude(mask)
    return (pred_b - gt_b).abs().mean(dim=(1, 2, 3))


def topk_mean_probability(probs: torch.Tensor, ratio: float = 0.01) -> torch.Tensor:
    """Mean of top-k% pixels. probs: [B, 1, H, W] -> [B]."""
    flat = probs.flatten(1)
    n = flat.shape[-1]
    k = max(1, int(round(ratio * n)))
    values, _ = torch.topk(flat, k=k, dim=-1)
    return values.mean(dim=-1)


def sample_localization_error(
    logits: torch.Tensor,
    mask: torch.Tensor,
    image_label: torch.Tensor,
    *,
    topk_ratio: float = 0.01,
) -> torch.Tensor:
    """Differentiable per-sample localization error.

    E = 1.0 * BCEWithLogits
      + 1.0 * SoftDice
      + 0.2 * BoundaryL1
      + 0.2 * BCE(TopKMeanProbability, image_label)

    Args:
        logits: [B, 1, H, W] (or [B, H, W])
        mask: same spatial shape, values in {0, 1}
        image_label: [B] image-level anomaly label in {0, 1}
    Returns:
        [B] per-sample error
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if logits.shape != mask.shape:
        raise ValueError(f"logits/mask shape mismatch: {logits.shape} vs {mask.shape}")
    if image_label.ndim != 1 or image_label.shape[0] != logits.shape[0]:
        raise ValueError("image_label must have shape [B]")

    mask = mask.to(dtype=logits.dtype)
    image_label = image_label.to(dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    bce = bce.mean(dim=(1, 2, 3))

    probs = torch.sigmoid(logits)
    dice = soft_dice_loss(probs, mask)
    boundary = boundary_l1_loss(probs, mask)

    topk = topk_mean_probability(probs, ratio=topk_ratio).clamp(1e-6, 1.0 - 1e-6)
    # BCE between top-k mean probability and image label (per sample)
    topk_bce = F.binary_cross_entropy(topk, image_label, reduction="none")

    return 1.0 * bce + 1.0 * dice + 0.2 * boundary + 0.2 * topk_bce
