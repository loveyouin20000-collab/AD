from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_softmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.amax(dim=dim, keepdim=True)
    ex = torch.exp(x)
    return ex / ex.sum(dim=dim, keepdim=True).clamp_min(eps)


def _spatial_entropy(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy of spatial softmax over HxW. maps: [..., H, W] -> [...]"""
    flat = maps.flatten(-2)
    probs = _masked_softmax(flat, dim=-1, eps=eps)
    return -(probs * (probs.clamp_min(eps).log())).sum(dim=-1)


def _sobel_magnitude(maps: torch.Tensor) -> torch.Tensor:
    """maps: [B, L, H, W] -> [B, L, H, W]"""
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=maps.device,
        dtype=maps.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=maps.device,
        dtype=maps.dtype,
    ).view(1, 1, 3, 3)
    b, l, h, w = maps.shape
    x = maps.reshape(b * l, 1, h, w)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12).view(b, l, h, w)


def _topk_mean(flat: torch.Tensor, k: int) -> torch.Tensor:
    """flat: [B, L, N] -> [B, L] mean of top-k along last dim."""
    k = max(1, min(k, flat.shape[-1]))
    values, _ = torch.topk(flat, k=k, dim=-1)
    return values.mean(dim=-1)


def _bottomk_mean(flat: torch.Tensor, k: int) -> torch.Tensor:
    k = max(1, min(k, flat.shape[-1]))
    values, _ = torch.topk(flat, k=k, dim=-1, largest=False)
    return values.mean(dim=-1)


def _spearman_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """a,b: [B, L, N] -> [B, L] Spearman correlation."""
    a_rank = a.argsort(dim=-1).argsort(dim=-1).float()
    b_rank = b.argsort(dim=-1).argsort(dim=-1).float()
    a_c = a_rank - a_rank.mean(dim=-1, keepdim=True)
    b_c = b_rank - b_rank.mean(dim=-1, keepdim=True)
    num = (a_c * b_c).sum(dim=-1)
    den = torch.sqrt((a_c * a_c).sum(dim=-1) * (b_c * b_c).sum(dim=-1)).clamp_min(eps)
    return num / den


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    """IoU of top-k index sets. a,b: [B, L, N] -> [B, L]."""
    k = max(1, min(k, a.shape[-1]))
    _, a_idx = torch.topk(a, k=k, dim=-1)
    _, b_idx = torch.topk(b, k=k, dim=-1)
    # Build masks
    a_mask = torch.zeros_like(a, dtype=torch.bool)
    b_mask = torch.zeros_like(b, dtype=torch.bool)
    a_mask.scatter_(-1, a_idx, True)
    b_mask.scatter_(-1, b_idx, True)
    inter = (a_mask & b_mask).float().sum(dim=-1)
    union = (a_mask | b_mask).float().sum(dim=-1).clamp_min(eps)
    return inter / union


class LayerDescriptorExtractor(nn.Module):
    """Extract 18-D descriptors per layer from checkpoint-conditioned maps."""

    def __init__(self, top_k_ratio: float = 0.1) -> None:
        super().__init__()
        if not 0.0 < top_k_ratio <= 1.0:
            raise ValueError("top_k_ratio must be in (0, 1]")
        self.top_k_ratio = top_k_ratio

    def forward(
        self,
        maps: torch.Tensor,
        valid_mask: torch.Tensor,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            maps: [B, L, H, W]
            valid_mask: [B, L] bool
            fused: optional [B, 1, H, W]; defaults to equal-weight sum over valid layers
        Returns:
            [B, L, 18]
        """
        if maps.ndim != 4:
            raise ValueError("maps must have shape [B, L, H, W]")
        b, l, h, w = maps.shape
        flat = maps.flatten(2)  # [B, L, N]
        n = flat.shape[-1]
        k = max(1, int(round(self.top_k_ratio * n)))

        if fused is None:
            weights = valid_mask.to(maps.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            fused = (maps * weights[:, :, None, None]).sum(dim=1, keepdim=True)
        fused_flat = fused.flatten(2).expand(-1, l, -1)

        margin_mean = flat.mean(dim=-1)
        margin_std = flat.std(dim=-1, unbiased=False)
        margin_max = flat.amax(dim=-1)
        margin_topk = _topk_mean(flat, k)
        background = _bottomk_mean(flat, k)
        background_contrast = margin_topk - background

        response = flat.abs()
        response_topk_mean = _topk_mean(response, k)
        response_max = response.amax(dim=-1)
        thresh = margin_mean.unsqueeze(-1) + margin_std.unsqueeze(-1)
        sparsity = (flat > thresh).float().mean(dim=-1)

        # top entropy over top-k mass; global spatial entropy
        top_vals, _ = torch.topk(flat, k=k, dim=-1)
        top_entropy = -(
            _masked_softmax(top_vals, dim=-1) * F.log_softmax(top_vals, dim=-1)
        ).sum(dim=-1)
        global_entropy = _spatial_entropy(maps)

        rank_spearman = _spearman_corr(flat, fused_flat)
        topk_overlap = _topk_overlap(flat, fused_flat, k)
        fused_map_change = (flat - fused_flat).abs().mean(dim=-1)

        # Complementarity vs fused / other layers
        response_comp = 1.0 - torch.cosine_similarity(
            response, fused_flat.abs(), dim=-1
        )
        absolute_comp = 1.0 - torch.cosine_similarity(flat, fused_flat, dim=-1)
        boundaries = _sobel_magnitude(maps).flatten(2)
        fused_boundary = _sobel_magnitude(fused).flatten(2).expand(-1, l, -1)
        boundary_comp = 1.0 - torch.cosine_similarity(boundaries, fused_boundary, dim=-1)

        # Trends along layer axis (pad first layer with zeros)
        response_mean = response.mean(dim=-1)
        entropy = global_entropy
        zeros = torch.zeros(b, 1, device=maps.device, dtype=maps.dtype)
        response_trend = torch.cat([zeros, response_mean[:, 1:] - response_mean[:, :-1]], dim=1)
        entropy_trend = torch.cat([zeros, entropy[:, 1:] - entropy[:, :-1]], dim=1)

        feats = torch.stack(
            [
                margin_mean,
                margin_std,
                margin_max,
                margin_topk,
                background_contrast,
                response_topk_mean,
                response_max,
                sparsity,
                top_entropy,
                global_entropy,
                rank_spearman,
                topk_overlap,
                fused_map_change,
                response_comp,
                absolute_comp,
                boundary_comp,
                response_trend,
                entropy_trend,
            ],
            dim=-1,
        )
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = feats * valid_mask.to(feats.dtype).unsqueeze(-1)
        return feats


class CheckpointContextExtractor(nn.Module):
    """Extract 8-D checkpoint context features."""

    def __init__(self, backbone_depth: int) -> None:
        super().__init__()
        if backbone_depth < 1:
            raise ValueError("backbone_depth must be >= 1")
        self.backbone_depth = backbone_depth

    def forward(
        self,
        maps: torch.Tensor,
        valid_mask: torch.Tensor,
        layer_ids: torch.Tensor,
        weights: torch.Tensor | None = None,
        prev_fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            maps: [B, L, H, W]
            valid_mask: [B, L]
            layer_ids: [B, L] integer layer indices
            weights: optional [B, L]; default equal over valid
            prev_fused: optional [B, 1, H, W]
        Returns:
            [B, 8]
        """
        b, l, h, w = maps.shape
        valid_f = valid_mask.to(maps.dtype)
        n_valid = valid_f.sum(dim=1).clamp_min(1.0)

        if weights is None:
            weights = valid_f / n_valid.unsqueeze(1)
        else:
            weights = weights * valid_f
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        fused = (maps * weights[:, :, None, None]).sum(dim=1, keepdim=True)

        # current depth = max valid layer id
        neg_inf = torch.full_like(layer_ids, -1)
        current_depth = torch.where(valid_mask, layer_ids, neg_inf).amax(dim=1).float()
        depth_ratio = current_depth / float(self.backbone_depth)

        map_entropy = _spatial_entropy(fused.squeeze(1))
        boundary = _sobel_magnitude(fused)
        boundary_entropy = _spatial_entropy(boundary.squeeze(1))

        flat = fused.flatten(2).squeeze(1)
        n = flat.shape[-1]
        k = max(1, int(round(0.1 * n)))
        image_anomaly_confidence = _topk_mean(flat.unsqueeze(1), k).squeeze(1)
        image_normal_confidence = -flat.mean(dim=-1)

        # weight entropy
        w = weights.clamp_min(1e-8)
        weight_entropy = -(w * w.log()).sum(dim=1)

        if prev_fused is None:
            prev_change = torch.zeros(b, device=maps.device, dtype=maps.dtype)
        else:
            prev_change = (fused - prev_fused).abs().mean(dim=(1, 2, 3))

        ctx = torch.stack(
            [
                depth_ratio,
                map_entropy,
                boundary_entropy,
                image_normal_confidence,
                image_anomaly_confidence,
                weight_entropy,
                n_valid,
                prev_change,
            ],
            dim=-1,
        )
        return torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)


class DescriptorNormalizer:
    """Median/IQR normalizer with hard clamp; fit on source-train only."""

    def __init__(self, clamp: tuple[float, float] = (-8.0, 8.0), eps: float = 1e-6) -> None:
        self.clamp = clamp
        self.eps = eps
        self.median: torch.Tensor | None = None
        self.iqr: torch.Tensor | None = None

    def fit(self, features: torch.Tensor) -> DescriptorNormalizer:
        if features.ndim != 2:
            raise ValueError("features must be [N, D]")
        q25 = torch.quantile(features.float(), 0.25, dim=0)
        q75 = torch.quantile(features.float(), 0.75, dim=0)
        self.median = torch.quantile(features.float(), 0.5, dim=0)
        self.iqr = (q75 - q25).clamp_min(self.eps)
        return self

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.median is None or self.iqr is None:
            raise RuntimeError("DescriptorNormalizer must be fit before transform")
        median = self.median.to(device=features.device, dtype=torch.float32)
        iqr = self.iqr.to(device=features.device, dtype=torch.float32)
        out = (features.float() - median) / iqr
        return out.clamp(self.clamp[0], self.clamp[1])

    def fit_from_cache(self, cache_dir: Path | str, max_samples: int | None = None) -> DescriptorNormalizer:
        from rad.data.cache_dataset import TeacherCacheDataset

        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "meta.json").read_text())
        split = str(meta.get("split", ""))
        if split != "train":
            raise ValueError(
                f"DescriptorNormalizer.fit_from_cache may only read source-train cache, got split={split!r}"
            )

        dataset = TeacherCacheDataset(cache_dir)
        layers = tuple(int(x) for x in meta["candidate_layers"])
        extractor = LayerDescriptorExtractor()
        rows: list[torch.Tensor] = []
        n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        try:
            from tqdm import tqdm

            indices = tqdm(range(n), desc="fit_descriptors")
        except ImportError:
            indices = range(n)
        for i in indices:
            record = dataset[i]
            depth = layers[-1]
            layer_maps = record["maps"][depth]
            stacked = torch.stack([layer_maps[layer] for layer in layers], dim=0)  # [L,H,W]
            maps = stacked.unsqueeze(0)
            valid = torch.ones(1, len(layers), dtype=torch.bool)
            desc = extractor(maps, valid_mask=valid)  # [1, L, 18]
            rows.append(desc.reshape(-1, 18))
        features = torch.cat(rows, dim=0)
        return self.fit(features)

    def save(self, path: Path | str) -> None:
        if self.median is None or self.iqr is None:
            raise RuntimeError("cannot save unfitted normalizer")
        payload = {
            "clamp": list(self.clamp),
            "eps": self.eps,
            "median": self.median.tolist(),
            "iqr": self.iqr.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> DescriptorNormalizer:
        payload: dict[str, Any] = json.loads(Path(path).read_text())
        obj = cls(clamp=tuple(payload["clamp"]), eps=float(payload.get("eps", 1e-6)))
        obj.median = torch.tensor(payload["median"], dtype=torch.float32)
        obj.iqr = torch.tensor(payload["iqr"], dtype=torch.float32)
        return obj
