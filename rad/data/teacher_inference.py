from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rad.models.checkpoint_maps import anomaly_map_from_tokens
from rad.types import CheckpointOutput


@dataclass
class TeacherBundle:
    model: nn.Module
    layer_transforms: nn.ModuleDict
    cross_attn: nn.Module | None
    features_list: list[int]
    image_size: int
    device: torch.device


def load_teacher_bundle(
    checkpoint_path: Path | str,
    *,
    device: str | torch.device = "cuda:0",
    backbone: str | None = None,
) -> TeacherBundle:
    import VisualAD_lib
    from utils.feature_transform import create_feature_transform
    from utils.spatial_cross_attention import build_layer_adaptive_cross_attention

    device_t = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_t)
    backbone_name = backbone or checkpoint.get("backbone", "ViT-L/14@336px")
    image_size = int(checkpoint.get("image_size", 518))
    features_list = [int(x) for x in checkpoint.get("features_list", [6, 12, 18, 24])]

    model, _ = VisualAD_lib.load(backbone_name, device=device_t)
    model.eval()
    model.to(device_t)
    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device_t)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device_t)

    feature_dim = model.visual.embed_dim
    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        for layer_name, state_dict in checkpoint["layer_transforms"].items():
            hidden_dim = state_dict["mlp.0.weight"].shape[0]
            module = create_feature_transform(
                transform_type="mlp",
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0,
            ).to(device_t)
            module.load_state_dict(state_dict)
            module.eval()
            layer_transforms[layer_name] = module

    cross_attn = None
    if "cross_attn" in checkpoint:
        config = checkpoint.get("cross_attn_config", {})
        cross_attn = build_layer_adaptive_cross_attention(
            layers=features_list,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            res_scale_init=config.get("res_scale_init", 0.01),
        ).to(device_t)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    return TeacherBundle(
        model=model,
        layer_transforms=layer_transforms,
        cross_attn=cross_attn,
        features_list=features_list,
        image_size=image_size,
        device=device_t,
    )


def _apply_layer_transform(
    layer_transforms: nn.ModuleDict,
    layer: int,
    patch_tokens: torch.Tensor,
) -> torch.Tensor:
    key = f"layer_{layer}"
    if key not in layer_transforms:
        return patch_tokens
    batch, num_tokens, dim = patch_tokens.shape
    transformed = layer_transforms[key](patch_tokens.reshape(-1, dim))
    return transformed.view(batch, num_tokens, dim)


def build_causal_maps_and_ingredients(
    bundle: TeacherBundle,
    image: torch.Tensor,
    candidate_layers: Sequence[int],
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[str, Any], torch.Tensor]:
    """Run staged teacher forward and build causal A_{l|d} maps.

    Returns:
        maps: depth -> layer -> [H, W]
        ingredients: raw tokens for descriptor extraction
        teacher_logits: fused full-depth map [H, W]
    """
    layers = tuple(int(x) for x in candidate_layers)
    with torch.no_grad():
        staged: Mapping[int, CheckpointOutput] = bundle.model.visual.forward_staged(
            image, list(layers)
        )

        ingredients = {
            "patch_tokens": {
                depth: staged[depth].patch_tokens[0].detach().cpu() for depth in layers
            },
            "anomaly_tokens": {
                depth: staged[depth].anomaly_token[0].detach().cpu() for depth in layers
            },
            "normal_tokens": {
                depth: staged[depth].normal_token[0].detach().cpu() for depth in layers
            },
        }

        maps: dict[int, dict[int, torch.Tensor]] = {}
        for depth in layers:
            available = [layer for layer in layers if layer <= depth]
            anomaly = staged[depth].anomaly_token
            normal = staged[depth].normal_token
            patch_list = [staged[layer].patch_tokens for layer in available]

            if bundle.cross_attn is not None:
                adapted = bundle.cross_attn(anomaly, normal, patch_list, available)
                anomaly_list = [item["anomaly"] for item in adapted]
                normal_list = [item["normal"] for item in adapted]
            else:
                anomaly_list = [anomaly] * len(available)
                normal_list = [normal] * len(available)

            depth_maps: dict[int, torch.Tensor] = {}
            for idx, layer in enumerate(available):
                patch = _apply_layer_transform(
                    bundle.layer_transforms, layer, staged[layer].patch_tokens
                )
                amap = anomaly_map_from_tokens(
                    anomaly_list[idx],
                    normal_list[idx],
                    patch,
                    bundle.image_size,
                )
                depth_maps[layer] = amap[0].detach().cpu()
            maps[depth] = depth_maps

        final_depth = layers[-1]
        teacher_logits = torch.stack(list(maps[final_depth].values()), dim=0).sum(dim=0)

    return maps, ingredients, teacher_logits
