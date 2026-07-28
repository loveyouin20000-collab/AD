"""Selector-signal layout and masking for LSE-path descriptor ablations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rad.models.descriptors import LAYER_DESCRIPTOR_FEATURE_NAMES

REQUIRED_SELECTOR_SIGNAL_NAMES: tuple[str, ...] = (
    "response",
    "uncertainty",
    "stability",
    "complementarity",
    "token_separation",
)

SELECTOR_SIGNAL_FEATURES: dict[str, tuple[str, ...]] = {
    "token_separation": (
        "margin_mean",
        "margin_std",
        "margin_max",
        "margin_topk",
        "background_contrast",
    ),
    "response": (
        "response_topk_mean",
        "response_max",
        "sparsity",
        "response_trend",
    ),
    "uncertainty": (
        "top_entropy",
        "global_entropy",
        "entropy_trend",
    ),
    "stability": (
        "rank_spearman",
        "topk_overlap",
        "fused_map_change",
    ),
    "complementarity": (
        "response_comp",
        "absolute_comp",
        "boundary_comp",
    ),
}

SELECTOR_MASK_STAGE = "post_normalization_pre_lse"


def _validate_feature_partition() -> None:
    all_group_features = [
        name for group in SELECTOR_SIGNAL_FEATURES.values() for name in group
    ]
    if len(all_group_features) != 18:
        raise ValueError(
            f"selector signal features must cover 18 dims, got {len(all_group_features)}"
        )
    if len(set(all_group_features)) != 18:
        raise ValueError("selector signal features must be unique (no overlap)")
    if set(all_group_features) != set(LAYER_DESCRIPTOR_FEATURE_NAMES):
        missing = set(LAYER_DESCRIPTOR_FEATURE_NAMES) - set(all_group_features)
        extra = set(all_group_features) - set(LAYER_DESCRIPTOR_FEATURE_NAMES)
        raise ValueError(
            f"selector signal features mismatch descriptor names; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    if set(SELECTOR_SIGNAL_FEATURES) != set(REQUIRED_SELECTOR_SIGNAL_NAMES):
        raise ValueError("SELECTOR_SIGNAL_FEATURES keys must match required signal names")


_validate_feature_partition()


def resolve_feature_indices(feature_names: Sequence[str]) -> tuple[int, ...]:
    index = {name: i for i, name in enumerate(LAYER_DESCRIPTOR_FEATURE_NAMES)}
    missing = [n for n in feature_names if n not in index]
    if missing:
        raise ValueError(f"unknown descriptor feature names: {missing}")
    return tuple(index[n] for n in feature_names)


@dataclass(frozen=True)
class SelectorSignalLayout:
    """Mapping from signal group name to explicit descriptor feature indices."""

    groups: Mapping[str, tuple[int, ...]]

    def validate(self, descriptor_dim: int) -> None:
        if descriptor_dim < 1:
            raise ValueError("descriptor_dim must be >= 1")
        missing = [n for n in REQUIRED_SELECTOR_SIGNAL_NAMES if n not in self.groups]
        if missing:
            raise ValueError(f"missing required selector signal groups: {missing}")
        unknown = sorted(set(self.groups) - set(REQUIRED_SELECTOR_SIGNAL_NAMES))
        if unknown:
            raise ValueError(f"unknown selector signal groups: {unknown}")

        seen: dict[int, str] = {}
        for name, indices in self.groups.items():
            if not indices:
                raise ValueError(f"selector group {name} must be non-empty")
            for idx in indices:
                if idx < 0 or idx >= descriptor_dim:
                    raise ValueError(
                        f"selector group {name} index {idx} outside descriptor "
                        f"dimension {descriptor_dim}"
                    )
                if idx in seen:
                    raise ValueError(
                        f"selector layout overlap at index {idx}: "
                        f"{seen[idx]} and {name}"
                    )
                seen[idx] = name

    def canonical_dict(self) -> dict[str, list[int]]:
        return {
            name: list(self.groups[name])
            for name in REQUIRED_SELECTOR_SIGNAL_NAMES
        }

    def layout_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def feature_name_layout(self) -> dict[str, list[str]]:
        names = list(LAYER_DESCRIPTOR_FEATURE_NAMES)
        return {
            group: [names[i] for i in indices]
            for group, indices in self.canonical_dict().items()
        }


def build_default_selector_signal_layout() -> SelectorSignalLayout:
    groups = {
        name: resolve_feature_indices(features)
        for name, features in SELECTOR_SIGNAL_FEATURES.items()
    }
    layout = SelectorSignalLayout(groups=groups)
    layout.validate(descriptor_dim=len(LAYER_DESCRIPTOR_FEATURE_NAMES))
    return layout


def default_enabled_signals() -> dict[str, bool]:
    return {name: True for name in REQUIRED_SELECTOR_SIGNAL_NAMES}


def parse_enabled_signals(raw: Mapping[str, Any] | None) -> dict[str, bool]:
    if raw is None:
        return default_enabled_signals()
    missing = [n for n in REQUIRED_SELECTOR_SIGNAL_NAMES if n not in raw]
    if missing:
        raise ValueError(f"missing required selector signal entries: {missing}")
    unknown = sorted(set(raw) - set(REQUIRED_SELECTOR_SIGNAL_NAMES))
    if unknown:
        raise ValueError(f"unknown selector signal names: {unknown}")
    out: dict[str, bool] = {}
    for name in REQUIRED_SELECTOR_SIGNAL_NAMES:
        value = raw[name]
        if not isinstance(value, bool):
            raise ValueError(f"selector signal {name} must be bool, got {type(value)}")
        out[name] = value
    return out


def apply_selector_signal_mask(
    descriptor: torch.Tensor,
    *,
    layout: SelectorSignalLayout,
    enabled_signals: Mapping[str, bool],
) -> torch.Tensor:
    """Clone descriptor and zero disabled signal groups (last feature dim)."""
    if descriptor.ndim < 1:
        raise ValueError("descriptor must be a tensor with a feature dimension")
    descriptor_dim = int(descriptor.shape[-1])
    layout.validate(descriptor_dim=descriptor_dim)

    missing = [n for n in REQUIRED_SELECTOR_SIGNAL_NAMES if n not in enabled_signals]
    if missing:
        raise ValueError(f"missing required selector signal entries: {missing}")
    unknown = sorted(set(enabled_signals) - set(REQUIRED_SELECTOR_SIGNAL_NAMES))
    if unknown:
        raise ValueError(f"unknown selector signal names: {unknown}")

    masked = descriptor.clone()
    for name in REQUIRED_SELECTOR_SIGNAL_NAMES:
        if not bool(enabled_signals[name]):
            indices = layout.groups[name]
            masked[..., indices] = 0.0
    return masked


def any_signal_disabled(enabled_signals: Mapping[str, bool]) -> bool:
    return any(not bool(enabled_signals[n]) for n in REQUIRED_SELECTOR_SIGNAL_NAMES)


def selector_signal_provenance(
    *,
    enabled_signals: Mapping[str, bool],
    layout: SelectorSignalLayout,
    mask_applied: bool,
) -> dict[str, Any]:
    signals = {name: bool(enabled_signals[name]) for name in REQUIRED_SELECTOR_SIGNAL_NAMES}
    return {
        "selector_signals": signals,
        "selector_mask_applied": bool(mask_applied),
        "selector_mask_stage": SELECTOR_MASK_STAGE,
        "selector_signal_layout": layout.feature_name_layout(),
        "selector_signal_layout_hash": layout.layout_hash(),
    }
