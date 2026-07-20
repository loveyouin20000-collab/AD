"""Selector-signal mask unit and integration tests (TDD)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from rad.models.descriptors import LAYER_DESCRIPTOR_FEATURE_NAMES
from rad.models.selector_signals import (
    REQUIRED_SELECTOR_SIGNAL_NAMES,
    SELECTOR_SIGNAL_FEATURES,
    SelectorSignalLayout,
    apply_selector_signal_mask,
    build_default_selector_signal_layout,
    selector_signal_provenance,
)


def _all_enabled() -> dict[str, bool]:
    return {name: True for name in REQUIRED_SELECTOR_SIGNAL_NAMES}


def _disable(group: str) -> dict[str, bool]:
    signals = _all_enabled()
    signals[group] = False
    return signals


def test_feature_name_groups_cover_18d_exactly_once() -> None:
    all_names = [n for group in SELECTOR_SIGNAL_FEATURES.values() for n in group]
    assert len(all_names) == 18
    assert len(set(all_names)) == 18
    assert set(all_names) == set(LAYER_DESCRIPTOR_FEATURE_NAMES)


def test_default_layout_resolves_authoritative_indices() -> None:
    layout = build_default_selector_signal_layout()
    assert layout.groups["token_separation"] == (0, 1, 2, 3, 4)
    assert layout.groups["response"] == (5, 6, 7, 16)
    assert layout.groups["uncertainty"] == (8, 9, 17)
    assert layout.groups["stability"] == (10, 11, 12)
    assert layout.groups["complementarity"] == (13, 14, 15)
    layout.validate(descriptor_dim=18)


def test_disabling_one_group_zeros_exactly_that_normalized_group() -> None:
    layout = build_default_selector_signal_layout()
    desc = torch.arange(18, dtype=torch.float32).view(1, 1, 18) + 1.0
    masked = apply_selector_signal_mask(
        desc, layout=layout, enabled_signals=_disable("stability")
    )
    for idx in layout.groups["stability"]:
        assert torch.all(masked[..., idx] == 0.0)
    for name, indices in layout.groups.items():
        if name == "stability":
            continue
        for idx in indices:
            assert torch.allclose(masked[..., idx], desc[..., idx])


def test_enabled_groups_remain_unchanged() -> None:
    layout = build_default_selector_signal_layout()
    desc = torch.randn(2, 3, 18)
    masked = apply_selector_signal_mask(
        desc, layout=layout, enabled_signals=_disable("response")
    )
    keep = [i for g, idxs in layout.groups.items() if g != "response" for i in idxs]
    assert torch.allclose(masked[..., keep], desc[..., keep])


def test_unknown_signal_names_fail() -> None:
    layout = build_default_selector_signal_layout()
    desc = torch.randn(1, 2, 18)
    bad = _all_enabled()
    bad["not_a_signal"] = False
    with pytest.raises(ValueError, match="unknown"):
        apply_selector_signal_mask(desc, layout=layout, enabled_signals=bad)


def test_missing_required_signal_entries_fail() -> None:
    layout = build_default_selector_signal_layout()
    desc = torch.randn(1, 2, 18)
    incomplete = {"response": True, "uncertainty": True}
    with pytest.raises(ValueError, match="missing"):
        apply_selector_signal_mask(desc, layout=layout, enabled_signals=incomplete)


def test_overlapping_layout_entries_fail() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SelectorSignalLayout(
            groups={
                "response": (0, 1),
                "uncertainty": (1, 2),
                "stability": (3,),
                "complementarity": (4,),
                "token_separation": (5,),
            }
        ).validate(descriptor_dim=18)


def test_masking_does_not_mutate_input_tensor() -> None:
    layout = build_default_selector_signal_layout()
    desc = torch.randn(1, 4, 18)
    original = desc.clone()
    _ = apply_selector_signal_mask(
        desc, layout=layout, enabled_signals=_disable("complementarity")
    )
    assert torch.allclose(desc, original)


def test_masking_supports_arbitrary_configured_descriptor_dimensions() -> None:
    # Candidate-layer agnostic: only last-dim feature layout matters.
    layout = SelectorSignalLayout(
        groups={
            "token_separation": (0, 1),
            "response": (2,),
            "uncertainty": (3,),
            "stability": (4,),
            "complementarity": (5,),
        }
    )
    layout.validate(descriptor_dim=6)
    for shape in [(1, 1, 6), (2, 5, 6), (3, 7, 6)]:
        desc = torch.randn(*shape)
        masked = apply_selector_signal_mask(
            desc, layout=layout, enabled_signals=_disable("response")
        )
        assert masked.shape == desc.shape
        assert torch.all(masked[..., 2] == 0.0)
        assert torch.allclose(masked[..., :2], desc[..., :2])


def test_indices_outside_descriptor_dim_fail() -> None:
    layout = SelectorSignalLayout(
        groups={
            "token_separation": (0,),
            "response": (1,),
            "uncertainty": (2,),
            "stability": (3,),
            "complementarity": (99,),
        }
    )
    with pytest.raises(ValueError, match="outside|descriptor"):
        layout.validate(descriptor_dim=18)


def test_layout_hash_is_canonical_not_repr() -> None:
    a = build_default_selector_signal_layout()
    b = build_default_selector_signal_layout()
    assert a.layout_hash() == b.layout_hash()
    assert len(a.layout_hash()) == 64
    # Hash must not depend on Python object identity / repr formatting.
    assert a.layout_hash() != repr(a)


def test_provenance_records_mask_stage_and_layout() -> None:
    layout = build_default_selector_signal_layout()
    signals = _disable("uncertainty")
    prov = selector_signal_provenance(
        enabled_signals=signals,
        layout=layout,
        mask_applied=True,
    )
    assert prov["selector_signals"] == signals
    assert prov["selector_mask_applied"] is True
    assert prov["selector_mask_stage"] == "post_normalization_pre_lse"
    assert "token_separation" in prov["selector_signal_layout"]
    assert prov["selector_signal_layout_hash"] == layout.layout_hash()


class _CaptureLSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.captured: list[torch.Tensor] = []

    def forward(self, state: torch.Tensor, depth_id: torch.Tensor) -> Any:
        self.captured.append(state.detach().clone())
        b = state.shape[0]
        from rad.models.lse import GainPrediction

        return GainPrediction(
            mean=torch.zeros(b),
            log_variance=torch.zeros(b),
            sufficiency_logit=torch.zeros(b),
        )


def test_integration_masked_stability_reaches_lse_forward() -> None:
    """Install a fake LSE and verify the tensor entering LSE.forward."""
    from rad.inference.adaptive_engine import AdaptiveEngine
    from rad.models.checkpoint_maps import CheckpointMapGenerator
    from rad.models.descriptors import (
        CheckpointContextExtractor,
        LayerDescriptorExtractor,
    )
    from rad.models.dlcm import DLCM
    from rad.models.policy import PolicyProfile
    layout = build_default_selector_signal_layout()
    capture = _CaptureLSE()
    layers = (6, 12, 18, 24)
    engine = AdaptiveEngine(
        visual=nn.Identity(),  # unused; we call _fuse_at_depth directly
        map_generator=CheckpointMapGenerator(image_size=8, candidate_layers=layers),
        dlcm=DLCM(max_layer_id=24, alpha=0.0),
        lse=capture,  # type: ignore[arg-type]
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=24),
        profile=PolicyProfile.aggressive(gain_threshold=1e6, kappa=0.0),
        candidate_layers=layers,
        early_depths=(12,),
        full_depth=24,
        image_size=8,
        normalizer=None,
        enabled_signals=_disable("stability"),
        selector_layout=layout,
    )

    # Bypass ViT: feed synthetic maps via map_generator.build.
    b, h, w = 1, 8, 8

    def fake_build(depth: int, outputs: Any) -> dict[int, torch.Tensor]:
        avail = [x for x in layers if x <= depth]
        return {lid: torch.randn(b, 1, h, w) for lid in avail}

    engine.map_generator.build = fake_build  # type: ignore[method-assign]
    outputs: dict[int, Any] = {d: object() for d in layers}

    # Controllable normalized descriptor via layer_extractor patch.
    fixed_desc = torch.arange(18, dtype=torch.float32).view(1, 1, 18).expand(1, 2, 18).clone()

    def fake_extract(maps: torch.Tensor, valid_mask: torch.Tensor, fused=None):
        l = maps.shape[1]
        return fixed_desc[:, :l, :].clone()

    engine.layer_extractor.forward = fake_extract  # type: ignore[method-assign]

    fused, weights, state = engine._fuse_at_depth(outputs, depth=12, prev_fused=None)
    assert fused.shape[0] == 1
    assert weights.shape[0] == 1

    # Invoke LSE the same way infer() does.
    depth_id = torch.tensor([12], dtype=torch.long)
    _ = engine.lse(state, depth_id)
    assert len(capture.captured) == 1
    captured = capture.captured[0]
    assert captured.shape[-1] == 26

    pooled = fixed_desc.mean(dim=1)  # [1, 18] — before mask, mean of identical rows
    for idx in layout.groups["stability"]:
        assert torch.all(captured[..., idx] == 0.0)
    for name, indices in layout.groups.items():
        if name == "stability":
            continue
        for idx in indices:
            assert torch.allclose(captured[..., idx], pooled[..., idx])
    # Context (18:26) must remain present (not forced to zero by descriptor mask).
    assert captured[..., 18:].shape[-1] == 8


def test_dlcm_input_unchanged_when_lse_signal_disabled() -> None:
    from rad.inference.adaptive_engine import AdaptiveEngine
    from rad.models.checkpoint_maps import CheckpointMapGenerator
    from rad.models.descriptors import (
        CheckpointContextExtractor,
        LayerDescriptorExtractor,
    )
    from rad.models.dlcm import DLCM
    from rad.models.lse import LSE
    from rad.models.policy import PolicyProfile
    layout = build_default_selector_signal_layout()
    layers = (6, 12)
    seen_dlcm: list[torch.Tensor] = []

    class TrackingDLCM(DLCM):
        def forward(self, layer_desc, ctx, layer_ids, valid_mask, **kwargs):  # type: ignore[no-untyped-def]
            seen_dlcm.append(layer_desc.detach().clone())
            return super().forward(layer_desc, ctx, layer_ids, valid_mask, **kwargs)

    engine = AdaptiveEngine(
        visual=nn.Identity(),
        map_generator=CheckpointMapGenerator(image_size=8, candidate_layers=layers),
        dlcm=TrackingDLCM(max_layer_id=12, alpha=0.0),
        lse=LSE(state_dim=26, early_depths=(12,)),
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=12),
        profile=PolicyProfile.balanced(
            gain_threshold=0.0,
            kappa=0.0,
            map_uncertainty_threshold=1.0,
            image_confidence_margin=0.0,
        ),
        candidate_layers=layers,
        early_depths=(12,),
        full_depth=12,
        image_size=8,
        normalizer=None,
        enabled_signals=_disable("response"),
        selector_layout=layout,
    )

    fixed = torch.arange(18, dtype=torch.float32).view(1, 1, 18).expand(1, 2, 18).contiguous() + 1.0

    def fake_extract(maps, valid_mask, fused=None):  # type: ignore[no-untyped-def]
        return fixed[:, : maps.shape[1], :].clone()

    engine.layer_extractor.forward = fake_extract  # type: ignore[method-assign]
    engine.map_generator.build = (  # type: ignore[method-assign]
        lambda depth, outputs: {
            lid: torch.randn(1, 1, 8, 8) for lid in layers if lid <= depth
        }
    )

    outputs: dict[int, Any] = {d: object() for d in layers}
    _, _, state = engine._fuse_at_depth(outputs, depth=12, prev_fused=None)

    assert len(seen_dlcm) == 1
    assert torch.allclose(seen_dlcm[0], fixed)
    # LSE state response dims zeroed; DLCM still saw full descriptor.
    for idx in layout.groups["response"]:
        assert torch.all(state[..., idx] == 0.0)
    for idx in layout.groups["response"]:
        assert not torch.all(seen_dlcm[0][..., idx] == 0.0)
