from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rad.models.checkpoint_maps import CheckpointMapGenerator
from rad.models.descriptors import (
    CheckpointContextExtractor,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM
from rad.models.lse import LSE
from rad.models.policy import PolicyProfile

_VISUALAD_PATH = Path(__file__).resolve().parents[2] / "VisualAD_lib" / "VisualAD.py"
_SPEC = importlib.util.spec_from_file_location("visualad_core_adaptive", _VISUALAD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
VisualAD = _MODULE.VisualAD


@pytest.fixture
def visualad_model() -> VisualAD:
    model = VisualAD(
        embed_dim=64,
        image_resolution=32,
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
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def block_call_counter(
    visualad_model: VisualAD,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    counter = SimpleNamespace(total=0)
    for block in visualad_model.visual.transformer.resblocks:
        original = block.forward

        def counted(self_block, x, _original=original, _counter=counter):
            _counter.total += 1
            return _original(x)

        monkeypatch.setattr(block, "forward", counted.__get__(block, type(block)))
    return counter


def _always_exit_profile() -> PolicyProfile:
    return PolicyProfile.aggressive(gain_threshold=1e6, kappa=0.0)


def _never_exit_until_full_profile() -> PolicyProfile:
    # UCB always high relative to threshold -> never exit early
    return PolicyProfile.aggressive(gain_threshold=-1.0, kappa=0.0)


def _build_engine(
    visualad_model: VisualAD,
    *,
    profile: PolicyProfile,
    candidate_layers: tuple[int, ...] = (6, 12, 18, 24),
    early_depths: tuple[int, ...] = (12, 18),
    image_size: int = 32,
):
    from rad.inference.adaptive_engine import AdaptiveEngine

    max_layer = max(candidate_layers)
    return AdaptiveEngine(
        visual=visualad_model.visual,
        map_generator=CheckpointMapGenerator(
            image_size=image_size, candidate_layers=candidate_layers
        ),
        dlcm=DLCM(max_layer_id=max_layer, alpha=0.0),
        lse=LSE(state_dim=26, early_depths=early_depths),
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=max_layer),
        profile=profile,
        candidate_layers=candidate_layers,
        early_depths=early_depths,
        full_depth=max_layer,
        image_size=image_size,
    )


def test_exit_at_12_executes_exactly_12_blocks_and_skips_deeper_maps(
    visualad_model: VisualAD,
    sample_image: torch.Tensor,
    block_call_counter: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    built_depths: list[int] = []
    original_build = CheckpointMapGenerator.build

    def tracked_build(self, depth, outputs):
        built_depths.append(int(depth))
        return original_build(self, depth, outputs)

    monkeypatch.setattr(CheckpointMapGenerator, "build", tracked_build)

    engine = _build_engine(visualad_model, profile=_always_exit_profile())
    # Force LSE to predict near-zero gain so aggressive always-exit profile exits
    with torch.no_grad():
        for p in engine.lse.parameters():
            p.zero_()

    result = engine.infer(sample_image)

    assert block_call_counter.total == 12
    assert result.selected_depth == 12
    assert 18 not in result.checkpoint_trace
    assert 24 not in result.checkpoint_trace
    assert 18 not in built_depths
    assert 24 not in built_depths
    assert max(built_depths) == 12


def test_continue_reuses_cached_sequence_not_restart(
    visualad_model: VisualAD,
    sample_image: torch.Tensor,
    block_call_counter: SimpleNamespace,
):
    """Continue past 12 then exit at 18 must execute 18 blocks total, not 12+18."""
    # Exit only when depth>=18 by using a custom policy wrapper via high threshold
    # until we patch should_exit behavior through a profile that never exits on
    # first early depth: use never-exit profile but force exit by replacing LSE
    # after depth 12... simpler: monkeypatch should_exit.
    calls = {"n": 0}

    def exit_from_second_checkpoint(prediction, signals, profile):
        calls["n"] += 1
        return calls["n"] >= 2  # first early depth continue, second exit

    engine = _build_engine(visualad_model, profile=_never_exit_until_full_profile())
    # Patch module-level should_exit used by engine
    import rad.inference.adaptive_engine as mod

    original = mod.should_exit
    mod.should_exit = exit_from_second_checkpoint
    try:
        result = engine.infer(sample_image)
    finally:
        mod.should_exit = original

    assert result.selected_depth == 18
    assert block_call_counter.total == 18
    assert result.checkpoint_trace == [6, 12, 18]


def test_forced_full_depth_matches_dynamic_fusion_only(
    visualad_model: VisualAD,
    sample_image: torch.Tensor,
):
    engine = _build_engine(visualad_model, profile=_always_exit_profile())
    with torch.no_grad():
        forced = engine.infer(sample_image, force_full_depth=True)
        fusion_only = engine.fuse_full_depth(sample_image)

    assert forced.selected_depth == 24
    assert fusion_only.selected_depth == 24
    assert torch.allclose(forced.final_map, fusion_only.final_map, atol=1e-5, rtol=1e-5)
    assert torch.allclose(forced.image_score, fusion_only.image_score, atol=1e-5, rtol=1e-5)


def test_timing_breakdown_keys_present_with_cuda_sync_hooks(
    visualad_model: VisualAD,
    sample_image: torch.Tensor,
):
    engine = _build_engine(visualad_model, profile=_always_exit_profile())
    with torch.no_grad():
        for p in engine.lse.parameters():
            p.zero_()
        result = engine.infer(sample_image, measure_timing=True)

    keys = set(result.timing_breakdown)
    assert "backbone" in keys
    assert "maps" in keys
    assert "dlcm" in keys
    assert "lse" in keys
    assert "total" in keys
