from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_VISUALAD_PATH = Path(__file__).resolve().parents[2] / "VisualAD_lib" / "VisualAD.py"
_SPEC = importlib.util.spec_from_file_location("visualad_core", _VISUALAD_PATH)
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
    return torch.randn(2, 3, 32, 32)


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


def test_staged_outputs_match_legacy_outputs(visualad_model: VisualAD, sample_image: torch.Tensor):
    visualad_model.eval()
    with torch.no_grad():
        legacy = visualad_model.encode_image(sample_image, [6, 12, 18, 24])
        staged = visualad_model.visual.forward_staged(sample_image, [6, 12, 18, 24])
    legacy_patches = legacy["patch_tokens"]
    for idx, depth in enumerate([6, 12, 18, 24]):
        assert torch.allclose(
            staged[depth].patch_tokens,
            legacy_patches[idx][:, 3:, :],
            atol=1e-6,
            rtol=1e-5,
        )


def test_exit_at_12_executes_exactly_12_blocks(
    visualad_model: VisualAD,
    sample_image: torch.Tensor,
    block_call_counter: SimpleNamespace,
):
    cache = visualad_model.visual.prepare_stage(sample_image)
    visualad_model.visual.run_to(cache, 12)
    assert block_call_counter.total == 12


def test_continue_12_to_18_matches_direct_18(visualad_model: VisualAD, sample_image: torch.Tensor):
    cache = visualad_model.visual.prepare_stage(sample_image)
    _out12, cache12 = visualad_model.visual.run_to(cache, 12)
    out18, _ = visualad_model.visual.run_to(cache12, 18)
    direct = visualad_model.visual.forward_staged(sample_image, [18])[18]
    assert torch.allclose(out18.patch_tokens, direct.patch_tokens, atol=1e-6, rtol=1e-5)
