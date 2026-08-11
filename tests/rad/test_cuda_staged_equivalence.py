"""Phase B1: staged-backbone CUDA numerical-equivalence qualification tests."""

from __future__ import annotations

# mypy: disable-error-code=no-untyped-def,valid-type,attr-defined
import importlib.util
from pathlib import Path

import pytest
import torch

from tests.rad.b1_cuda_helpers import (
    B1_ACCEPTED_CHECKPOINT,
    B1_ACCEPTED_CHECKPOINT_SHA256,
    B1_ATOL,
    B1_RTOL,
    B1_SEED,
    DEFAULT_CANDIDATE_LAYERS,
    apply_deterministic_cuda_settings,
    build_official_full_depth_outputs,
    build_staged_full_depth_outputs,
    deterministic_synthetic,
    diagnose_four_path_divergence,
    install_block_counter,
    load_preprocessed_image,
    load_teacher_production,
    reset_block_counter,
    run_deterministic_cuda_control,
    run_operational_noise_envelope,
    run_same_chain_gate,
    tensor_diff,
    tensor_layout_info,
    validate_checkpoint,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for B1")

_VISUALAD_PATH = Path(__file__).resolve().parents[2] / "VisualAD_lib" / "VisualAD.py"
_SPEC = importlib.util.spec_from_file_location("visualad_core_b1", _VISUALAD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
VisualAD = _MODULE.VisualAD


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    device = torch.device("cuda:0")
    apply_deterministic_cuda_settings(B1_SEED)
    return device


@pytest.fixture(scope="module")
def accepted_checkpoint() -> Path:
    if not B1_ACCEPTED_CHECKPOINT.is_file():
        pytest.skip(f"B1 accepted checkpoint unavailable: {B1_ACCEPTED_CHECKPOINT}")
    return validate_checkpoint(B1_ACCEPTED_CHECKPOINT, B1_ACCEPTED_CHECKPOINT_SHA256)


@pytest.fixture(scope="module")
def teacher_bundle(accepted_checkpoint: Path, cuda_device: torch.device):
    return load_teacher_production(
        accepted_checkpoint,
        B1_ACCEPTED_CHECKPOINT_SHA256,
        cuda_device,
    )


@pytest.fixture(scope="module")
def real_image(cuda_device: torch.device) -> torch.Tensor:
    return load_preprocessed_image(
        "/root/autodl-tmp/data/mvtec/bottle/test/good/000.png",
        cuda_device,
    )


@pytest.fixture(scope="module")
def synthetic_image(cuda_device: torch.device) -> torch.Tensor:
    return deterministic_synthetic(cuda_device)


@pytest.fixture
def tiny_cuda_model(cuda_device: torch.device) -> VisualAD:
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
    return model.to(cuda_device)


def test_production_loader_smoke(accepted_checkpoint: Path, cuda_device: torch.device):
    bundle = load_teacher_production(
        accepted_checkpoint,
        B1_ACCEPTED_CHECKPOINT_SHA256,
        cuda_device,
    )
    assert bundle.model is not None


def test_deterministic_cuda_control_documents_decision(
    teacher_bundle,
    real_image: torch.Tensor,
):
    result = run_deterministic_cuda_control(teacher_bundle, real_image)
    assert result.decision in {"A", "B", "C"}
    assert "A1_vs_A2" in result.comparisons
    assert "S1_vs_S2" in result.comparisons
    assert "A1_vs_S1" in result.comparisons
    assert "A2_vs_S2" in result.comparisons


def test_same_chain_gate_algorithmic_equivalence(teacher_bundle, real_image: torch.Tensor):
    gate = run_same_chain_gate(teacher_bundle, real_image)
    assert gate.status == "passed", gate.errors
    for key, diff in gate.feature_diffs.items():
        assert diff.max_abs <= B1_ATOL, f"{key} max_abs={diff.max_abs}"
    for key, diff in gate.map_diffs.items():
        assert diff.max_abs <= B1_ATOL, f"{key} max_abs={diff.max_abs}"
    assert gate.continuation_live_tensor_preserved is True


def test_operational_noise_envelope_gate(teacher_bundle, real_image: torch.Tensor):
    gate = run_operational_noise_envelope(
        teacher_bundle,
        [("mvtec_sample_000", real_image)],
        repeats=5,
    )
    assert gate.ratio_pass, gate.errors
    assert gate.excess_pass, gate.errors
    assert gate.status == "passed"


def test_full_depth_maps_same_chain_style(teacher_bundle, real_image: torch.Tensor):
    """Maps compared via independent forwards; Gate 1 covers strict same-chain."""
    official = build_official_full_depth_outputs(teacher_bundle, real_image)
    staged = build_staged_full_depth_outputs(teacher_bundle, real_image)
    for layer in DEFAULT_CANDIDATE_LAYERS:
        diff = tensor_diff(staged["layer_maps"][layer], official["layer_maps"][layer])
        assert diff.max_abs <= 5e-2, f"layer map {layer}: {diff.max_abs}"


def test_exit_block_counts_real_teacher(teacher_bundle, real_image: torch.Tensor):
    visual = teacher_bundle.model.visual
    counter = install_block_counter(visual)
    cache = visual.prepare_stage(real_image)
    for depth, expected in ((12, 12), (18, 18), (24, 24)):
        reset_block_counter(counter)
        cache = visual.prepare_stage(real_image)
        visual.run_to(cache, depth)
        assert counter.total == expected, f"exit {depth}: got {counter.total}"


def test_continuation_12_to_18_real_teacher(teacher_bundle, real_image: torch.Tensor):
    visual = teacher_bundle.model.visual
    counter = install_block_counter(visual)
    cache = visual.prepare_stage(real_image)
    _, cache12 = visual.run_to(cache, 12)
    assert counter.total == 12
    reset_block_counter(counter)
    visual.run_to(cache12, 18)
    assert counter.total == 6
    assert counter.per_call == list(range(13, 19))


def test_continuation_18_to_24_real_teacher(teacher_bundle, real_image: torch.Tensor):
    visual = teacher_bundle.model.visual
    counter = install_block_counter(visual)
    cache = visual.prepare_stage(real_image)
    _, cache18 = visual.run_to(cache, 18)
    reset_block_counter(counter)
    visual.run_to(cache18, 24)
    assert counter.total == 6
    assert counter.per_call == list(range(19, 25))


def test_continuation_does_not_recompute_blocks_1_to_12(
    teacher_bundle,
    real_image: torch.Tensor,
):
    visual = teacher_bundle.model.visual
    counter = install_block_counter(visual)
    cache = visual.prepare_stage(real_image)
    _, cache12 = visual.run_to(cache, 12)
    first_blocks = list(counter.per_call)
    assert first_blocks == list(range(1, 13))
    with pytest.raises(ValueError, match="before cache.next_block"):
        visual.run_to(cache12, 12)


def test_nonstandard_candidate_layers_synthetic_cuda(
    tiny_cuda_model: VisualAD,
    cuda_device: torch.device,
):
    layers = [2, 4, 6, 8]
    image = torch.randn(1, 3, 32, 32, device=cuda_device)
    with torch.no_grad():
        legacy = tiny_cuda_model.encode_image(image, layers)
        staged = tiny_cuda_model.visual.forward_staged(image, layers)
    for idx, depth in enumerate(layers):
        official = legacy["patch_tokens"][idx][:, legacy["patch_start_idx"] :, :]
        assert torch.allclose(
            staged[depth].patch_tokens,
            official,
            atol=B1_ATOL,
            rtol=B1_RTOL,
        )


def test_continuation_tensor_layout_preserved_across_stop_resume(
    teacher_bundle,
    real_image: torch.Tensor,
):
    visual = teacher_bundle.model.visual
    with torch.no_grad():
        cache = visual.prepare_stage(real_image)
        layout_prepare = tensor_layout_info(cache.sequence)
        _, cache6 = visual.run_to(cache, 6)
        layout6 = tensor_layout_info(cache6.sequence)
        _, cache12_seg = visual.run_to(cache6, 12)
        layout12_seg = tensor_layout_info(cache12_seg.sequence)

        cache_direct = visual.prepare_stage(real_image)
        _, cache12_direct = visual.run_to(cache_direct, 12)
        layout12_direct = tensor_layout_info(cache12_direct.sequence)

    assert layout_prepare["contiguous"] is True
    assert layout6["contiguous"] is True
    assert layout12_seg["contiguous"] is True
    assert layout12_direct["contiguous"] is True
    assert layout12_seg["strides"] == layout12_direct["strides"]
    assert layout12_seg["storage_offset"] == layout12_direct["storage_offset"]


def test_same_chain_clone_snapshot_matches_live_ln_post(
    teacher_bundle,
    real_image: torch.Tensor,
):
    visual = teacher_bundle.model.visual
    with torch.inference_mode():
        sequence = visual._embed_image(real_image).permute(1, 0, 2)
        live = sequence
        for block in visual.transformer.resblocks[:12]:
            live = block(live)
        snap = live.clone()
        from tests.rad.b1_cuda_helpers import _ln_post_patch_tokens

        live_patch = _ln_post_patch_tokens(visual, live)
        snap_patch = _ln_post_patch_tokens(visual, snap)
    diff = tensor_diff(live_patch, snap_patch)
    assert diff.max_abs == 0.0


def test_four_path_diagnosis_localizes_first_divergence(
    teacher_bundle,
    real_image: torch.Tensor,
):
    diagnosis = diagnose_four_path_divergence(
        teacher_bundle,
        real_image,
        sample_id="mvtec_sample_000",
        max_block=12,
    )
    assert diagnosis["root_cause_classification"] in {
        "case_b_attention_backend_or_official_clone_path",
        "case_c_cuda_kernel_dispatch_non_determinism",
        "case_a_checkpoint_state_capture_resume",
        "case_d_unlocalized",
    }
    divergent_rows = [
        row
        for row in diagnosis["comparison_summary_table"]
        if row["first_divergent_block"] is not None
    ]
    if divergent_rows:
        assert divergent_rows[0]["comparison"] == "A_vs_B"
        assert divergent_rows[0]["first_divergent_block"] >= 7
    assert diagnosis["same_chain_control"]["clone_vs_live_ln_post"]["max_abs"] == 0.0
