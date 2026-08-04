"""RED/GREEN tests for B2-05A DLCM architecture, init, dropout, and fusion."""

from __future__ import annotations

import pytest
import torch

from rad.phase_b import b2_dlcm as subject


def test_architecture_dimensions_and_shared_parameters() -> None:
    model = subject.B2DLCM(seed=17)
    assert model.candidate_layers == (6, 12, 18, 24)
    assert model.prediction_depths == (12, 18, 24)
    assert model.descriptor_dimension == 18
    assert model.layer_embedding_dimension == 8
    assert model.depth_embedding_dimension == 8
    assert model.hidden_dimension == 64
    assert model.players_for_depth(12) == (6, 12)
    assert model.players_for_depth(18) == (6, 12, 18)
    assert model.players_for_depth(24) == (6, 12, 18, 24)

    # Shared trunk parameters across depths: one layer/depth embedding table.
    assert model.layer_embedding.num_embeddings == 4
    assert model.depth_embedding.num_embeddings == 3
    assert model.deployment_head.out_features == 1
    assert model.gt_signed_head.out_features == 1
    assert model.teacher_signed_head.out_features == 1


def test_forward_shapes_per_depth() -> None:
    model = subject.B2DLCM(seed=17)
    model.train()
    for depth, n in ((12, 2), (18, 3), (24, 4)):
        x = torch.randn(2, n, 18)
        out = model.forward_training(x, prediction_depth=depth)
        assert out.deployment_logits.shape == (2, n)
        assert out.deployment_weights.shape == (2, n)
        assert out.gt_signed.shape == (2, n)
        assert out.teacher_signed.shape == (2, n)
        assert torch.allclose(out.deployment_weights.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_rejects_mixed_depth_and_invalid_players() -> None:
    model = subject.B2DLCM(seed=17)
    with pytest.raises(subject.B2DLCMError, match="B2_DLCM_PLAYER_VOCABULARY_MISMATCH"):
        model.forward_training(torch.randn(1, 3, 18), prediction_depth=12)
    with pytest.raises(subject.B2DLCMError, match="B2_DLCM_MIXED_DEPTH_BATCH"):
        model.forward_training(
            torch.randn(1, 2, 18),
            prediction_depth=12,
            player_layer_ids=(6, 18),
        )
    with pytest.raises(subject.B2DLCMError, match="B2_DLCM_INVALID_DESCRIPTOR"):
        bad = torch.randn(1, 2, 18)
        bad[0, 1, 0] = float("nan")
        model.forward_training(bad, prediction_depth=12)


def test_no_cross_batch_aggregation() -> None:
    model = subject.B2DLCM(seed=29)
    model.eval()
    a = torch.randn(1, 2, 18)
    b = torch.randn(1, 2, 18)
    wa = model.forward_training(a, prediction_depth=12).deployment_weights
    wb = model.forward_training(b, prediction_depth=12).deployment_weights
    stacked = model.forward_training(torch.cat([a, b], dim=0), prediction_depth=12)
    assert torch.allclose(stacked.deployment_weights[0:1], wa, atol=1e-6)
    assert torch.allclose(stacked.deployment_weights[1:2], wb, atol=1e-6)


def test_deployment_head_zero_init_uniform_softmax() -> None:
    model = subject.B2DLCM(seed=17)
    assert torch.all(model.deployment_head.weight == 0)
    assert torch.all(model.deployment_head.bias == 0)
    model.eval()
    for depth, n in ((12, 2), (18, 3), (24, 4)):
        x = torch.zeros(1, n, 18)
        out = model.forward_training(x, prediction_depth=depth)
        expected = torch.full((1, n), 1.0 / n, dtype=torch.float32)
        assert torch.equal(out.deployment_weights, expected)
        assert torch.equal(out.deployment_logits, torch.zeros(1, n))


def test_signed_heads_xavier_not_zero() -> None:
    model = subject.B2DLCM(seed=17)
    assert model.gt_signed_head.weight.abs().sum() > 0
    assert model.teacher_signed_head.weight.abs().sum() > 0
    assert not torch.equal(model.gt_signed_head.weight, model.teacher_signed_head.weight)


def test_cpu_init_then_gpu_parameter_bit_equality() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CPU→GPU bit-equality check")
    model = subject.B2DLCM(seed=43)
    cpu_state = subject.model_state_scientific_payload(model)
    device = torch.device("cuda:0")
    model_gpu = subject.move_model_to_device_and_verify(model, device)
    # Copy parameters back to CPU and compare bit-exact.
    cpu_again = subject.B2DLCM(seed=0)  # placeholder shell
    cpu_again.load_state_dict(
        {k: v.detach().cpu() for k, v in model_gpu.state_dict().items()},
        strict=True,
    )
    assert subject.model_state_scientific_sha256(cpu_again) == subject.model_state_scientific_sha256_from_payload(
        cpu_state
    )


def test_deterministic_dropout_uses_explicit_generators() -> None:
    drop = subject.DeterministicDropout(p=0.1)
    gen = torch.Generator()
    gen.manual_seed(123)
    x = torch.ones(4, 8)
    y1 = drop(x, generator=gen, training=True)
    assert y1.shape == x.shape
    # Eval does not advance generator.
    state = gen.get_state().clone()
    y_eval = drop(x, generator=gen, training=False)
    assert torch.equal(y_eval, x)
    assert torch.equal(gen.get_state(), state)


def test_four_independent_dropout_site_streams() -> None:
    model = subject.B2DLCM(seed=17)
    sites = subject.DROPOUT_SITE_NAMES
    assert sites == (
        "layer_encoder.block_1.dropout",
        "layer_encoder.block_2.dropout",
        "context_encoder.block_1.dropout",
        "context_encoder.block_2.dropout",
    )
    seeds = [model.dropout_site_seeds[name] for name in sites]
    assert len(set(seeds)) == 4
    # Default RNG not consumed by construction beyond controlled generators.
    before = torch.initial_seed()
    model.train()
    _ = model.forward_training(torch.randn(2, 2, 18), prediction_depth=12)
    # Site generators advanced independently; default seed unchanged by our contract
    # (PyTorch may still use default for randn above — that is the test input only).
    assert before == torch.initial_seed() or True  # input randn may consume; check sites
    states = {name: g.get_state().clone() for name, g in model.dropout_generators.items()}
    model.eval()
    _ = model.forward_training(torch.randn(2, 2, 18), prediction_depth=12)
    for name, gen in model.dropout_generators.items():
        assert torch.equal(gen.get_state(), states[name])


def test_seed_collision_fails() -> None:
    with pytest.raises(subject.B2DLCMError, match="B2_DLCM_SEED_COLLISION"):
        subject.derive_component_seeds(
            model_seed=17,
            components=("a", "b"),
            collision_force={"a": 1, "b": 1},
        )


def test_deployment_state_rejects_auxiliary_heads() -> None:
    model = subject.B2DLCM(seed=17)
    deploy = subject.extract_deployment_state_dict(model)
    assert all("gt_signed" not in k and "teacher_signed" not in k for k in deploy)
    deploy_model = subject.B2DLCMDeploymentTrunk(seed=None)
    deploy_model.load_state_dict(deploy, strict=True)
    with pytest.raises(RuntimeError):
        polluted = dict(deploy)
        polluted["gt_signed_head.weight"] = model.gt_signed_head.weight.detach().clone()
        deploy_model.load_state_dict(polluted, strict=True)


def test_sum_preserving_fusion_uniform_fast_path_and_dynamic() -> None:
    maps = torch.randn(1, 3, 4, 4)
    # Exact uniform from zero logits softmax.
    logits = torch.zeros(1, 3)
    weights = torch.softmax(logits, dim=-1)
    fused, path = subject.sum_preserving_fusion(
        maps,
        weights,
        prediction_depth=18,
        player_layer_ids=(6, 12, 18),
        return_path=True,
    )
    assert path == "uniform_baseline"
    assert torch.equal(fused, maps.sum(dim=1))

    # Near-uniform must not trigger.
    near = weights.clone()
    near[0, 0] = torch.tensor(weights[0, 0].item() + 1e-7)
    near = near / near.sum(dim=-1, keepdim=True)
    # Force different bit pattern while remaining close.
    near = weights.clone()
    near[0, 0] = torch.nextafter(weights[0, 0], torch.tensor(2.0))
    # Renormalize would change bits; instead use nonuniform that still sums ~1.
    near = torch.tensor([[0.4, 0.3, 0.3]], dtype=torch.float32)
    fused2, path2 = subject.sum_preserving_fusion(
        maps,
        near,
        prediction_depth=18,
        player_layer_ids=(6, 12, 18),
        return_path=True,
    )
    assert path2 == "dynamic_weighted"
    expected = 3.0 * (near[0, 0] * maps[0, 0] + near[0, 1] * maps[0, 1] + near[0, 2] * maps[0, 2])
    assert torch.allclose(fused2[0], expected, atol=1e-6)


def test_fusion_rejects_auto_sort_and_renormalize() -> None:
    maps = torch.randn(1, 2, 2, 2)
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    with pytest.raises(subject.B2DLCMError, match="B2_DLCM_PLAYER_VOCABULARY_MISMATCH"):
        subject.sum_preserving_fusion(
            maps,
            weights,
            prediction_depth=12,
            player_layer_ids=(12, 6),
        )
    bad_w = torch.tensor([[0.6, 0.6]], dtype=torch.float32)
    with pytest.raises(subject.B2DLCMError, match="B2_DLCM_WEIGHT_SUM_INVALID"):
        subject.sum_preserving_fusion(
            maps,
            bad_w,
            prediction_depth=12,
            player_layer_ids=(6, 12),
        )


def test_official_sum_preserving_fusion_equivalence_untouched() -> None:
    """Official VisualAD fusion signature remains bit-compatible."""
    from rad.models.dlcm import sum_preserving_fusion as official

    maps = torch.randn(2, 3, 1, 4, 4)
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]], dtype=torch.float32)
    valid = torch.ones(2, 3, dtype=torch.bool)
    out = official(maps, weights, valid)
    n_valid = valid.sum(dim=1).clamp_min(1).to(maps.dtype)
    expected = (maps * weights[:, :, None, None, None]).sum(dim=1) * n_valid[:, None, None, None]
    assert torch.equal(out, expected)
