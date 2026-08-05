"""RED/GREEN tests for B2-05C1 V2 model architecture and deployment extract."""

from __future__ import annotations

import torch

from rad.phase_b import b2_dlcm_v2 as subject
from rad.phase_b import b2_dlcm_v2_deployment as deploy


def test_four_heads_and_uniform_zero_init_allocation() -> None:
    model = subject.B2DLCMV2(seed=17)
    assert model.gt_deployment_head.out_features == 1
    assert model.teacher_allocation_head.out_features == 1
    assert model.gt_signed_head.out_features == 1
    assert model.teacher_signed_head.out_features == 1
    assert torch.all(model.gt_deployment_head.weight == 0)
    assert torch.all(model.gt_deployment_head.bias == 0)
    assert torch.all(model.teacher_allocation_head.weight == 0)
    assert torch.all(model.teacher_allocation_head.bias == 0)
    assert not torch.equal(model.gt_signed_head.weight, model.teacher_signed_head.weight)
    # Independent parameter tensors.
    assert model.gt_deployment_head.weight.data_ptr() != model.teacher_allocation_head.weight.data_ptr()
    model.eval()
    for depth, n in ((12, 2), (18, 3), (24, 4)):
        x = torch.zeros(1, n, 18)
        out = model.forward_training(x, prediction_depth=depth)
        expected = torch.full((1, n), 1.0 / n)
        assert torch.equal(out.gt_deployment_weights, expected)
        assert torch.equal(out.teacher_allocation_weights, expected)


def test_deployment_extract_drops_auxiliary_heads() -> None:
    model = subject.B2DLCMV2(seed=29)
    state = subject.extract_deployment_state_dict(model)
    for prefix in subject.AUXILIARY_HEAD_PREFIXES:
        assert not any(name.startswith(prefix) for name in state)
    assert any(name.startswith("gt_deployment_head") for name in state)
    assert "layer_encoder.block_1.linear.weight" in state or any(
        "layer_encoder" in name for name in state
    )


def test_production_wrapper_exposes_only_deploy_weights() -> None:
    from tests.rad.b2_dlcm_fixtures import fixture_normalization_artifact

    model = subject.B2DLCMV2(seed=43)
    ckpt = deploy.export_v2_deployment_checkpoint(
        model,
        normalization_stats=fixture_normalization_artifact(),
        contribution_target_collection_scientific_sha256="d" * 64,
    )
    assert ckpt["auxiliary_heads_present"] is False
    wrapper = deploy.load_v2_deployment_wrapper(ckpt)
    n = 2
    weights = wrapper.forward(
        torch.zeros(1, n, 18),
        depth=12,
        player_ids=[6, 12],
    )
    assert weights.shape == (1, n)
    try:
        wrapper.forward_diagnostic()
        raise AssertionError("expected auxiliary forbid")
    except deploy.B2DLCMV2DeploymentError as exc:
        assert exc.code == "B2_DLCM_V2_CONTRACT_MISMATCH"


def test_v1_history_identity_pins_immutable() -> None:
    pins = subject.v1_immutable_identity()
    assert pins["tag"] == "b2-dlcm-unqualified-evidence-v1"
    assert pins["commit"] == "43d856f5ff771957f9f39d0909b1bc87d6b7081b"
    assert pins["canonical_seed"] == 17
    assert pins["verdict"] == "localized_but_target_fidelity_unqualified"
    assert pins["accepted_training_plan"] == (
        "59e20f4cb337ef42384f70bb8b3dad5211d906341b0a2d41f7e6847610635980"
    )
    assert pins["seed_collection"] == (
        "94a6a9332a0694889c7a0255814ac13fe8316c601529197063165ce14ec1277f"
    )
    assert pins["deployment_scientific_sha256"] == (
        "4cbc6fb88f39ed86deacfbbe48580f7682453b94becb046ec6ef1b1302df378a"
    )
    assert pins["evaluation_unlock"] == (
        "19dca41e9f647d12afce9877a7340f5af58bf9a23997d7339dded26d89fe73dd"
    )
    assert pins["qualification_scientific_sha256"] == (
        "da51e5fc1302cf507bc844f87e82cb66f7d2fa0a13e61f28a0dba14333201c49"
    )


def test_architecture_constants() -> None:
    model = subject.B2DLCMV2(seed=17)
    assert model.candidate_layers == (6, 12, 18, 24)
    assert model.prediction_depths == (12, 18, 24)
    assert model.descriptor_dimension == 18
    assert model.hidden_dimension == 64
    assert model.players_for_depth(12) == (6, 12)
    assert model.players_for_depth(18) == (6, 12, 18)
    assert model.players_for_depth(24) == (6, 12, 18, 24)
