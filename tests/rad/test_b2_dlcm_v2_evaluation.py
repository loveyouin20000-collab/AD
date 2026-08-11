"""RED/GREEN tests for B2-05C1 V2 evaluation gates and identities."""

from __future__ import annotations

import pytest
import torch

from rad.phase_b import b2_dlcm_v2_evaluation as subject
from rad.phase_b import b2_dlcm_v2_protocol as protocol


def _passing_gate_kwargs() -> dict:
    return {
        "depth24_gt_kl_macro": 0.10,
        "depth24_uniform_gt_kl_macro": 0.20,
        "per_category_gt_kl": {"bottle": 0.11, "carpet": 0.12},
        "per_category_uniform_gt_kl": {"bottle": 0.20, "carpet": 0.20},
        "delta_pixel_ap_macro": 0.01,
        "delta_pixel_auroc_macro": 0.0,
        "delta_aupro_macro": 0.0,
        "per_category_localization": {
            "bottle": {
                "delta_pixel_ap": 0.0,
                "delta_pixel_auroc": 0.0,
                "delta_aupro": 0.0,
            },
            "carpet": {
                "delta_pixel_ap": 0.0,
                "delta_pixel_auroc": 0.0,
                "delta_aupro": 0.0,
            },
        },
    }


def test_development_gates_pass_and_fail() -> None:
    ok = subject.evaluate_development_gates(**_passing_gate_kwargs())
    assert ok["passed"] is True
    subject.require_development_go(ok)

    bad_kwargs = _passing_gate_kwargs()
    bad_kwargs["depth24_gt_kl_macro"] = 0.199999
    bad = subject.evaluate_development_gates(**bad_kwargs)
    assert bad["passed"] is False
    with pytest.raises(subject.B2DLCMV2EvaluationError, match="B2_DLCM_DEVELOPMENT_UNQUALIFIED"):
        subject.require_development_go(bad)


def test_development_failure_blocks_materialization_unlock() -> None:
    bad = subject.evaluate_development_gates(
        **{**_passing_gate_kwargs(), "delta_pixel_ap_macro": -0.01}
    )
    with pytest.raises(protocol.B2DLCMV2ProtocolError, match="B2_DLCM_DEVELOPMENT_UNQUALIFIED"):
        protocol.build_materialization_unlock(
            development_go=bool(bad["passed"]),
            development_evidence_sha256="a" * 64,
            implementation_commit="deadbeef",
        )


def test_aux_diagnostics_required_non_blocking() -> None:
    manifest = subject.build_auxiliary_diagnostics_manifest(
        diagnostics={
            "depth_24": {
                "teacher_alloc_kl": 0.1,
                "teacher_alloc_jsd": 0.05,
                "gt_signed_huber": 0.2,
            }
        },
        source_checkpoint_kind="canonical_best_training_checkpoint",
    )
    assert manifest["qualification_blocking"] is False
    assert manifest["not_available_from_deployment_artifact"] is True
    with pytest.raises(
        subject.B2DLCMV2EvaluationError, match="B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID"
    ):
        subject.build_auxiliary_diagnostics_manifest(
            diagnostics={"x": 1.0},
            source_checkpoint_kind="deployment_checkpoint",
        )
    with pytest.raises(
        subject.B2DLCMV2EvaluationError, match="B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID"
    ):
        subject.build_auxiliary_diagnostics_manifest(
            diagnostics={"x": float("nan")},
            source_checkpoint_kind="canonical_best_training_checkpoint",
        )


def test_reject_signed_proxy() -> None:
    weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    with pytest.raises(
        subject.B2DLCMV2EvaluationError, match="B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID"
    ):
        subject.reject_signed_proxy_from_deployment_weights(weights)


def test_h_decision_excludes_development_teacher_and_accepted_forbidden() -> None:
    gate = subject.evaluate_development_gates(**_passing_gate_kwargs())
    decision = subject.build_final_decision_from_metrics(
        gt_target_learning={"depth24_gt_kl_macro": 0.1},
        localization={"delta_pixel_ap_macro": 0.01},
        gate_result=gate,
    )
    assert "H_decision" in decision
    assert "development" not in str(decision["gt_target_learning"]).lower()
    fail_gate = {**gate, "passed": False}
    with pytest.raises(
        subject.B2DLCMV2EvaluationError, match="B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN"
    ):
        subject.finalize_accepted_or_forbid(
            final_gate=fail_gate,
            h_deploy="a" * 64,
            h_decision=decision["H_decision"],
            h_evidence="c" * 64,
            h_selection="d" * 64,
            upstream={},
            v2_contract_sha256="e" * 64,
        )
