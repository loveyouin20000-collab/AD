"""RED→GREEN: deployment allocation weights must not proxy signed Shapley diagnostics."""

from __future__ import annotations

import pytest
import torch

from rad.phase_b import b2_dlcm as dlcm
from rad.phase_b import b2_dlcm_deployment as deployment
from rad.phase_b import b2_dlcm_evaluation as subject


def test_signed_huber_rejects_allocation_weight_proxy() -> None:
    weights = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    phi = torch.tensor([1.0, 0.5, -0.5, -1.0], dtype=torch.float32)
    with pytest.raises(subject.B2DLCMSignedDiagnosticError, match="allocation.?weight|cannot.*substitut"):
        subject.signed_huber_diagnostic(weights, phi)


def test_deployment_only_reports_signed_metrics_unavailable() -> None:
    model = dlcm.B2DLCM(seed=17)
    trunk_sd = dlcm.extract_deployment_state_dict(model)
    trunk = dlcm.B2DLCMDeploymentTrunk(seed=None, initialize=False)
    trunk.load_state_dict(trunk_sd, strict=True)
    trunk.eval()

    n = 2
    phi_gt = torch.linspace(-1.0, 1.0, n)
    phi_t = phi_gt.flip(0)
    report = subject.compute_signed_diagnostics_report(
        model=trunk,
        descriptors=torch.randn(n, 18),
        prediction_depth=12,
        player_layer_ids=(6, 12),
        phi_gt=phi_gt,
        phi_t=phi_t,
        artifact_kind="deployment",
    )
    for key in ("huber_gt", "huber_teacher", "signed_spearman_gt", "signed_spearman_teacher",
                "signed_pairwise_ranking_gt", "signed_pairwise_ranking_teacher"):
        entry = report[key]
        assert entry["status"] == "not_available_in_deployment_artifact"
        assert entry["reason"] == "training_only_auxiliary_heads_removed"
        assert "value" not in entry or entry["value"] is None
    assert report.get("signed_diagnostics_available") is False


def test_full_training_checkpoint_uses_correct_auxiliary_heads() -> None:
    model = dlcm.B2DLCM(seed=17)
    model.eval()
    n = 2
    desc = torch.randn(n, 18)
    phi_gt = torch.tensor([1.0, -1.0])
    phi_t = torch.tensor([-0.5, 0.5])
    with torch.no_grad():
        out = model.forward_training(desc.unsqueeze(0), prediction_depth=12, player_layer_ids=(6, 12))
    report = subject.compute_signed_diagnostics_report(
        model=model,
        descriptors=desc,
        prediction_depth=12,
        player_layer_ids=(6, 12),
        phi_gt=phi_gt,
        phi_t=phi_t,
        artifact_kind="training",
        diagnostic_source="canonical_best_training_checkpoint",
    )
    assert report["diagnostic_source"] == "canonical_best_training_checkpoint"
    assert report["not_part_of_deployment_artifact"] is True
    assert report["huber_gt"]["status"] == "ok"
    assert report["huber_teacher"]["status"] == "ok"
    expected_gt = subject.signed_huber_diagnostic(out.gt_signed.reshape(-1), phi_gt)
    expected_t = subject.signed_huber_diagnostic(out.teacher_signed.reshape(-1), phi_t)
    assert report["huber_gt"]["value"] == pytest.approx(expected_gt)
    assert report["huber_teacher"]["value"] == pytest.approx(expected_t)


def test_gt_and_teacher_auxiliary_heads_remain_separate() -> None:
    model = dlcm.B2DLCM(seed=29)
    model.eval()
    desc = torch.randn(2, 18)
    phi_gt = torch.tensor([1.0, -1.0])
    phi_t = torch.tensor([-1.0, 1.0])
    report = subject.compute_signed_diagnostics_report(
        model=model,
        descriptors=desc,
        prediction_depth=12,
        player_layer_ids=(6, 12),
        phi_gt=phi_gt,
        phi_t=phi_t,
        artifact_kind="training",
    )
    # Heads produce different signed predictions → different Huber vs swapped targets.
    assert report["huber_gt"]["value"] != report["huber_teacher"]["value"]
    with torch.no_grad():
        out = model.forward_training(desc.unsqueeze(0), prediction_depth=12, player_layer_ids=(6, 12))
    assert not torch.allclose(out.gt_signed, out.teacher_signed)


def test_changing_deployment_weights_does_not_change_signed_when_aux_fixed() -> None:
    model = dlcm.B2DLCM(seed=43)
    model.eval()
    desc = torch.randn(2, 18)
    phi_gt = torch.linspace(-1.0, 1.0, 2)
    phi_t = phi_gt.flip(0)
    base = subject.compute_signed_diagnostics_report(
        model=model,
        descriptors=desc,
        prediction_depth=12,
        player_layer_ids=(6, 12),
        phi_gt=phi_gt,
        phi_t=phi_t,
        artifact_kind="training",
    )
    with torch.no_grad():
        model.deployment_head.weight.add_(10.0)
        model.deployment_head.bias.add_(5.0)
    after = subject.compute_signed_diagnostics_report(
        model=model,
        descriptors=desc,
        prediction_depth=12,
        player_layer_ids=(6, 12),
        phi_gt=phi_gt,
        phi_t=phi_t,
        artifact_kind="training",
    )
    assert after["huber_gt"]["value"] == pytest.approx(base["huber_gt"]["value"])
    assert after["huber_teacher"]["value"] == pytest.approx(base["huber_teacher"]["value"])
    assert after["signed_spearman_gt"]["value"] == pytest.approx(base["signed_spearman_gt"]["value"])


def test_missing_auxiliary_heads_cannot_silently_pass_signed_validity() -> None:
    model = dlcm.B2DLCM(seed=17)
    trunk_sd = dlcm.extract_deployment_state_dict(model)
    trunk = dlcm.B2DLCMDeploymentTrunk(seed=None, initialize=False)
    trunk.load_state_dict(trunk_sd, strict=True)
    report = subject.compute_signed_diagnostics_report(
        model=trunk,
        descriptors=torch.randn(2, 18),
        prediction_depth=12,
        player_layer_ids=(6, 12),
        phi_gt=torch.linspace(-1, 1, 2),
        phi_t=torch.linspace(1, -1, 2),
        artifact_kind="deployment",
    )
    validity = subject.signed_metrics_validity_check(report)
    assert validity["signed_diagnostics_available"] is False
    assert validity["passed"] is False
    assert validity["status"] == "not_available_in_deployment_artifact"


def test_qualification_status_unchanged_signed_not_hard_gate() -> None:
    """Signed diagnostics are diagnostic-only; KL/loc gates alone set qualification."""

    per_cat = {
        "bottle": {
            "gt": 0.18,
            "gt_uniform": 0.27,
            "teacher": 0.04,
            "teacher_uniform": 0.01,
        },
        "carpet": {
            "gt": 0.035,
            "gt_uniform": 0.042,
            "teacher": 0.097,
            "teacher_uniform": 0.095,
        },
    }
    loc = {
        "bottle": {"delta_pixel_ap": 0.004, "delta_pixel_auroc": 0.0005, "delta_aupro": 0.002},
        "carpet": {"delta_pixel_ap": 0.004, "delta_pixel_auroc": 0.0001, "delta_aupro": 0.0},
    }
    evidence = deployment.FormalLocalizationGateEvidence(
        metric_source_identity="b2_dlcm_formal_localization_v1",
        delta_pixel_ap_macro=0.004,
        delta_pixel_auroc_macro=0.0003,
        delta_aupro_macro=0.001,
        per_category_localization=loc,
    )
    gates = deployment.evaluate_qualification_gates(
        kl_dlcm_gt_macro=0.110,
        kl_uniform_gt_macro=0.160,
        kl_dlcm_teacher_macro=0.069,
        kl_uniform_teacher_macro=0.053,
        per_category_kl=per_cat,
        delta_pixel_ap_macro=0.004,
        delta_pixel_auroc_macro=0.0003,
        delta_aupro_macro=0.001,
        per_category_localization=loc,
        best_epoch=229,
        localization_evidence=evidence,
    )
    assert gates["state"] == "localized_but_target_fidelity_unqualified"
    assert gates["deployment_qualified"] is False
    # Signed fields are not inputs to evaluate_qualification_gates.
    assert "signed" not in str(gates.get("reasons", [])).lower()


def test_evaluate_rows_do_not_embed_weight_proxy_huber() -> None:
    """Row builder must refuse weight→signed substitution even if caller errs."""

    weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
    phi = torch.tensor([1.0, -1.0], dtype=torch.float32)
    with pytest.raises(subject.B2DLCMSignedDiagnosticError):
        subject.build_target_fidelity_row_signed_fields(
            signed_pred_gt=weights,
            signed_pred_teacher=weights,
            phi_gt=phi,
            phi_t=phi,
            allow_allocation_proxy=False,
        )
