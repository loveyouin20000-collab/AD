"""Tests for B2-05A deployment export, loader, gates, and fusion paths."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from rad.phase_b import b2_dlcm as model_mod
from rad.phase_b import b2_dlcm_deployment as subject
from tests.rad.b2_dlcm_fixtures import fixture_normalization_artifact


def _toy_checkpoint() -> dict:
    model = model_mod.B2DLCM(seed=17)
    norm = fixture_normalization_artifact()
    return subject.export_deployment_checkpoint(
        training_model=model,
        normalization=norm,
        canonical_seed=17,
        source_original_best_identity="aa" * 32,
        source_reproduction_best_identity="aa" * 32,
        contribution_target_collection_scientific_sha256="bb" * 32,
    )


def test_export_strips_auxiliary_and_binds_golden() -> None:
    ckpt = _toy_checkpoint()
    assert all("gt_signed" not in k and "teacher_signed" not in k for k in ckpt["state_dict"])
    assert "optimizer" not in ckpt
    assert len(ckpt["golden_cases"]) == 9
    assert ckpt["deployment_scientific_sha256"] == subject.deployment_scientific_sha256(ckpt)
    subject.run_cpu_golden_self_test(ckpt)


def test_immutable_wrapper_rejects_mutation() -> None:
    ckpt = _toy_checkpoint()
    wrapper = subject.load_qualified_deployment(
        ckpt,
        checkpoint_file_sha256="cc" * 32,
        environment_contract_sha256="dd" * 32,
        device=torch.device("cpu"),
    )
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_IMMUTABLE_TRAIN"):
        wrapper.train(True)
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_IMMUTABLE_LOAD"):
        wrapper.load_state_dict({})
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_IMMUTABLE_DEVICE"):
        wrapper.to("cpu")
    raw = torch.zeros(1, 2, 18)
    weights = wrapper.forward(raw, 12, (6, 12))
    assert weights.shape == (1, 2)
    assert weights.device.type == "cpu"


def test_batch_independence() -> None:
    ckpt = _toy_checkpoint()
    wrapper = subject.load_qualified_deployment(
        ckpt,
        checkpoint_file_sha256="cc" * 32,
        environment_contract_sha256="dd" * 32,
        device=torch.device("cpu"),
    )
    a = torch.randn(1, 2, 18)
    b = torch.randn(1, 2, 18)
    wa = wrapper.forward(a, 12, (6, 12))
    wb = wrapper.forward(b, 12, (6, 12))
    stacked = wrapper.forward(torch.cat([a, b], dim=0), 12, (6, 12))
    assert torch.allclose(stacked[0:1], wa, atol=1e-6)
    assert torch.allclose(stacked[1:2], wb, atol=1e-6)
    reordered = wrapper.forward(torch.cat([b, a], dim=0), 12, (6, 12))
    assert torch.allclose(reordered[0:1], wb, atol=1e-6)


def test_canonical_seed_selection_rules() -> None:
    summaries = [
        {
            "seed": 17,
            "best_epoch": 0,
            "calibration_primary": 1.0,
            "calibration_secondary": 9.0,
            "best_model_state_identity": "11" * 32,
        },
        {
            "seed": 29,
            "best_epoch": 0,
            "calibration_primary": 1.0,
            "calibration_secondary": 1.0,
            "best_model_state_identity": "22" * 32,
        },
        {
            "seed": 43,
            "best_epoch": 0,
            "calibration_primary": 1.0,
            "calibration_secondary": 0.1,
            "best_model_state_identity": "33" * 32,
        },
    ]
    sel = subject.select_canonical_seed(summaries)
    assert sel["canonical_seed"] == 17
    assert sel["rule"] == "epoch0_cross_seed_seed17"

    trained = [
        {
            "seed": 17,
            "best_epoch": 10,
            "calibration_primary": 0.5,
            "calibration_secondary": 1.0,
            "best_model_state_identity": "11" * 32,
        },
        {
            "seed": 29,
            "best_epoch": 12,
            "calibration_primary": 0.4,
            "calibration_secondary": 2.0,
            "best_model_state_identity": "22" * 32,
        },
        {
            "seed": 43,
            "best_epoch": 8,
            "calibration_primary": 0.4,
            "calibration_secondary": 1.5,
            "best_model_state_identity": "33" * 32,
        },
    ]
    sel2 = subject.select_canonical_seed(trained)
    assert sel2["canonical_seed"] == 43


def test_qualification_state_matrix() -> None:
    evidence = subject.FormalLocalizationGateEvidence(
        metric_source_identity=subject.FORMAL_LOCALIZATION_ADAPTER_ID,
        delta_pixel_ap_macro=0.01,
        delta_pixel_auroc_macro=0.0,
        delta_aupro_macro=0.0,
        per_category_localization={
            "bottle": {"delta_pixel_ap": 0.0, "delta_pixel_auroc": 0.0, "delta_aupro": 0.0}
        },
    )
    base = dict(
        kl_dlcm_gt_macro=0.1,
        kl_uniform_gt_macro=0.2,
        kl_dlcm_teacher_macro=0.1,
        kl_uniform_teacher_macro=0.2,
        per_category_kl={
            "bottle": {"gt": 0.1, "gt_uniform": 0.2, "teacher": 0.1, "teacher_uniform": 0.2}
        },
        delta_pixel_ap_macro=0.01,
        delta_pixel_auroc_macro=0.0,
        delta_aupro_macro=0.0,
        per_category_localization={
            "bottle": {"delta_pixel_ap": 0.0, "delta_pixel_auroc": 0.0, "delta_aupro": 0.0}
        },
        best_epoch=5,
        localization_evidence=evidence,
    )
    assert subject.evaluate_qualification_gates(**base)["state"] == "deployment_qualified"
    # Bare floats without formal evidence cannot qualify.
    bare = dict(base)
    bare["localization_evidence"] = None
    assert subject.evaluate_qualification_gates(**bare)["deployment_qualified"] is False
    loc_fail_evidence = subject.FormalLocalizationGateEvidence(
        metric_source_identity=subject.FORMAL_LOCALIZATION_ADAPTER_ID,
        delta_pixel_ap_macro=-0.01,
        delta_pixel_auroc_macro=0.0,
        delta_aupro_macro=0.0,
        per_category_localization={
            "bottle": {"delta_pixel_ap": 0.0, "delta_pixel_auroc": 0.0, "delta_aupro": 0.0}
        },
    )
    loc_fail = dict(base)
    loc_fail["localization_evidence"] = loc_fail_evidence
    assert (
        subject.evaluate_qualification_gates(**loc_fail)["state"]
        == "trained_but_not_deployment_qualified"
    )
    tgt_fail = dict(base)
    tgt_fail["kl_dlcm_gt_macro"] = 0.2
    assert (
        subject.evaluate_qualification_gates(**tgt_fail)["state"]
        == "localized_but_target_fidelity_unqualified"
    )
    epoch0 = dict(base)
    epoch0["best_epoch"] = 0
    assert subject.evaluate_qualification_gates(**epoch0)["deployment_qualified"] is False


def test_formal_localization_adapter_invokes_production(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    calls: list[str] = []
    from rad.evaluation import paper_metrics
    from rad.phase_b import b2_contribution_targets as contrib_mod

    real_compute = paper_metrics.compute_paper_metrics
    real_spearman = contrib_mod.spearman_fidelity
    real_top1 = contrib_mod.top1_overlap

    def wrap(name: str, fn: Any):
        def _inner(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            return fn(*args, **kwargs)

        return _inner

    monkeypatch.setattr(paper_metrics, "compute_paper_metrics", wrap("paper", real_compute))
    monkeypatch.setattr(contrib_mod, "spearman_fidelity", wrap("spearman", real_spearman))
    monkeypatch.setattr(contrib_mod, "top1_overlap", wrap("top1", real_top1))

    h = w = 8
    masks = np.zeros((2, h, w), dtype=np.float64)
    masks[1, 2:5, 2:5] = 1.0
    maps = np.random.rand(2, h, w).astype(np.float64) * 0.1
    maps[1, 2:5, 2:5] = 0.9
    teacher = maps.copy()
    labels = np.array([0.0, 1.0])
    scores = np.array([0.1, 0.9])
    metrics = subject.compute_formal_localization_metrics(
        image_labels=labels,
        image_scores=scores,
        masks=masks,
        anomaly_maps=maps,
        teacher_map=teacher,
    )
    assert metrics.metric_source_identity == subject.FORMAL_LOCALIZATION_ADAPTER_ID
    assert calls[0] == "paper"
    assert calls.count("spearman") >= 1
    assert calls.count("top1") >= 1
    assert metrics.invocation_proof["compute_paper_metrics"] is True


def test_unqualified_formal_loader_rejection() -> None:
    ckpt = _toy_checkpoint()
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_NOT_ACCEPTED"):
        subject.load_qualified_deployment(
            ckpt,
            checkpoint_file_sha256="cc" * 32,
            environment_contract_sha256="dd" * 32,
            device=torch.device("cpu"),
            require_accepted_manifest={"deployment_qualified": False},
        )


def test_jsd_top1_spearman_contracts() -> None:
    p = torch.tensor([[0.5, 0.5]])
    w = torch.tensor([[0.5, 0.5]])
    assert float(subject.allocation_jsd(p, w)) == pytest.approx(0.0)
    assert subject.top1_set_agreement(torch.tensor([1.0, 1.0]), torch.tensor([0.6, 0.4])) == 1.0
    assert subject.spearman_average_ranks(torch.ones(3), torch.ones(3)) == 1.0
    assert subject.spearman_average_ranks(torch.ones(3), torch.tensor([1.0, 2.0, 3.0])) == 0.0


def test_accepted_manifest_identities() -> None:
    ids = subject.build_accepted_manifest_identities(
        deploy_identity="aa" * 32,
        qualification_identity="bb" * 32,
        selection_identity="cc" * 32,
        upstream_identities={"descriptor": "dd" * 32},
    )
    assert len(ids["accepted_identity"]) == 64
