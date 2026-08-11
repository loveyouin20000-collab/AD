"""RED→GREEN tests for B2-05C0 equal-family / weighted-family oracles and diagnosis guards."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rad.phase_b import b2_dlcm_target_conflict_diagnosis as subject


def test_exact_equal_family_oracle_derivation() -> None:
    """Unconstrained simplex optimum of 0.5 KL(p_gt||w)+0.5 KL(p_t||w) is (p_gt+p_t)/2."""

    p_gt = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    p_t = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    w = subject.equal_family_oracle(p_gt, p_t)
    expected = 0.5 * p_gt + 0.5 * p_t
    assert torch.allclose(w, expected)
    assert float(w.sum()) == pytest.approx(1.0)
    # Any other simplex point has higher or equal dual-family mean KL.
    for alt in (
        torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64),
        p_gt,
        p_t,
        torch.tensor([0.9, 0.05, 0.05], dtype=torch.float64),
    ):
        assert subject.dual_family_mean_kl(p_gt, p_t, w) <= subject.dual_family_mean_kl(
            p_gt, p_t, alt
        ) + 1e-12


def test_weighted_family_oracle_matches_alpha_formula() -> None:
    p_gt = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    p_t = torch.tensor([0.2, 0.2, 0.6], dtype=torch.float64)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = subject.weighted_family_oracle(p_gt, p_t, alpha=alpha)
        expected = alpha * p_gt + (1.0 - alpha) * p_t
        assert torch.allclose(w, expected)
        assert float(w.sum()) == pytest.approx(1.0)


def test_alpha_grid_determinism() -> None:
    grid_a = subject.alpha_grid()
    grid_b = subject.alpha_grid()
    assert grid_a == grid_b
    assert grid_a[0] == pytest.approx(0.0)
    assert grid_a[-1] == pytest.approx(1.0)
    assert len(grid_a) == 101
    assert all(abs(grid_a[i] - i * 0.01) < 1e-12 for i in range(101))


def test_no_evaluation_based_alpha_selection() -> None:
    """Feasible intervals are reported; evaluation must not choose a single alpha."""

    # Synthetic: alpha near 0.7 passes both gates on calib; eval interval different.
    calib = {
        "feasible_alphas": [0.65, 0.66, 0.67, 0.68, 0.69, 0.70],
        "interval": [0.65, 0.70],
    }
    eval_posthoc = {
        "feasible_alphas": [0.40, 0.41],
        "interval": [0.40, 0.41],
    }
    report = subject.finalize_alpha_feasibility_report(
        calibration=calib,
        evaluation_posthoc=eval_posthoc,
    )
    assert "selected_alpha" not in report
    assert report["calibration_feasible_alpha_interval"] == [0.65, 0.70]
    assert report["evaluation_posthoc_feasible_alpha_interval"] == [0.40, 0.41]
    assert report["evaluation_used_for_alpha_selection"] is False
    assert report["note"] == "evaluation interval is diagnostic only; never used to choose alpha"


def test_split_separated_reporting_keys() -> None:
    payload = subject.empty_split_depth_report_scaffold()
    for split in ("training", "calibration", "evaluation"):
        assert split in payload
        for depth in (12, 18, 24):
            assert depth in payload[split] or str(depth) in payload[split]
            cell = payload[split][depth] if depth in payload[split] else payload[split][str(depth)]
            assert "gt" in cell and "teacher" in cell
            assert "equal_family_diagnostic_average" in cell  # secondary only


def test_gt_teacher_not_mixed_in_primary() -> None:
    metrics = subject.oracle_candidate_metrics(
        p_gt=torch.tensor([0.8, 0.2], dtype=torch.float64),
        p_t=torch.tensor([0.3, 0.7], dtype=torch.float64),
        weights=torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    assert "kl_gt" in metrics and "kl_teacher" in metrics
    assert "kl_mixed" not in metrics
    assert metrics["dual_family_mean_kl"] == pytest.approx(
        0.5 * metrics["kl_gt"] + 0.5 * metrics["kl_teacher"]
    )


def test_diagnosis_refuses_checkpoint_modification(tmp_path: Path) -> None:
    ckpt = tmp_path / "best_training_checkpoint.pt"
    torch.save({"epoch": 1, "model": {"x": torch.zeros(1)}}, ckpt)
    before = ckpt.read_bytes()
    subject.assert_checkpoint_bytes_immutable(ckpt, expected_bytes=before)
    # Mutating then checking must fail.
    ckpt.write_bytes(before + b"x")
    with pytest.raises(subject.B2DLCMDiagnosisError, match="checkpoint.*modif|immutable"):
        subject.assert_checkpoint_bytes_immutable(ckpt, expected_bytes=before)


def test_diagnosis_refuses_training_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):  # noqa: ANN001
        raise AssertionError("training must not be invoked")

    monkeypatch.setattr(subject, "_FORBIDDEN_TRAINING_HOOK", _boom)
    with pytest.raises(subject.B2DLCMDiagnosisError, match="training"):
        subject.guard_no_training_invocation(attempt_train=True)


def test_diagnosis_refuses_teacher_backbone_invocation() -> None:
    with pytest.raises(subject.B2DLCMDiagnosisError, match="teacher|backbone"):
        subject.guard_no_teacher_backbone_invocation(teacher_forward_count=1)


def test_diagnosis_refuses_accepted_manifest_generation(tmp_path: Path) -> None:
    with pytest.raises(subject.B2DLCMDiagnosisError, match="accepted.?manifest"):
        subject.guard_no_accepted_manifest_write(
            output_dir=tmp_path,
            filename="accepted_deployment_manifest.json",
        )


def test_unchanged_qualification_status_contract() -> None:
    observed = {
        "qualification_status": "localized_but_target_fidelity_unqualified",
        "deployment_qualified": False,
        "accepted_deployment_manifest_created": False,
        "identities": {
            "accepted_training_plan": "59e20f4cb337ef42384f70bb8b3dad5211d906341b0a2d41f7e6847610635980",
            "seed_collection": "94a6a9332a0694889c7a0255814ac13fe8316c601529197063165ce14ec1277f",
            "canonical_selection": "e3bc06dfa02d6109544648020680d907bf0fce5ed7a093372d74009f9e69e142",
            "deployment_scientific": "4cbc6fb88f39ed86deacfbbe48580f7682453b94becb046ec6ef1b1302df378a",
            "evaluation_unlock": "19dca41e9f647d12afce9877a7340f5af58bf9a23997d7339dded26d89fe73dd",
            "qualification_scientific": "da51e5fc1302cf507bc844f87e82cb66f7d2fa0a13e61f28a0dba14333201c49",
        },
    }
    subject.assert_qualification_frozen(observed)


def test_diagnosis_kl_nonnegative_with_zero_support_targets() -> None:
    p = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    # Target with mass where p is zero — raw eval helper can go negative; diagnosis must not.
    q = torch.tensor([0.2, 0.2, 0.6], dtype=torch.float64)
    kl = subject.kl_p_vs_w(q, p)
    assert kl >= 0.0


def test_signed_diagnostics_not_proxied_in_diagnosis_module() -> None:
    weights = torch.tensor([0.25, 0.25, 0.25, 0.25])
    phi = torch.tensor([1.0, 0.0, -0.5, -0.5])
    with pytest.raises(Exception, match="allocation.?weight|substitut|signed"):
        subject.refuse_signed_proxy_from_weights(weights, phi)
