"""Gap-filling RED/GREEN tests for B2-05A §§53–56 closure."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rad.phase_b import b2_dlcm as model_mod
from rad.phase_b import b2_dlcm_deployment as deploy
from rad.phase_b import b2_dlcm_training as training
from tests.rad.b2_dlcm_fixtures import ACCEPTED_UPSTREAM


def test_production_hermetic_fixture_contract() -> None:
    records = training.build_hermetic_contract_records()
    assert len(records) == 32
    splits = {name: sum(1 for r in records if r["split"] == name) for name in ("training", "calibration", "evaluation")}
    assert splits == {"training": 16, "calibration": 8, "evaluation": 8}
    for record in records:
        for depth, n in ((12, 2), (18, 3), (24, 4)):
            desc = record["descriptors"][depth]
            assert desc.shape == (n, 18)
            assert desc.dtype == torch.float32
            assert bool(torch.isfinite(desc).all())
            assert record["p_gt"][depth].shape == (n,)
            assert record["phi_gt"][depth].shape == (n,)
            assert record["anomaly_maps"][depth].shape[0] == n


def test_dry_run_complete_hermetic_validation_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    out = tmp_path / "must_not_exist"
    before = {p for p in tmp_path.rglob("*")}
    result = training.dry_run_complete_contract_validation(
        config={
            "contract_stage": "b2_05a",
            "real_training_enabled": False,
            "candidate_layers": [6, 12, 18, 24],
            "prediction_depths": [12, 18, 24],
            "seeds": [17, 29, 43],
            "accepted_upstream": ACCEPTED_UPSTREAM,
            "authoritative_base_commit": "97a4f497f6f2b096dd4a339555f81e7296ec3035",
        },
        seed=17,
        output_root=out,
    )
    assert result["hermetic_records_validated"] == 32
    assert result["artifact_written"] is False
    assert result["run_directory_created"] is False
    assert result["real_training_started"] is False
    assert result["evaluation_unlocked"] is False
    assert result["teacher_forward_count"] == 0
    assert not out.exists()
    assert {p for p in tmp_path.rglob("*")} == before


def test_dropout_resume_reproduces_next_mask() -> None:
    model = model_mod.B2DLCM(seed=17)
    model.train()
    x = torch.ones(2, 2, 18)
    _ = model.forward_training(x, prediction_depth=12)
    # Snapshot generator states after first forward.
    states = {name: g.get_state().clone() for name, g in model.dropout_generators.items()}
    y1 = model.forward_training(x, prediction_depth=12).deployment_logits.detach().clone()
    # Restore and reproduce the same next masks / logits.
    for name, state in states.items():
        model.dropout_generators[name].set_state(state)
    y2 = model.forward_training(x, prediction_depth=12).deployment_logits.detach().clone()
    assert torch.equal(y1, y2)


def test_optimizer_state_remapped_by_parameter_name() -> None:
    model = model_mod.B2DLCM(seed=17)
    optim, groups = training.build_adamw(model, lr=3e-6)
    # Take one step so AdamW state exists.
    loss = model.forward_training(torch.randn(2, 2, 18), prediction_depth=12).deployment_logits.pow(2).mean()
    loss.backward()
    optim.step()
    mapping = training.optimizer_state_by_parameter_name(model, optim, groups)
    named = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(mapping) == named
    for _name, entry in mapping.items():
        assert entry["optimizer_group"] in {"decay", "no_decay"}
        assert "exp_avg" in entry and "exp_avg_sq" in entry


def test_scheduler_reconstruction_resume_next_lr() -> None:
    sched = training.ExplicitLRSchedule()
    sched.note_successful_update(1)
    state = sched.contract_state()
    restored = training.ExplicitLRSchedule.from_contract_state(state)
    assert restored.global_optimizer_step == 1
    assert restored.next_learning_rate == pytest.approx(sched.next_learning_rate)
    assert restored.learning_rate_for_step(100) == pytest.approx(3e-4)


def test_collection_failure_manifest_blocks_continuation(tmp_path: Path) -> None:
    payload = training.build_collection_failure_manifest(
        failed_seed=29,
        error_code="B2_DLCM_NONFINITE_LOSS",
        completed_seeds=[17],
        environment_identity="ee" * 32,
    )
    assert payload["collection_status"] == "seed_collection_failed"
    path = tmp_path / "collection_failure_manifest.json"
    training.persist_collection_failure_manifest(path, payload)
    assert path.is_file()
    with pytest.raises(deploy.B2DLCMDeploymentError, match="B2_DLCM_COLLECTION_FAILED"):
        deploy.require_passed_seed_collection_for_canonical(tmp_path)


def test_reproduction_failure_blocks_evaluation_export() -> None:
    original = {"nodes": [{"epoch": 0, "h": "aa"}], "model": "m"}
    bad = {"nodes": [{"epoch": 0, "h": "bb"}], "model": "m"}
    cmp = deploy.compare_reproduction(original, bad)
    with pytest.raises(deploy.B2DLCMDeploymentError, match="B2_DLCM_REPRO_BLOCKED"):
        deploy.require_reproduction_passed_for_evaluation(cmp)


def test_dual_family_diagnostic_invalid_if_either_family_invalid() -> None:
    ok = deploy.dual_family_diagnostic({"kl": 0.1}, {"kl": 0.3}, metric="kl")
    assert ok["status"] == "ok"
    assert ok["value"] == pytest.approx(0.2)
    bad = deploy.dual_family_diagnostic({"kl": 0.1, "valid": False}, {"kl": 0.3}, metric="kl")
    assert bad["status"] == "invalid"
    assert bad["value"] is None


def test_cli_rejects_forbidden_flags() -> None:
    from tools import train_b2_dlcm as cli

    parser = cli._parser()
    for banned in (
        "--fixture",
        "--skip-identity",
        "--skip-environment",
        "--force-resume",
        "--evaluation-unlock",
        "--accepted-override",
        "--cpu-fallback",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])  # ensure parser builds
        option_strings = {opt for action in parser._actions for opt in action.option_strings}
        assert banned not in option_strings


def test_scope_rejects_runtime_fields_in_scientific_hashes() -> None:
    payload = {
        "schema_version": "x",
        "value": 1,
        "hostname": "bad",
        "gpu_uuid": "bad",
        "absolute_path": "/tmp/x",
    }
    with pytest.raises(deploy.B2DLCMDeploymentError, match="B2_DLCM_SCI_RUNTIME_FIELD"):
        deploy.scientific_hash_rejecting_runtime_fields(
            payload,
            whitelist=("schema_version", "value"),
        )
