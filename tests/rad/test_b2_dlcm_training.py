"""RED/GREEN tests for B2-05A deterministic DLCM training lifecycle."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from rad.phase_b import b2_dlcm as model_mod
from rad.phase_b import b2_dlcm_training as subject
from tests.rad.b2_dlcm_fixtures import build_hermetic_dlcm_fixture, records_by_split


def test_optimizer_decay_groups_cover_all_parameters() -> None:
    model = model_mod.B2DLCM(seed=17)
    groups = subject.build_adamw_param_groups(model, weight_decay=1e-3, lr=3e-4)
    assert {g["group_name"] for g in groups} == {"decay", "no_decay"}
    decay_names = groups[0]["ordered_parameter_names"] if groups[0]["group_name"] == "decay" else groups[1]["ordered_parameter_names"]
    no_decay_names = groups[0]["ordered_parameter_names"] if groups[0]["group_name"] == "no_decay" else groups[1]["ordered_parameter_names"]
    assert decay_names == sorted(decay_names)
    assert no_decay_names == sorted(no_decay_names)
    all_names = set(decay_names) | set(no_decay_names)
    named = {n for n, _ in model.named_parameters()}
    assert all_names == named
    assert not (set(decay_names) & set(no_decay_names))
    # Embeddings and norms in no_decay; Linear weights in decay.
    assert any(n.endswith("layer_embedding.weight") for n in no_decay_names)
    assert any("deployment_head.weight" in n for n in decay_names)
    assert any(n.endswith(".bias") for n in no_decay_names)


def test_scheduler_warmup_and_cosine_endpoints() -> None:
    sched = subject.ExplicitLRSchedule(
        maximum_learning_rate=3e-4,
        minimum_learning_rate=3e-6,
        warmup_steps=100,
        maximum_optimizer_steps=2000,
    )
    assert sched.learning_rate_for_step(1) == pytest.approx(3e-6)
    assert sched.learning_rate_for_step(100) == pytest.approx(3e-4)
    assert sched.learning_rate_for_step(2000) == pytest.approx(3e-6)
    # Install first LR before any update.
    assert sched.next_learning_rate == pytest.approx(3e-6)
    sched.note_successful_update(1)
    assert sched.global_optimizer_step == 1
    assert sched.last_applied_learning_rate == pytest.approx(3e-6)
    assert sched.next_learning_rate == pytest.approx(sched.learning_rate_for_step(2))
    # Failed step does not advance.
    before = sched.contract_state()
    sched.note_failed_update()
    assert sched.contract_state() == before


def test_epoch_zero_does_not_advance_scheduler() -> None:
    sched = subject.ExplicitLRSchedule(
        maximum_learning_rate=3e-4,
        minimum_learning_rate=3e-6,
        warmup_steps=100,
        maximum_optimizer_steps=2000,
    )
    assert sched.global_optimizer_step == 0
    # Epoch 0 baseline must leave scheduler untouched.
    subject.run_epoch0_baseline_marker(sched)
    assert sched.global_optimizer_step == 0
    assert sched.next_learning_rate == pytest.approx(3e-6)


def test_checkpoint_selection_epoch0_protection() -> None:
    sel = subject.CheckpointSelector(min_delta=1e-5)
    # Epoch 0 initial best.
    assert sel.consider(epoch=0, primary=1.0, secondary=10.0) is True
    assert sel.best_epoch == 0
    # Secondary-only improvement cannot replace epoch 0.
    assert sel.consider(epoch=1, primary=1.0, secondary=0.1) is False
    assert sel.best_epoch == 0
    # Significant primary improvement replaces epoch 0.
    assert sel.consider(epoch=2, primary=1.0 - 2e-5, secondary=9.0) is True
    assert sel.best_epoch == 2
    # After replacement, secondary tie-break works within primary min_delta.
    assert sel.consider(epoch=3, primary=1.0 - 2e-5 + 5e-6, secondary=8.0) is True
    assert sel.best_epoch == 3
    # Both within delta → retain earlier.
    assert sel.consider(epoch=4, primary=sel.best_primary, secondary=sel.best_secondary) is False


def test_early_stopping_patience_fifty() -> None:
    stopper = subject.EarlyStopController(patience=50, maximum_epochs=500)
    sel = subject.CheckpointSelector(min_delta=1e-5)
    sel.consider(epoch=0, primary=1.0, secondary=1.0)
    status = "running"
    for epoch in range(1, 51):
        improved = sel.consider(epoch=epoch, primary=1.0, secondary=1.0)
        status = stopper.after_epoch(epoch=epoch, improved=improved)
    assert status == "early_stopped"
    assert stopper.patience_counter == 50


def test_float_bits_hex_roundtrip_and_signed_zero() -> None:
    meta = model_mod.float_to_bits_hex(0.0, dtype="float32")
    assert meta["bits_hex"] == "00000000"
    neg = model_mod.float_to_bits_hex(-0.0, dtype="float32")
    assert neg["bits_hex"] == "80000000"
    assert model_mod.bits_hex_to_float(meta) == 0.0
    assert math.copysign(1.0, model_mod.bits_hex_to_float(neg)) == -1.0
    with pytest.raises(model_mod.B2DLCMError, match="B2_DLCM_FLOAT_NONFINITE"):
        model_mod.float_to_bits_hex(float("nan"), dtype="float32")


def test_trace_hash_chain_continuity_and_tamper() -> None:
    chain = subject.TraceHashChain(schema_version="b2_dlcm_trace_chain_v1")
    r0 = subject.scientific_epoch_record(
        epoch=0,
        primary=1.0,
        secondary=2.0,
        total_loss=3.0,
        global_optimizer_step=0,
        sample_ids=["fixture-00"],
    )
    h0 = chain.append(r0)
    r1 = subject.scientific_epoch_record(
        epoch=1,
        primary=0.9,
        secondary=1.5,
        total_loss=2.5,
        global_optimizer_step=4,
        sample_ids=["fixture-00"],
    )
    h1 = chain.append(r1)
    assert chain.tail == h1
    assert chain.nodes[1]["previous_sha256"] == h0
    # Tamper rejection.
    with pytest.raises(subject.B2DLCMTrainingError, match="B2_DLCM_TRACE_CHAIN_INVALID"):
        subject.verify_trace_chain(chain.nodes + [{"schema_version": "x", "chain_index": 2, "previous_sha256": "00" * 32, "record": {}}])
    # Reorder rejection.
    with pytest.raises(subject.B2DLCMTrainingError, match="B2_DLCM_TRACE_CHAIN_INVALID"):
        subject.verify_trace_chain([chain.nodes[1], chain.nodes[0]])


def test_epoch_transaction_commit_and_partial_staging(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_17"
    tx = subject.EpochTransaction(seed_dir)
    tx.begin()
    staging = seed_dir / ".epoch_staging"
    assert staging.is_dir()
    # Write staging artifacts.
    (staging / "last_training_checkpoint.pt").write_bytes(b"last-v1")
    (staging / "training_trace.json").write_text("{}", encoding="utf-8")
    (staging / "best_training_checkpoint.pt").write_bytes(b"best-v1")
    manifest = {
        "epoch": 0,
        "best_epoch": 0,
        "best_file_sha256": subject.sha256_bytes(b"best-v1"),
        "last_file_sha256": subject.sha256_bytes(b"last-v1"),
    }
    tx.commit(manifest, update_best=True)
    committed = seed_dir / "committed"
    assert (committed / "best_training_checkpoint.pt").read_bytes() == b"best-v1"
    assert (committed / "last_training_checkpoint.pt").read_bytes() == b"last-v1"
    assert not staging.exists()
    # Partial staging discarded on resume check.
    tx.begin()
    (staging / "last_training_checkpoint.pt").write_bytes(b"partial")
    assert subject.load_committed_manifest(seed_dir)["epoch"] == 0
    tx.abort()
    assert not staging.exists()
    assert (committed / "last_training_checkpoint.pt").read_bytes() == b"last-v1"


def test_resume_rejected_for_passed_or_failed(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_17"
    seed_dir.mkdir()
    (seed_dir / "seed_manifest.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    with pytest.raises(subject.B2DLCMTrainingError, match="B2_DLCM_RESUME_FORBIDDEN"):
        subject.assert_seed_resumable(seed_dir)
    (seed_dir / "seed_manifest.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    with pytest.raises(subject.B2DLCMTrainingError, match="B2_DLCM_RESUME_FORBIDDEN"):
        subject.assert_seed_resumable(seed_dir)


def test_failure_attestation_and_no_commit(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_29"
    tx = subject.EpochTransaction(seed_dir)
    tx.begin()
    subject.write_failure_attestation(
        seed_dir,
        seed=29,
        stage="optimizer_step",
        error_code="B2_DLCM_NONFINITE_LOSS",
        last_valid_committed_epoch=-1,
        global_optimizer_step=0,
        trace_chain_tail=None,
        identities={"model_state_scientific_sha256": "ab" * 32},
        environment_identity="cd" * 32,
        uncommitted_staging=["last_training_checkpoint.pt"],
    )
    assert (seed_dir / "failure_attestation.json").is_file()
    assert (seed_dir / "failure_attestation.json.sha256").is_file()
    # Staging must not become committed.
    assert not (seed_dir / "committed" / "epoch_state_manifest.json").exists()


def test_environment_contract_immutable_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    contract = subject.collect_environment_contract(
        visible_gpu_count=0,
        allow_cpu_for_hermetic=True,
    )
    assert contract["training_dtype"] == "float32"
    assert contract["amp_enabled"] is False
    assert contract["deterministic_algorithms_enabled"] is True
    assert contract["visible_gpu_count"] in (0, 1)
    path = tmp_path / "environment_contract.json"
    digest = subject.persist_environment_contract(path, contract)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert subject.environment_contract_sha256(loaded) == digest
    # Mutation of scientific fields changes identity.
    loaded2 = dict(loaded)
    loaded2["amp_enabled"] = True
    assert subject.environment_contract_sha256(loaded2) != digest


def test_sampler_permutation_covers_sixteen_once() -> None:
    records = build_hermetic_dlcm_fixture()
    train = records_by_split(records)["training"]
    ids = [r.stable_sample_id for r in train]
    gen_seed = model_mod.derive_u63_seed(model_seed=17, component="sampler")
    order = subject.deterministic_epoch_permutation(ids, epoch=0, sampler_seed=gen_seed)
    assert sorted(order) == sorted(ids)
    assert len(order) == 16
    order2 = subject.deterministic_epoch_permutation(ids, epoch=1, sampler_seed=gen_seed)
    assert order != order2
    # Distinct seeds → distinct sequences.
    gen_seed_b = model_mod.derive_u63_seed(model_seed=29, component="sampler")
    order_b = subject.deterministic_epoch_permutation(ids, epoch=0, sampler_seed=gen_seed_b)
    assert order_b != order


def test_hermetic_contract_dry_train_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract dry path: epoch-0 + one tiny epoch on CPU hermetic data only."""

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    records = build_hermetic_dlcm_fixture()
    result = subject.run_hermetic_contract_training(
        output_root=tmp_path / "run",
        seed=17,
        records=records,
        maximum_epochs=1,
        patience=50,
        device="cpu",
    )
    assert result["status"] in {"passed", "early_stopped", "completed_epoch"}
    assert result["real_training_started"] is False
    assert result["evaluation_unlocked"] is False
    seed_dir = tmp_path / "run" / "seed_17"
    assert (seed_dir / "committed" / "training_trace.json").is_file()
    assert (seed_dir / "committed" / "epoch_state_manifest.json").is_file()
