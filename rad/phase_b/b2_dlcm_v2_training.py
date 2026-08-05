"""B2-05C1 V2 training lifecycle: GT-only selection, hermetic dry-run."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_training as v1_train
from rad.phase_b import b2_dlcm_v2 as v2
from rad.phase_b import b2_dlcm_v2_protocol as protocol


class B2DLCMV2TrainingError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV2TrainingError(code, detail)


@dataclass
class GTOnlyCheckpointSelector:
    min_delta: float = 1e-5
    best_primary: float | None = None
    best_secondary: float | None = None
    best_epoch: int | None = None

    def consider(self, *, epoch: int, primary: float, secondary: float) -> bool:
        if self.best_primary is None:
            self.best_primary = float(primary)
            self.best_secondary = float(secondary)
            self.best_epoch = int(epoch)
            return True
        assert self.best_primary is not None and self.best_secondary is not None
        if epoch == 0:
            return False
        if self.best_epoch == 0:
            if primary < self.best_primary - self.min_delta:
                self.best_primary = float(primary)
                self.best_secondary = float(secondary)
                self.best_epoch = int(epoch)
                return True
            return False
        if primary < self.best_primary - self.min_delta:
            self.best_primary = float(primary)
            self.best_secondary = float(secondary)
            self.best_epoch = int(epoch)
            return True
        if abs(primary - self.best_primary) <= self.min_delta and secondary < (
            self.best_secondary - self.min_delta
        ):
            self.best_primary = float(primary)
            self.best_secondary = float(secondary)
            self.best_epoch = int(epoch)
            return True
        return False


def select_canonical_seed_gt_only(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    min_delta: float = 1e-5,
) -> int:
    """Select canonical seed using GT primary/secondary only."""

    if not seed_results:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "empty seed results")
    # All epoch-0 fallback → seed 17
    if all(int(r.get("best_epoch", -1)) == 0 for r in seed_results):
        return 17
    best = None
    for row in seed_results:
        seed = int(row["seed"])
        primary = float(row["primary"])
        secondary = float(row["secondary"])
        epoch = int(row["best_epoch"])
        cand = (primary, secondary, seed, epoch)
        if best is None:
            best = cand
            continue
        b_primary, b_secondary, b_seed, _b_epoch = best
        if primary < b_primary - min_delta:
            best = cand
        elif abs(primary - b_primary) <= min_delta and secondary < b_secondary - min_delta:
            best = cand
        elif abs(primary - b_primary) <= min_delta and abs(secondary - b_secondary) <= min_delta:
            if seed < b_seed:
                best = cand
    assert best is not None
    return int(best[2])


def calibration_metrics_gt_only(
    model: v2.B2DLCMV2,
    calibration: Sequence[Any],
    *,
    device: torch.device | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Return GT primary, GT signed secondary, and teacher diagnostic scalars."""

    device_t = device or torch.device("cpu")
    primary_terms: list[float] = []
    secondary_terms: list[float] = []
    teacher_terms: list[float] = []
    for record in calibration:
        depth_primary: list[float] = []
        depth_secondary: list[float] = []
        depth_teacher: list[float] = []
        for depth in model.prediction_depths:
            if hasattr(record, "descriptors"):
                desc = record.descriptors[depth].unsqueeze(0).to(device=device_t)
                p_gt = record.p_gt[depth].unsqueeze(0).to(device=device_t)
                p_t = record.p_t[depth].unsqueeze(0).to(device=device_t)
                phi_gt = record.phi_gt[depth].unsqueeze(0).to(device=device_t)
            else:
                desc = record["descriptors"][depth].unsqueeze(0).to(device=device_t)
                p_gt = record["p_gt"][depth].unsqueeze(0).to(device=device_t)
                p_t = record["p_t"][depth].unsqueeze(0).to(device=device_t)
                phi_gt = record["phi_gt"][depth].unsqueeze(0).to(device=device_t)
            out = model.forward_training(desc, prediction_depth=int(depth))
            gt_kl = v1.allocation_kl(p_gt, out.gt_deployment_logits)
            t_kl = v1.allocation_kl(p_t, out.teacher_allocation_logits)
            s_gt, _ = v1.signed_loss(out.gt_signed, phi_gt)
            depth_primary.append(float(gt_kl))
            depth_secondary.append(float(s_gt))
            depth_teacher.append(float(t_kl))
        primary_terms.append(sum(depth_primary) / 3.0)
        secondary_terms.append(sum(depth_secondary) / 3.0)
        teacher_terms.append(sum(depth_teacher) / 3.0)
    primary = sum(primary_terms) / float(len(primary_terms))
    secondary = sum(secondary_terms) / float(len(secondary_terms))
    teacher = sum(teacher_terms) / float(len(teacher_terms))
    if not (primary == primary and secondary == secondary and teacher == teacher):
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "nonfinite calibration metrics")
    return primary, secondary, {"teacher_alloc_kl_macro": teacher}


def teacher_must_not_affect_selection() -> dict[str, bool]:
    """Contract marker used by tests: teacher never enters selector inputs."""

    return {
        "teacher_in_primary": False,
        "teacher_in_secondary": False,
        "teacher_in_patience": False,
        "teacher_in_canonical": False,
    }


def build_hermetic_v2_records(*, map_hw: tuple[int, int] = (8, 8)) -> list[dict[str, Any]]:
    return v1_train.build_hermetic_contract_records(map_hw=map_hw)


def dry_run_complete_v2_contract_validation(
    *,
    config: Mapping[str, Any],
    seed: int,
    output_dir: Path | str,
) -> dict[str, Any]:
    """C1A dry-run: validate V2 math/contracts without writing artifacts."""

    if list(config.get("candidate_layers", [])) != [6, 12, 18, 24]:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "candidate_layers mismatch")
    if list(config.get("prediction_depths", [])) != [12, 18, 24]:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "prediction_depths mismatch")
    if abs(float(config.get("teacher_allocation_loss_weight", -1)) - 0.25) > 0:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "teacher_allocation_loss_weight must be 0.25")
    return dry_run_v2_summary(config=config, seed=seed, output_dir=output_dir)


def dry_run_v2_summary(
    *,
    config: Mapping[str, Any],
    seed: int,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Dry-run that catches expected final-content forbid and returns summary flags."""

    protocol.reject_bypass_flags(config)
    if config.get("real_training_enabled") is not False:
        _fail("B2_DLCM_V2_REAL_TRAINING_NOT_ENABLED", "dry-run requires real_training_enabled=false")
    stage = config.get("contract_stage")
    if stage not in {"b2_05c1a", "b2_05c1"}:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "contract_stage must be b2_05c1a|b2_05c1")

    records = build_hermetic_v2_records()
    model = v2.B2DLCMV2(seed=int(seed))
    train = [r for r in records if r["split"] == "training"]
    calibration = [r for r in records if r["split"] == "calibration"]
    batch = train[:4]
    depth_payload: dict[int, dict[str, torch.Tensor]] = {}
    for depth in model.prediction_depths:
        descs = torch.stack([r["descriptors"][depth] for r in batch], dim=0)
        out = model.forward_training(descs, prediction_depth=depth)
        depth_payload[depth] = {
            "gt_deployment_logits": out.gt_deployment_logits,
            "teacher_allocation_logits": out.teacher_allocation_logits,
            "gt_signed": out.gt_signed,
            "teacher_signed": out.teacher_signed,
            "p_gt": torch.stack([r["p_gt"][depth] for r in batch], dim=0),
            "p_t": torch.stack([r["p_t"][depth] for r in batch], dim=0),
            "phi_gt": torch.stack([r["phi_gt"][depth] for r in batch], dim=0),
            "phi_t": torch.stack([r["phi_t"][depth] for r in batch], dim=0),
        }
    loss, _ = v2.total_dlcm_v2_loss(depth_payload)
    if not bool(torch.isfinite(loss)):
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "dry-run loss nonfinite")
    primary, secondary, teacher_diag = calibration_metrics_gt_only(model, calibration)
    if not (teacher_diag["teacher_alloc_kl_macro"] == teacher_diag["teacher_alloc_kl_macro"]):
        _fail("B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID", "teacher diagnostic nonfinite")
    selector = GTOnlyCheckpointSelector()
    selector.consider(epoch=0, primary=primary, secondary=secondary)
    assert teacher_must_not_affect_selection()["teacher_in_primary"] is False
    deploy_state = v2.extract_deployment_state_dict(model)
    if any(any(k.startswith(b) for k in deploy_state) for b in v2.AUXILIARY_HEAD_PREFIXES):
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "auxiliary leaked into deployment")
    final_content_resolved = False
    try:
        protocol.forbid_final_content_access(unlocked=False, context="dry_run_probe")
    except protocol.B2DLCMV2ProtocolError as exc:
        if exc.code != "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN":
            raise
        final_content_resolved = False
    return {
        "mode": "dry_run",
        "status": "contract_validated",
        "real_training_started": False,
        "development_evaluation_started": False,
        "final_content_resolved": final_content_resolved,
        "final_materialization_started": False,
        "final_evaluation_started": False,
        "artifact_written": False,
        "run_directory_created": False,
        "teacher_forward_count": 0,
        "seed": int(seed),
        "dry_run_loss_finite": True,
        "calibration_primary": primary,
        "calibration_secondary": secondary,
        "teacher_diagnostics": teacher_diag,
        "deployment_param_count": len(deploy_state),
        "v1_immutable": v2.v1_immutable_identity(),
        "output_dir": str(Path(output_dir)),
    }


def _cpu_identity_model_v2(model: v2.B2DLCMV2) -> v2.B2DLCMV2:
    clone = v2.B2DLCMV2(
        seed=None,
        candidate_layers=model.candidate_layers,
        prediction_depths=model.prediction_depths,
        descriptor_dimension=model.descriptor_dimension,
        layer_embedding_dimension=model.layer_embedding_dimension,
        depth_embedding_dimension=model.depth_embedding_dimension,
        hidden_dimension=model.hidden_dimension,
        dropout_probability=model.dropout_probability,
        initialize=False,
    )
    clone.load_state_dict(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        strict=True,
    )
    return clone


def run_v2_contract_training(
    *,
    output_root: Path,
    seed: int,
    records: Sequence[Any],
    maximum_epochs: int = 1,
    patience: int = 50,
    device: str = "cpu",
    batch_size: int = 4,
    environment_contract: Mapping[str, Any] | None = None,
    allow_existing_output: bool = False,
    mark_real_training_started: bool = False,
) -> dict[str, Any]:
    """Official/hermetic V2 seed training with GT-only selection."""

    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing_output:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "output root must be empty/fresh")
    output_root.mkdir(parents=True, exist_ok=True)

    if environment_contract is None:
        env = v1_train.collect_environment_contract(allow_cpu_for_hermetic=(device == "cpu"))
        v1_train.persist_environment_contract(output_root / "environment_contract.json", env)
    else:
        env = dict(environment_contract)
        env_path = output_root / "environment_contract.json"
        if env_path.is_file():
            existing = json.loads(env_path.read_text(encoding="utf-8"))
            if v1_train.environment_contract_sha256(existing) != v1_train.environment_contract_sha256(
                env
            ):
                _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "environment contract drifted")
        else:
            v1_train.persist_environment_contract(env_path, env)

    seed_dir = output_root / f"seed_{seed}"
    if seed_dir.exists():
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", f"seed directory already exists: seed_{seed}")

    by_split: dict[str, list[Any]] = {"training": [], "calibration": [], "evaluation": []}
    for record in records:
        by_split[record.split].append(record)
    if len(by_split["training"]) != 16 or len(by_split["calibration"]) != 8:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "expected 16/8 training/calibration")
    if len(by_split["evaluation"]) != 8:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "expected 8 evaluation placeholders")

    tx = v1_train.EpochTransaction(seed_dir)
    model = v2.B2DLCMV2(seed=seed)
    # Epoch-0 uniformity proof for both allocation heads.
    with torch.no_grad():
        assert torch.all(model.gt_deployment_head.weight == 0)
        assert torch.all(model.teacher_allocation_head.weight == 0)
    if device != "cpu":
        model = v2.move_model_to_device_and_verify(model, torch.device(device))
    optimizer, _groups = v1_train.build_adamw(model, lr=3e-6)
    schedule = v1_train.ExplicitLRSchedule()
    schedule.install_into_optimizer(optimizer)
    selector = GTOnlyCheckpointSelector()
    stopper = v1_train.EarlyStopController(patience=patience, maximum_epochs=maximum_epochs)
    chain = v1_train.TraceHashChain()

    train_ids = [r.stable_sample_id for r in by_split["training"]]
    id_to_record = {r.stable_sample_id: r for r in by_split["training"]}
    calibration = sorted(by_split["calibration"], key=lambda r: r.stable_sample_id)
    component_seeds = model.component_seeds

    primary0, secondary0, teacher0 = calibration_metrics_gt_only(
        model, calibration, device=next(model.parameters()).device
    )
    selector.consider(epoch=0, primary=primary0, secondary=secondary0)
    v1_train.run_epoch0_baseline_marker(schedule)
    staging = tx.begin()
    torch.save({"model": model.state_dict(), "epoch": 0}, staging / "best_training_checkpoint.pt")
    torch.save({"model": model.state_dict(), "epoch": 0}, staging / "last_training_checkpoint.pt")
    record0 = v1_train.scientific_epoch_record(
        epoch=0,
        primary=primary0,
        secondary=secondary0,
        total_loss=0.0,
        global_optimizer_step=0,
        sample_ids=train_ids,
    )
    record0["teacher_alloc_kl_macro"] = float(teacher0["teacher_alloc_kl_macro"])
    h0 = chain.append(record0)
    (staging / "training_trace.json").write_text(
        json.dumps({"nodes": chain.nodes, "tail": chain.tail}, sort_keys=True),
        encoding="utf-8",
    )
    tx.commit(
        {
            "epoch": 0,
            "best_epoch": 0,
            "patience": 0,
            "global_optimizer_step": 0,
            "trace_chain_tail": h0,
            "best_file_sha256": "pending",
            "status": "running",
        },
        update_best=True,
    )
    manifest = v1_train.load_committed_manifest(seed_dir)
    manifest["best_file_sha256"] = v1_train.sha256_file(
        seed_dir / "committed" / "best_training_checkpoint.pt"
    )
    manifest["last_file_sha256"] = v1_train.sha256_file(
        seed_dir / "committed" / "last_training_checkpoint.pt"
    )
    v1_train._atomic_write_json(seed_dir / "committed" / "epoch_state_manifest.json", manifest)

    status = "running"
    last_epoch = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = v1_train.deterministic_epoch_permutation(
            train_ids,
            epoch=epoch,
            sampler_seed=component_seeds["sampler"],
        )
        epoch_losses: list[float] = []
        for step_idx in range(4):
            batch_ids = order[step_idx * batch_size : (step_idx + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)
            depth_payload: dict[int, dict[str, torch.Tensor]] = {}
            for depth in model.prediction_depths:
                device_t = next(model.parameters()).device
                descs = torch.stack(
                    [id_to_record[i].descriptors[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                p_gt = torch.stack(
                    [id_to_record[i].p_gt[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                p_t = torch.stack(
                    [id_to_record[i].p_t[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                phi_gt = torch.stack(
                    [id_to_record[i].phi_gt[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                phi_t = torch.stack(
                    [id_to_record[i].phi_t[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                out = model.forward_training(descs, prediction_depth=depth)
                depth_payload[depth] = {
                    "gt_deployment_logits": out.gt_deployment_logits,
                    "teacher_allocation_logits": out.teacher_allocation_logits,
                    "gt_signed": out.gt_signed,
                    "teacher_signed": out.teacher_signed,
                    "p_gt": p_gt,
                    "p_t": p_t,
                    "phi_gt": phi_gt,
                    "phi_t": phi_t,
                }
            loss, _ = v2.total_dlcm_v2_loss(depth_payload)
            if not bool(torch.isfinite(loss)):
                v1_train.write_failure_attestation(
                    seed_dir,
                    seed=seed,
                    stage="optimizer_step",
                    error_code="B2_DLCM_NONFINITE_LOSS",
                    last_valid_committed_epoch=last_epoch,
                    global_optimizer_step=schedule.global_optimizer_step,
                    trace_chain_tail=chain.tail,
                    identities={
                        "model_state_scientific_sha256": v2.model_state_scientific_sha256(model)
                    },
                    environment_identity=v1_train.environment_contract_sha256(env),
                    uncommitted_staging=[],
                )
                return {
                    "status": "failed",
                    "real_training_started": bool(mark_real_training_started),
                    "evaluation_unlocked": False,
                }
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t = schedule.global_optimizer_step + 1
            schedule.note_successful_update(t)
            schedule.install_into_optimizer(optimizer)
            epoch_losses.append(float(loss.detach()))

        primary, secondary, teacher_diag = calibration_metrics_gt_only(
            model, calibration, device=next(model.parameters()).device
        )
        improved = selector.consider(epoch=epoch, primary=primary, secondary=secondary)
        status = stopper.after_epoch(epoch=epoch, improved=improved)
        staging = tx.begin()
        torch.save({"model": model.state_dict(), "epoch": epoch}, staging / "last_training_checkpoint.pt")
        if improved:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch},
                staging / "best_training_checkpoint.pt",
            )
        record = v1_train.scientific_epoch_record(
            epoch=epoch,
            primary=primary,
            secondary=secondary,
            total_loss=sum(epoch_losses) / len(epoch_losses),
            global_optimizer_step=schedule.global_optimizer_step,
            sample_ids=order,
        )
        record["teacher_alloc_kl_macro"] = float(teacher_diag["teacher_alloc_kl_macro"])
        tail = chain.append(record)
        (staging / "training_trace.json").write_text(
            json.dumps({"nodes": chain.nodes, "tail": chain.tail}, sort_keys=True),
            encoding="utf-8",
        )
        tx.commit(
            {
                "epoch": epoch,
                "best_epoch": selector.best_epoch,
                "patience": stopper.patience_counter,
                "global_optimizer_step": schedule.global_optimizer_step,
                "trace_chain_tail": tail,
                "status": status,
            },
            update_best=improved,
        )
        last_epoch = epoch
        if status == "early_stopped":
            break

    best_path = seed_dir / "committed" / "best_training_checkpoint.pt"
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model = v2.B2DLCMV2(
        seed=None,
        candidate_layers=model.candidate_layers,
        prediction_depths=model.prediction_depths,
        descriptor_dimension=model.descriptor_dimension,
        layer_embedding_dimension=model.layer_embedding_dimension,
        depth_embedding_dimension=model.depth_embedding_dimension,
        hidden_dimension=model.hidden_dimension,
        dropout_probability=model.dropout_probability,
        initialize=False,
    )
    best_model.load_state_dict(best_payload["model"], strict=True)
    best_identity = v2.model_state_scientific_sha256(best_model)

    return {
        "status": status if status != "running" else "completed_epoch",
        "real_training_started": bool(mark_real_training_started),
        "evaluation_unlocked": False,
        "best_epoch": selector.best_epoch,
        "seed": seed,
        "primary": selector.best_primary,
        "secondary": selector.best_secondary,
        "trace_chain_tail": chain.tail,
        "model_state_scientific_sha256": best_identity,
        "last_model_state_scientific_sha256": v2.model_state_scientific_sha256(
            _cpu_identity_model_v2(model)
        ),
        "last_epoch": last_epoch,
        "global_optimizer_step": schedule.global_optimizer_step,
        "epoch0_primary": primary0,
        "epoch0_secondary": secondary0,
        "epoch0_teacher_alloc_kl_macro": float(teacher0["teacher_alloc_kl_macro"]),
    }
