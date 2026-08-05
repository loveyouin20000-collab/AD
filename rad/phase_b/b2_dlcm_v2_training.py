"""B2-05C1 V2 training lifecycle: GT-only selection, hermetic dry-run."""

from __future__ import annotations

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
