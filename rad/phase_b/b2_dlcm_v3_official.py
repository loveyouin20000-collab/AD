"""B2-05C2B official V3 category-robust DLCM training orchestration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_deployment as v1_deployment
from rad.phase_b import b2_dlcm_training as v1_training
from rad.phase_b import b2_dlcm_v3 as v3
from rad.phase_b import b2_dlcm_v3_protocol as protocol
from rad.phase_b import b2_dlcm_v3_roster_adoption as roster_adoption
from rad.phase_b import b2_dlcm_v3_training as v3_training

OFFICIAL_PLAN_SCHEMA_VERSION = "b2_dlcm_v3_accepted_training_plan_v1"
C2A_CONTRACT_TAG = "b2-dlcm-category-robust-contract-v3"
C2A_ADOPTION_COMMIT = "c9dceb4be5438aa0c745fbaf4e3cf7bcba528e64"
V3_IMPLEMENTATION_COMMIT = "e4793c1dac29cc1a195fba098fe765d30bf66d74"


class B2DLCMV3OfficialError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV3OfficialError(code, detail)


def records_as_namespaces(rows: Sequence[Mapping[str, Any]]) -> list[SimpleNamespace]:
    out: list[SimpleNamespace] = []
    for row in rows:
        out.append(
            SimpleNamespace(
                stable_sample_id=str(row["stable_sample_id"]),
                split=str(row["split"]),
                category=str(row.get("category", "")),
                descriptors=row["descriptors"],
                p_gt=row["p_gt"],
                p_t=row["p_t"],
                phi_gt=row["phi_gt"],
                phi_t=row["phi_t"],
                anomaly_maps=row.get("anomaly_maps", {}),
                mask=row.get("mask"),
                descriptor_record_scientific_sha256=str(
                    row.get("descriptor_record_scientific_sha256", "")
                ),
                contribution_target_record_scientific_sha256=str(
                    row.get("contribution_target_record_scientific_sha256", "")
                ),
            )
        )
    return out


def scientific_v3_training_plan_payload(
    *,
    config: Mapping[str, Any],
    verified: v1_training.VerifiedB2DLCMTrainingInputs,
    roster_payload: Mapping[str, Any],
    adoption_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Accepted V3 plan — excludes operational/runtime fields per C2B contract."""

    def _record_ids(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "stable_sample_id": str(r["stable_sample_id"]),
                "descriptor_record_scientific_sha256": str(
                    r["descriptor_record_scientific_sha256"]
                ),
                "contribution_target_record_scientific_sha256": str(
                    r["contribution_target_record_scientific_sha256"]
                ),
                "split": str(r["split"]),
                "category": str(r.get("category", "")),
            }
            for r in rows
        ]

    return {
        "schema_version": OFFICIAL_PLAN_SCHEMA_VERSION,
        "architecture_contract_version": v3.ARCHITECTURE_CONTRACT_VERSION,
        "model_class_id": v3.MODEL_CLASS_ID,
        "v3_implementation_commit": V3_IMPLEMENTATION_COMMIT,
        "c2a_adoption_commit": C2A_ADOPTION_COMMIT,
        "c2a_contract_tag": C2A_CONTRACT_TAG,
        "accepted_upstream": dict(sorted(verified.verified_identities.items())),
        "training_records": _record_ids(verified.training_records),
        "calibration_records": _record_ids(verified.calibration_records),
        "development_record_ids": list(verified.evaluation_record_ids),
        "final_roster_identity": {
            "roster_scientific_sha256": roster_payload["roster_scientific_sha256"],
            "counts": dict(roster_payload["counts"]),
            "source_manifest_scientific_sha256": roster_payload.get(
                "source_manifest_scientific_sha256"
            ),
            "final_content_resolved": False,
            "paths_present": False,
        },
        "final_roster_adoption": {
            "adoption_scientific_sha256": adoption_payload.get("adoption_scientific_sha256"),
            "adopted_roster_scientific_sha256": adoption_payload.get(
                "adopted_roster_scientific_sha256",
                roster_payload["roster_scientific_sha256"],
            ),
            "selection_reused_without_change": True,
            "implementation_commit": adoption_payload.get("implementation_commit"),
        },
        "split_counts": {
            "training": 16,
            "calibration": 8,
            "development": 8,
            "final_roster": 16,
        },
        "architecture": {
            "candidate_layers": list(config["candidate_layers"]),
            "prediction_depths": list(config["prediction_depths"]),
            "descriptor_dimension": int(config["descriptor_dimension"]),
            "layer_embedding_dimension": int(config["layer_embedding_dimension"]),
            "depth_embedding_dimension": int(config["depth_embedding_dimension"]),
            "hidden_dimension": int(config["hidden_dimension"]),
            "dropout_probability": float(config["dropout_probability"]),
            "category_in_model": False,
        },
        "losses": {
            "gt_deployment_weight": 1.0,
            "gt_deployment_aggregation": "category_smooth_max",
            "smoothmax_tau": float(config.get("smoothmax_tau", v3.SMOOTHMAX_TAU)),
            "teacher_allocation_loss_weight": float(config["teacher_allocation_loss_weight"]),
            "gt_signed_loss_weight": float(config["gt_signed_loss_weight"]),
            "teacher_signed_loss_weight": float(config["teacher_signed_loss_weight"]),
            "huber_delta": float(config["huber_delta"]),
            "ranking_weight": float(config["ranking_weight"]),
            "ranking_tie_tolerance": float(config["ranking_tie_tolerance"]),
            "depth_weights": {str(k): float(v) for k, v in dict(config["depth_weights"]).items()},
            "selection": "constrained_worst_category",
        },
        "sampler": {
            "sampler_contract_version": str(
                config.get("sampler_contract_version", v3_training.SAMPLER_CONTRACT_VERSION)
            ),
            "training_categories": list(config.get("training_categories", ["bottle", "carpet"])),
            "per_category_per_batch": int(config.get("per_category_per_batch", 2)),
            "category_balanced_batch": True,
            "batch_composition": str(
                config.get("batch_composition", "2_bottle_2_carpet")
            ),
            "batches_per_epoch": 4,
            "batch_size": int(config["batch_size"]),
        },
        "optimizer": {
            "name": config["optimizer"],
            "maximum_learning_rate": float(config["maximum_learning_rate"]),
            "minimum_learning_rate": float(config["minimum_learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "betas": list(config["betas"]),
            "epsilon": float(config["epsilon"]),
            "gradient_clip_norm": float(config["gradient_clip_norm"]),
            "warmup_steps": int(config["warmup_steps"]),
            "maximum_optimizer_steps": int(config["maximum_optimizer_steps"]),
        },
        "lifecycle": {
            "batch_size": int(config["batch_size"]),
            "maximum_epochs": int(config["maximum_epochs"]),
            "patience": int(config["patience"]),
            "min_delta": float(config["min_delta"]),
            "seeds": list(config["seeds"]),
            "seed_summary_ddof": int(config["seed_summary_ddof"]),
            "training_dtype": config["training_dtype"],
            "amp_enabled": bool(config["amp_enabled"]),
            "canonical_training_device": config["canonical_training_device"],
            "visible_gpu_count": int(config["visible_gpu_count"]),
            "resume_boundary": config["resume_boundary"],
            "canonical_reproduction_runs": 1,
            "checkpoint_selection": "constrained_worst_category",
        },
        "development_gates": {
            "gt_macro_margin": 1e-5,
            "gt_per_category_slack": 1e-4,
            "localization": {
                "macro_pixel_ap_min_delta": 0.0,
                "macro_pixel_auroc_min_delta": -1e-4,
                "macro_aupro_min_delta": -1e-4,
                "category_floor": -1e-3,
            },
            "teacher_diagnostics_blocking": False,
            "formal_localization_adapter_id": v1_deployment.FORMAL_LOCALIZATION_ADAPTER_ID,
        },
        "final_protocols": {
            "final_materialization_enabled": False,
            "final_evaluation_enabled": False,
            "roster_paths_present": False,
            "final_content_resolved": False,
        },
        "artifact_schema_versions": {
            "scheduler": v1_training.SCHEDULER_CONTRACT_VERSION,
            "trace_chain": v1_training.TRACE_CHAIN_SCHEMA_VERSION,
            "environment": v1_training.ENVIRONMENT_CONTRACT_VERSION,
            "epoch_manifest": v1_training.EPOCH_MANIFEST_SCHEMA_VERSION,
            "v3_protocol": protocol.SCHEMA_VERSION,
        },
        "v1_immutable": v3.v1_immutable_identity(),
        "v2_immutable": v3.v2_immutable_identity(),
    }


def compute_accepted_v3_training_plan_scientific_sha256(
    *,
    config: Mapping[str, Any],
    verified: v1_training.VerifiedB2DLCMTrainingInputs,
    roster_payload: Mapping[str, Any],
    adoption_payload: Mapping[str, Any],
) -> str:
    payload = scientific_v3_training_plan_payload(
        config=config,
        verified=verified,
        roster_payload=roster_payload,
        adoption_payload=adoption_payload,
    )
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_frozen_roster_and_adoption(repo_root: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root)
    roster_payload = roster_adoption.load_and_verify_c1_roster(root)
    adoption_path = root / "docs/phase_b/b2_05c2_final_roster_adoption_manifest.json"
    adoption_payload = protocol.verify_json_receipt(adoption_path)
    roster_adoption.assert_adoption_matches_roster(adoption_payload, roster_payload)
    if adoption_payload.get("final_content_resolved") is not False:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "adoption must remain unresolved")
    if adoption_payload.get("paths_present") is not False:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "adoption paths must be absent")
    if adoption_payload.get("implementation_commit") != V3_IMPLEMENTATION_COMMIT:
        _fail(
            "B2_DLCM_V3_CONTRACT_MISMATCH",
            "adoption implementation_commit != V3_IMPLEMENTATION_COMMIT",
        )
    out_adoption = dict(adoption_payload)
    out_adoption["adoption_scientific_sha256"] = protocol.canonical_json_sha256(adoption_payload)
    out_adoption["adopted_roster_scientific_sha256"] = str(
        adoption_payload["source_roster_scientific_sha256"]
    )
    return dict(roster_payload), out_adoption


def verify_repository_identity_gate(
    *,
    config: Mapping[str, Any],
    repo_root: Path | str,
) -> dict[str, str]:
    root = Path(repo_root).resolve()
    if not bool(config.get("repository_identity_gate_enabled", True)):
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "repository identity gate must be enabled")
    expected_tag = str(config["expected_contract_tag"])
    expected_commit = str(config["expected_contract_commit"])
    expected_impl = str(config["expected_implementation_commit"])

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

    status = _git("status", "--porcelain")
    if status.returncode != 0:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", status.stderr.strip() or "git status failed")
    if status.stdout.strip():
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "worktree must be clean for official training")
    head = _git("rev-parse", "HEAD").stdout.strip()
    tag_proc = _git("rev-parse", f"{expected_tag}^{{commit}}")
    if tag_proc.returncode != 0:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", f"missing local tag {expected_tag}")
    tag_commit = tag_proc.stdout.strip()
    if tag_commit != expected_commit:
        _fail(
            "B2_DLCM_V3_CONTRACT_MISMATCH",
            f"tag {expected_tag} -> {tag_commit}, expected {expected_commit}",
        )
    if expected_impl != V3_IMPLEMENTATION_COMMIT:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "expected_implementation_commit pin mismatch")
    ancestor = _git("merge-base", "--is-ancestor", expected_commit, head)
    if ancestor.returncode != 0:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "HEAD does not descend from contract commit")
    return {
        "head": head,
        "expected_contract_tag": expected_tag,
        "expected_contract_commit": expected_commit,
        "expected_implementation_commit": expected_impl,
    }


def require_plan_sha_agreement(
    *,
    config: Mapping[str, Any],
    recomputed: str,
    cli_expected: str | None,
) -> None:
    pinned = config.get("expected_accepted_v3_training_plan_sha256")
    if pinned is not None and pinned != recomputed:
        _fail(
            "B2_DLCM_V3_CONTRACT_MISMATCH",
            "config.expected_accepted_v3_training_plan_sha256 != recomputed plan",
        )
    if cli_expected is not None and cli_expected != recomputed:
        _fail(
            "B2_DLCM_V3_CONTRACT_MISMATCH",
            "CLI --expected-plan-sha256 != recomputed plan",
        )
    if pinned is not None and cli_expected is not None and pinned != cli_expected:
        _fail(
            "B2_DLCM_V3_CONTRACT_MISMATCH",
            "CLI --expected-plan-sha256 != config pin",
        )


def official_v3_dry_run(
    *,
    config: Mapping[str, Any],
    descriptor_manifest: Path | str,
    descriptor_root: Path | str,
    contribution_target_manifest: Path | str,
    contribution_target_root: Path | str,
    output_root: Path | str,
    seed: int,
    expected_plan_sha256: str | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Verify real upstream + C2A identities + plan hash; no model/RNG/writes/eval."""

    protocol.reject_bypass_flags(config)
    if config.get("contract_stage") != "b2_05c2b":
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "official dry-run requires b2_05c2b")
    if config.get("real_training_enabled") is not True:
        _fail("B2_DLCM_V3_REAL_TRAINING_NOT_ENABLED", "dry-run requires real_training_enabled=true")
    if config.get("final_materialization_enabled") is not False:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "final materialization must be disabled")
    if config.get("final_evaluation_enabled") is not False:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "final evaluation must be disabled")
    if abs(float(config.get("smoothmax_tau", -1)) - v3.SMOOTHMAX_TAU) > 0:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "smoothmax_tau must be 0.05")

    output_root = Path(output_root)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    if repo_root is not None:
        verify_repository_identity_gate(config=config, repo_root=root)

    roster_payload, adoption_payload = load_frozen_roster_and_adoption(root)
    if roster_payload.get("final_content_resolved") is not False or roster_payload.get(
        "paths_present"
    ):
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "roster must remain sealed in dry-run")

    verified = v1_training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=descriptor_manifest,
        descriptor_root=descriptor_root,
        contribution_target_manifest=contribution_target_manifest,
        contribution_target_root=contribution_target_root,
        accepted_upstream=dict(config["accepted_upstream"]),
        evaluation_unlocked=False,
    )
    plan_sha = compute_accepted_v3_training_plan_scientific_sha256(
        config=config,
        verified=verified,
        roster_payload=roster_payload,
        adoption_payload=adoption_payload,
    )
    pinned = config.get("expected_accepted_v3_training_plan_sha256")
    if pinned is not None or expected_plan_sha256 is not None:
        require_plan_sha_agreement(
            config=config, recomputed=plan_sha, cli_expected=expected_plan_sha256
        )

    return {
        "mode": "dry_run",
        "status": "plan_validated",
        "artifact_written": False,
        "run_directory_created": False,
        "real_training_started": False,
        "development_evaluation_started": False,
        "final_content_resolved": False,
        "final_materialization_started": False,
        "final_evaluation_started": False,
        "teacher_forward_count": 0,
        "training_records": 16,
        "training_categories": list(config.get("training_categories", ["bottle", "carpet"])),
        "batch_size": int(config["batch_size"]),
        "batches_per_epoch": 4,
        "batch_composition": str(config.get("batch_composition", "2_bottle_2_carpet")),
        "calibration_records": 8,
        "development_records_declared": 8,
        "final_roster_records_declared": 16,
        "final_records_loaded": 0,
        "seeds": list(config["seeds"]),
        "canonical_reproduction_runs": 1,
        "smoothmax_tau": float(config.get("smoothmax_tau", v3.SMOOTHMAX_TAU)),
        "checkpoint_selection": "constrained_worst_category",
        "accepted_v3_training_plan_scientific_sha256": plan_sha,
        "seed": int(seed),
        "contract_stage": "b2_05c2b",
        "output_root_exists_before": output_root.exists(),
        "roster_scientific_sha256": roster_payload["roster_scientific_sha256"],
        "adoption_scientific_sha256": adoption_payload.get("adoption_scientific_sha256"),
    }


def run_official_v3_seed_training(
    *,
    output_root: Path,
    seed: int,
    training_records: Sequence[Any],
    calibration_records: Sequence[Any],
    environment_contract: Mapping[str, Any],
    maximum_epochs: int = 500,
    patience: int = 50,
    batch_size: int = 4,
    smoothmax_tau: float = v3.SMOOTHMAX_TAU,
    device: str = "cuda",
) -> dict[str, Any]:
    eval_placeholders = [
        SimpleNamespace(split="evaluation", stable_sample_id=f"eval-placeholder-{i}")
        for i in range(8)
    ]
    records = list(training_records) + list(calibration_records) + eval_placeholders
    result = v3_training.run_v3_contract_training(
        output_root=Path(output_root),
        seed=seed,
        records=records,
        maximum_epochs=maximum_epochs,
        patience=patience,
        device=device,
        batch_size=batch_size,
        smoothmax_tau=smoothmax_tau,
        environment_contract=environment_contract,
        allow_existing_output=True,
        mark_real_training_started=True,
    )
    result = dict(result)
    result["official"] = True
    return result
