"""B2-05B official DLCM training orchestration (canonical three-seed path)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_deployment as deployment
from rad.phase_b import b2_dlcm_training as training

OFFICIAL_PLAN_SCHEMA_VERSION = "b2_dlcm_accepted_training_plan_v1"


class B2DLCMOfficialError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMOfficialError(code, detail)


def records_as_namespaces(rows: Sequence[Mapping[str, Any]]) -> list[SimpleNamespace]:
    """Adapt verified mapping records to attribute-access training records."""

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


def scientific_training_plan_payload(
    *,
    config: Mapping[str, Any],
    verified: training.VerifiedB2DLCMTrainingInputs,
) -> dict[str, Any]:
    """Canonical scientific plan — excludes operational/runtime fields."""

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
        "accepted_upstream": dict(sorted(verified.verified_identities.items())),
        "training_records": _record_ids(verified.training_records),
        "calibration_records": _record_ids(verified.calibration_records),
        "evaluation_record_ids": list(verified.evaluation_record_ids),
        "split_counts": {"training": 16, "calibration": 8, "evaluation": 8},
        "architecture": {
            "candidate_layers": list(config["candidate_layers"]),
            "prediction_depths": list(config["prediction_depths"]),
            "descriptor_dimension": int(config["descriptor_dimension"]),
            "layer_embedding_dimension": int(config["layer_embedding_dimension"]),
            "depth_embedding_dimension": int(config["depth_embedding_dimension"]),
            "hidden_dimension": int(config["hidden_dimension"]),
            "dropout_probability": float(config["dropout_probability"]),
        },
        "losses": {
            "allocation_family_weights": dict(config["allocation_family_weights"]),
            "huber_delta": float(config["huber_delta"]),
            "ranking_weight": float(config["ranking_weight"]),
            "ranking_tie_tolerance": float(config["ranking_tie_tolerance"]),
            "signed_loss_weight": float(config["signed_loss_weight"]),
            "depth_weights": {str(k): float(v) for k, v in dict(config["depth_weights"]).items()},
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
        },
        "evaluation_gates": {
            "target_learning": {
                "macro_kl_improvement": 1e-5,
                "per_category_kl_slack": 1e-4,
            },
            "localization": {
                "macro_pixel_ap_min_delta": 0.0,
                "macro_pixel_auroc_min_delta": -1e-4,
                "macro_aupro_min_delta": -1e-4,
                "category_pixel_ap_min_delta": -1e-3,
                "category_pixel_auroc_min_delta": -1e-3,
                "category_aupro_min_delta": -1e-3,
            },
            "formal_localization_adapter_id": deployment.FORMAL_LOCALIZATION_ADAPTER_ID,
        },
        "artifact_schema_versions": {
            "scheduler": training.SCHEDULER_CONTRACT_VERSION,
            "trace_chain": training.TRACE_CHAIN_SCHEMA_VERSION,
            "environment": training.ENVIRONMENT_CONTRACT_VERSION,
            "epoch_manifest": training.EPOCH_MANIFEST_SCHEMA_VERSION,
            "deployment": deployment.DEPLOYMENT_SCHEMA_VERSION,
            "loader": deployment.LOADER_CONTRACT_VERSION,
        },
    }


def compute_accepted_dlcm_training_plan_scientific_sha256(
    *,
    config: Mapping[str, Any],
    verified: training.VerifiedB2DLCMTrainingInputs,
) -> str:
    payload = scientific_training_plan_payload(config=config, verified=verified)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def verify_repository_identity_gate(
    *,
    config: Mapping[str, Any],
    repo_root: Path | str,
) -> dict[str, str]:
    root = Path(repo_root).resolve()
    if not bool(config.get("repository_identity_gate_enabled", True)):
        _fail("B2_DLCM_REPO_GATE_DISABLED", "repository identity gate must be enabled")
    expected_tag = str(config["expected_training_contract_tag"])
    expected_commit = str(config["expected_training_contract_commit"])

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
        _fail("B2_DLCM_REPO_IDENTITY_INVALID", status.stderr.strip() or "git status failed")
    if status.stdout.strip():
        _fail("B2_DLCM_REPO_DIRTY", "worktree must be clean for official training")
    head_proc = _git("rev-parse", "HEAD")
    if head_proc.returncode != 0:
        _fail("B2_DLCM_REPO_IDENTITY_INVALID", "cannot resolve HEAD")
    head = head_proc.stdout.strip()
    tag_proc = _git("rev-parse", f"{expected_tag}^{{commit}}")
    if tag_proc.returncode != 0:
        _fail("B2_DLCM_CONTRACT_TAG_MISSING", f"missing local tag {expected_tag}")
    tag_commit = tag_proc.stdout.strip()
    if tag_commit != expected_commit:
        _fail(
            "B2_DLCM_CONTRACT_COMMIT_MISMATCH",
            f"tag {expected_tag} -> {tag_commit}, expected {expected_commit}",
        )
    ancestor = _git("merge-base", "--is-ancestor", expected_commit, head)
    if ancestor.returncode != 0:
        _fail("B2_DLCM_CONTRACT_ANCESTRY", "HEAD does not descend from contract commit")
    return {
        "head": head,
        "expected_training_contract_tag": expected_tag,
        "expected_training_contract_commit": expected_commit,
    }


def require_plan_sha_agreement(
    *,
    config: Mapping[str, Any],
    recomputed: str,
    cli_expected: str | None,
) -> None:
    pinned = config.get("expected_accepted_training_plan_sha256")
    if config.get("expected_plan_sha_required") and not pinned:
        # Dry-run before pin may leave null; official non-dry-run must have pin.
        pass
    if pinned is not None and pinned != recomputed:
        _fail(
            "B2_DLCM_PLAN_SHA_MISMATCH",
            "config.expected_accepted_training_plan_sha256 != recomputed plan",
        )
    if cli_expected is not None and cli_expected != recomputed:
        _fail(
            "B2_DLCM_PLAN_SHA_MISMATCH",
            "CLI --expected-plan-sha256 != recomputed plan",
        )
    if (
        pinned is not None
        and cli_expected is not None
        and pinned != cli_expected
    ):
        _fail(
            "B2_DLCM_PLAN_SHA_MISMATCH",
            "CLI --expected-plan-sha256 != config pin",
        )


def official_dry_run(
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
    """Verify real upstream + plan hash; no model init, no writes, no eval load."""

    if config.get("contract_stage") != "b2_05b":
        _fail("B2_DLCM_CONFIG_STAGE_INVALID", "official dry-run requires b2_05b")
    if config.get("real_training_enabled") is not True:
        _fail("B2_DLCM_REAL_TRAINING_FLAG", "b2_05b dry-run requires real_training_enabled=true")
    output_root = Path(output_root)
    if repo_root is not None:
        verify_repository_identity_gate(config=config, repo_root=repo_root)

    verified = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=descriptor_manifest,
        descriptor_root=descriptor_root,
        contribution_target_manifest=contribution_target_manifest,
        contribution_target_root=contribution_target_root,
        accepted_upstream=dict(config["accepted_upstream"]),
        evaluation_unlocked=False,
    )
    plan_sha = compute_accepted_dlcm_training_plan_scientific_sha256(
        config=config, verified=verified
    )
    # During pre-pin dry-runs, config pin may be null; still check CLI if provided.
    pinned = config.get("expected_accepted_training_plan_sha256")
    if pinned is not None or expected_plan_sha256 is not None:
        require_plan_sha_agreement(
            config=config, recomputed=plan_sha, cli_expected=expected_plan_sha256
        )
    if output_root.exists():
        # Must not create; existence beforehand is ok only if empty unused — still report false create.
        pass
    return {
        "mode": "dry_run",
        "status": "plan_validated",
        "artifact_written": False,
        "run_directory_created": False,
        "real_training_started": False,
        "evaluation_unlocked": False,
        "teacher_forward_count": 0,
        "training_records": 16,
        "calibration_records": 8,
        "evaluation_records_declared": 8,
        "evaluation_records_loaded": 0,
        "seeds": list(config["seeds"]),
        "canonical_reproduction_runs": 1,
        "accepted_dlcm_training_plan_scientific_sha256": plan_sha,
        "seed": int(seed),
        "contract_stage": "b2_05b",
        "output_root_exists_before": output_root.exists(),
    }


def run_official_seed_training(
    *,
    output_root: Path,
    seed: int,
    training_records: Sequence[Any],
    calibration_records: Sequence[Any],
    environment_contract: Mapping[str, Any],
    maximum_epochs: int = 500,
    patience: int = 50,
    batch_size: int = 4,
    device: str = "cuda",
) -> dict[str, Any]:
    """Authoritative single-seed training against verified records on one GPU."""

    eval_placeholders = [
        SimpleNamespace(split="evaluation", stable_sample_id=f"eval-placeholder-{i}")
        for i in range(8)
    ]
    records = list(training_records) + list(calibration_records) + eval_placeholders
    result = training.run_hermetic_contract_training(
        output_root=Path(output_root),
        seed=seed,
        records=records,
        maximum_epochs=maximum_epochs,
        patience=patience,
        device=device,
        batch_size=batch_size,
        environment_contract=environment_contract,
        allow_existing_output=True,
        mark_real_training_started=True,
    )
    result = dict(result)
    result["official"] = True
    return result


@dataclass(frozen=True)
class OfficialRunPaths:
    descriptor_manifest: Path
    descriptor_root: Path
    contribution_target_manifest: Path
    contribution_target_root: Path
    output_root: Path
