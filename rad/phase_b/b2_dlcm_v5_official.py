"""B2-05C4B official V5 uniform-anchored calibration orchestration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

import torch

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_deployment as v1_deployment
from rad.phase_b import b2_dlcm_training as v1_training
from rad.phase_b import b2_dlcm_v4 as v4
from rad.phase_b import b2_dlcm_v5 as v5
from rad.phase_b import b2_dlcm_v5_protocol as protocol
from rad.phase_b import b2_dlcm_v5_roster_adoption as roster_adoption

OFFICIAL_PLAN_SCHEMA_VERSION = "b2_dlcm_v5_accepted_calibration_plan_v1"
C4A_CONTRACT_TAG = "b2-dlcm-uniform-anchored-contract-v5"
C4A_ADOPTION_COMMIT = "017a76c7586107dd83db46959ab74a7057b585c4"
V5_IMPLEMENTATION_COMMIT = "62b2de3c5c144f217f65353e45fd061fa1e60e05"
C3_H_DEPLOY = "28896ef8c46b54240e8664c7236de4397defa3e877daa5a709249562f716449d"
C3_UNQUALIFIED_TAG = "b2-dlcm-uniform-relative-unqualified-evidence-v1"
C3_UNQUALIFIED_COMMIT = "a1447bdabdd7f54eb7883b717dfadc3da906da5b"
EXPECTED_ENVIRONMENT_IDENTITY = (
    "67677c4e9bb83475f7adc03294437bdd104a693e0465e107d3860096a9f03056"
)


class B2DLCMV5OfficialError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5OfficialError(code, detail)


def load_frozen_roster_and_adoption(repo_root: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root)
    roster_payload = roster_adoption.load_and_verify_c1_roster(root)
    adoption_path = root / "docs/phase_b/b2_05c4_final_roster_adoption_manifest.json"
    adoption_payload = protocol.verify_json_receipt(adoption_path)
    roster_adoption.assert_adoption_matches_roster(adoption_payload, roster_payload)
    if adoption_payload.get("final_content_resolved") is not False:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "adoption must remain unresolved")
    if adoption_payload.get("paths_present") is not False:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "adoption paths must be absent")
    if adoption_payload.get("implementation_commit") != V5_IMPLEMENTATION_COMMIT:
        _fail(
            "B2_DLCM_V5_CONTRACT_MISMATCH",
            "adoption implementation_commit != V5_IMPLEMENTATION_COMMIT",
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
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "repository identity gate must be enabled")
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
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", status.stderr.strip() or "git status failed")
    if status.stdout.strip():
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "worktree must be clean for official calibration")
    head = _git("rev-parse", "HEAD").stdout.strip()
    tag_proc = _git("rev-parse", f"{expected_tag}^{{commit}}")
    if tag_proc.returncode != 0:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", f"missing local tag {expected_tag}")
    tag_commit = tag_proc.stdout.strip()
    if tag_commit != expected_commit:
        _fail(
            "B2_DLCM_V5_CONTRACT_MISMATCH",
            f"tag {expected_tag} -> {tag_commit}, expected {expected_commit}",
        )
    if expected_impl != V5_IMPLEMENTATION_COMMIT:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "expected_implementation_commit pin mismatch")
    ancestor = _git("merge-base", "--is-ancestor", expected_commit, head)
    if ancestor.returncode != 0:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "HEAD does not descend from contract commit")
    return {
        "head": head,
        "expected_contract_tag": expected_tag,
        "expected_contract_commit": expected_commit,
        "expected_implementation_commit": expected_impl,
    }


def scientific_v5_calibration_plan_payload(
    *,
    config: Mapping[str, Any],
    verified: v1_training.VerifiedB2DLCMTrainingInputs,
    roster_payload: Mapping[str, Any],
    adoption_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Accepted V5 calibration plan — excludes paths/timestamps/runtime/GPU labels."""

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
        "architecture_contract_version": v5.ARCHITECTURE_CONTRACT_VERSION,
        "model_class_id": v5.MODEL_CLASS_ID,
        "calibration_contract_version": v5.CALIBRATION_CONTRACT_VERSION,
        "v5_implementation_commit": V5_IMPLEMENTATION_COMMIT,
        "c4a_adoption_commit": C4A_ADOPTION_COMMIT,
        "c4a_contract_tag": C4A_CONTRACT_TAG,
        "source_deployment_identity": str(
            config.get("source_deployment_identity", C3_H_DEPLOY)
        ),
        "source_canonical_seed": int(config.get("source_canonical_seed", 17)),
        "source_canonical_reproduction_required": True,
        "c3_unqualified_tag": C3_UNQUALIFIED_TAG,
        "c3_unqualified_commit": C3_UNQUALIFIED_COMMIT,
        "accepted_upstream": dict(sorted(verified.verified_identities.items())),
        "calibration_records": _record_ids(verified.calibration_records),
        "development_record_ids": list(verified.evaluation_record_ids),
        "final_roster_identity": {
            "roster_scientific_sha256": roster_payload["roster_scientific_sha256"],
            "counts": dict(roster_payload["counts"]),
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
            "calibration": 8,
            "development": 8,
            "final_roster": 16,
        },
        "beta_grid": {
            "start": "0.00",
            "stop": "1.00",
            "step": "0.01",
            "count": 101,
        },
        "calibration_objective": "depth24_leave_one_out_worst_category_relative_regret",
        "calibration_eligibility": "original_macro_and_per_category_gt_gates",
        "tie_break": "larger_beta_then_lower_macro_then_lower_grid_index",
        "loo_depth": 24,
        "loo_fold_count": 8,
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
        "flags": {
            "real_training_enabled": False,
            "calibration_enabled": True,
            "development_enabled": True,
        },
        "v1_immutable": v5.v1_immutable_identity(),
        "v2_immutable": v5.v2_immutable_identity(),
        "v3_immutable": v5.v3_immutable_identity(),
        "v4_immutable": v5.v4_immutable_identity(),
    }


def compute_accepted_v5_calibration_plan_scientific_sha256(
    *,
    config: Mapping[str, Any],
    verified: v1_training.VerifiedB2DLCMTrainingInputs,
    roster_payload: Mapping[str, Any],
    adoption_payload: Mapping[str, Any],
) -> str:
    payload = scientific_v5_calibration_plan_payload(
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


def require_plan_sha_agreement(
    *,
    config: Mapping[str, Any],
    recomputed: str,
    cli_expected: str | None,
) -> None:
    pinned = config.get("expected_accepted_v5_calibration_plan_sha256")
    if pinned is not None and pinned != recomputed:
        _fail(
            "B2_DLCM_V5_CONTRACT_MISMATCH",
            "config.expected_accepted_v5_calibration_plan_sha256 != recomputed plan",
        )
    if cli_expected is not None and cli_expected != recomputed:
        _fail(
            "B2_DLCM_V5_CONTRACT_MISMATCH",
            "CLI --expected-plan-sha256 != recomputed plan",
        )
    if pinned is not None and cli_expected is not None and pinned != cli_expected:
        _fail(
            "B2_DLCM_V5_CONTRACT_MISMATCH",
            "CLI --expected-plan-sha256 != config pin",
        )


def load_c3_deployment_checkpoint(path: Path | str) -> dict[str, Any]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "C3 deployment checkpoint must be a dict")
    h_deploy = str(ckpt.get("H_deploy", ""))
    if h_deploy != C3_H_DEPLOY:
        _fail(
            "B2_DLCM_V5_CONTRACT_MISMATCH",
            f"C3 H_deploy mismatch: {h_deploy} != {C3_H_DEPLOY}",
        )
    if ckpt.get("upstream", {}).get("accepted") is True:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "C3 deployment must remain unaccepted")
    return ckpt


def load_c3_deployment_trunk(checkpoint: Mapping[str, Any]) -> v4.B2DLCMV4DeploymentTrunk:
    trunk = v4.B2DLCMV4DeploymentTrunk(seed=None, initialize=False)
    state = checkpoint["state_dict"]
    trunk.load_state_dict(state, strict=True)
    trunk.eval()
    return trunk


def materialize_calibration_records_with_dynamic_weights(
    *,
    calibration_records: Sequence[Mapping[str, Any]],
    trunk: v4.B2DLCMV4DeploymentTrunk,
    depth: int = 24,
) -> list[dict[str, Any]]:
    """Build V5 calibration rows: p_gt + C3 dynamic weights at Depth-24."""

    out: list[dict[str, Any]] = []
    trunk.eval()
    with torch.no_grad():
        for row in calibration_records:
            desc = row["descriptors"][int(depth)]
            if not isinstance(desc, torch.Tensor):
                _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", "descriptor tensor missing")
            if desc.ndim == 3:
                desc = desc.reshape(desc.shape[-2], desc.shape[-1])
            x = desc.unsqueeze(0).to(dtype=torch.float32)
            _logits, weights = trunk.forward(x, prediction_depth=int(depth))
            p_gt = row["p_gt"][int(depth)].to(torch.float32)
            if p_gt.ndim == 2:
                p_gt = p_gt.reshape(-1)
            out.append(
                {
                    "stable_sample_id": str(row["stable_sample_id"]),
                    "category": str(row["category"]),
                    "depth": int(depth),
                    "p_gt": p_gt.detach().cpu().contiguous(),
                    "dynamic_weights": weights.reshape(-1).detach().cpu().contiguous(),
                    "descriptor_record_scientific_sha256": str(
                        row.get("descriptor_record_scientific_sha256", "")
                    ),
                    "contribution_target_record_scientific_sha256": str(
                        row.get("contribution_target_record_scientific_sha256", "")
                    ),
                }
            )
    return out


def official_v5_dry_run(
    *,
    config: Mapping[str, Any],
    descriptor_manifest: Path | str,
    descriptor_root: Path | str,
    contribution_target_manifest: Path | str,
    contribution_target_root: Path | str,
    output_root: Path | str,
    source_deployment_checkpoint: Path | str,
    expected_plan_sha256: str | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Verify real upstream + C4A identities + plan hash; no calibration/dev/writes."""

    protocol.reject_bypass_flags(config)
    if config.get("contract_stage") != "b2_05c4b":
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "official dry-run requires b2_05c4b")
    if config.get("real_training_enabled") is not False:
        _fail("B2_DLCM_V5_TRAINING_FORBIDDEN", "real_training_enabled must be false")
    if config.get("calibration_enabled") is not True:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "calibration_enabled must be true")
    if config.get("final_materialization_enabled") is not False:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "final materialization must be disabled")
    if config.get("final_evaluation_enabled") is not False:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "final evaluation must be disabled")

    output_root = Path(output_root)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    if repo_root is not None:
        verify_repository_identity_gate(config=config, repo_root=root)

    roster_payload, adoption_payload = load_frozen_roster_and_adoption(root)
    if roster_payload.get("final_content_resolved") is not False or roster_payload.get(
        "paths_present"
    ):
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "roster must remain sealed in dry-run")

    # Verify C3 deployment identity without mutating tensors.
    c3_ckpt = load_c3_deployment_checkpoint(source_deployment_checkpoint)
    digests_before = {
        name: v1.tensor_sha256(tensor) for name, tensor in sorted(c3_ckpt["state_dict"].items())
    }

    verified = v1_training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=descriptor_manifest,
        descriptor_root=descriptor_root,
        contribution_target_manifest=contribution_target_manifest,
        contribution_target_root=contribution_target_root,
        accepted_upstream=dict(config["accepted_upstream"]),
        evaluation_unlocked=False,
    )
    if len(verified.calibration_records) != 8:
        _fail("B2_DLCM_V5_CALIBRATION_INPUT_INVALID", "expected 8 calibration records")

    plan_sha = compute_accepted_v5_calibration_plan_scientific_sha256(
        config=config,
        verified=verified,
        roster_payload=roster_payload,
        adoption_payload=adoption_payload,
    )
    pinned = config.get("expected_accepted_v5_calibration_plan_sha256")
    if pinned is not None or expected_plan_sha256 is not None:
        require_plan_sha_agreement(
            config=config, recomputed=plan_sha, cli_expected=expected_plan_sha256
        )

    digests_after = {
        name: v1.tensor_sha256(tensor) for name, tensor in sorted(c3_ckpt["state_dict"].items())
    }
    if digests_before != digests_after:
        _fail("B2_DLCM_V5_CONTRACT_MISMATCH", "C3 checkpoint tensors mutated during dry-run")

    return {
        "mode": "dry_run",
        "status": "plan_validated",
        "artifact_written": False,
        "run_directory_created": False,
        "real_training_started": False,
        "calibration_started": False,
        "development_evaluation_started": False,
        "final_content_resolved": False,
        "final_materialization_started": False,
        "final_evaluation_started": False,
        "teacher_forward_count": 0,
        "beta_grid_count": 101,
        "beta_grid_start": "0.00",
        "beta_grid_stop": "1.00",
        "beta_grid_step": "0.01",
        "calibration_records": 8,
        "loo_fold_count": 8,
        "development_records_declared": 8,
        "final_roster_records_declared": 16,
        "final_records_loaded": 0,
        "source_deployment_identity": C3_H_DEPLOY,
        "source_canonical_seed": 17,
        "accepted_v5_calibration_plan_scientific_sha256": plan_sha,
        "contract_stage": "b2_05c4b",
        "output_root_exists_before": output_root.exists(),
        "roster_scientific_sha256": roster_payload["roster_scientific_sha256"],
        "adoption_scientific_sha256": adoption_payload.get("adoption_scientific_sha256"),
    }
