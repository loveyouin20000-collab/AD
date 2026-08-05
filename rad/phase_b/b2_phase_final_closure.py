"""B2 phase final closure helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn


class B2PhaseFinalClosureError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2PhaseFinalClosureError(code, detail)


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _fail("B2_PHASE_FINAL_CLOSURE_JSON_INVALID", f"{path} must contain a JSON object")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        _fail("B2_PHASE_FINAL_CLOSURE_JSON_INVALID", f"{key} must be an object")
    return value


def _require_schema(payload: Mapping[str, Any], expected: str) -> None:
    if payload.get("schema_version") != expected:
        _fail("B2_PHASE_FINAL_CLOSURE_SCHEMA_MISMATCH", expected)


def _require_equal(actual: Any, expected: Any, key: str) -> None:
    if actual != expected:
        _fail("B2_PHASE_FINAL_CLOSURE_IDENTITY_MISMATCH", f"{key} mismatch")


def _require_false(value: Any, key: str) -> None:
    if value is not False:
        _fail("B2_PHASE_FINAL_CLOSURE_BOUNDARY_VIOLATION", f"{key} must be false")


def _require_true(value: Any, key: str) -> None:
    if value is not True:
        _fail("B2_PHASE_FINAL_CLOSURE_BOUNDARY_VIOLATION", f"{key} must be true")


def build_phase_final_closure_manifest(
    *,
    accepted_gate_evidence: Mapping[str, Any],
    reference_packaging_evidence: Mapping[str, Any],
    prerequisite_evidence: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    qualification_decision: Mapping[str, Any],
    accepted_lse_manifest: Mapping[str, Any],
    accepted_lse_receipt: Mapping[str, Any],
    accepted_lse_evidence: Mapping[str, Any],
    git_sha: str,
    tracked_pt_count: int,
    pushed: bool,
    pr_opened: bool,
) -> dict[str, Any]:
    _require_schema(accepted_gate_evidence, "b2_06a_lse_accepted_gate_preflight_evidence_v1")
    _require_schema(reference_packaging_evidence, "b2_06b_accepted_v5_reference_packaging_evidence_v1")
    _require_schema(prerequisite_evidence, "b2_06c_lse_prerequisite_materialization_evidence_v1")
    _require_schema(training_evidence, "b2_06d_lse_first_controlled_run_evidence_v1")
    _require_schema(qualification_decision, "b2_06e_lse_qualification_decision_v1")
    _require_schema(accepted_lse_manifest, "b2_06f_lse_accepted_artifact_manifest_v1")
    _require_schema(accepted_lse_receipt, "b2_06f_lse_accepted_artifact_closure_receipt_v1")
    _require_schema(accepted_lse_evidence, "b2_06f_accepted_lse_artifact_closure_evidence_v1")

    if qualification_decision.get("verdict") != "qualified":
        _fail("B2_PHASE_FINAL_CLOSURE_LSE_NOT_QUALIFIED", "06E verdict must be qualified")
    _require_false(qualification_decision.get("accepted_artifact_generated"), "06E.accepted_artifact_generated")
    _require_true(accepted_lse_manifest.get("lse_qualified"), "06F.manifest.lse_qualified")
    _require_true(
        accepted_lse_manifest.get("accepted_artifact_generated"),
        "06F.manifest.accepted_artifact_generated",
    )
    _require_false(accepted_lse_manifest.get("training_started"), "06F.manifest.training_started")
    _require_false(accepted_lse_manifest.get("evaluation_started"), "06F.manifest.evaluation_started")
    _require_true(
        accepted_lse_receipt.get("accepted_artifact_generated"),
        "06F.receipt.accepted_artifact_generated",
    )
    _require_false(accepted_lse_receipt.get("training_started"), "06F.receipt.training_started")
    _require_false(accepted_lse_receipt.get("evaluation_started"), "06F.receipt.evaluation_started")

    if tracked_pt_count != 0:
        _fail("B2_PHASE_FINAL_CLOSURE_TRACKED_PT", "tracked .pt files must remain zero")
    _require_false(pushed, "push_performed")
    _require_false(pr_opened, "pr_opened")

    gate_preflight = _mapping(accepted_gate_evidence, "preflight_result")
    gate_ids = _mapping(accepted_gate_evidence, "frozen_scientific_identities")
    gate_boundaries = _mapping(accepted_gate_evidence, "stopped_boundaries")
    _require_true(gate_preflight.get("accepted_gate_passed"), "06A.accepted_gate_passed")
    _require_false(gate_preflight.get("training_started"), "06A.training_started")
    _require_false(gate_boundaries.get("push_performed"), "06A.push_performed")
    _require_false(gate_boundaries.get("pull_request_opened"), "06A.pull_request_opened")

    packaging_ids = _mapping(reference_packaging_evidence, "frozen_identities")
    packaging_boundary = _mapping(reference_packaging_evidence, "boundary")
    _require_false(packaging_boundary.get("accepted_identity_changed"), "06B.accepted_identity_changed")
    _require_false(packaging_boundary.get("final_re_evaluated"), "06B.final_re_evaluated")
    _require_false(packaging_boundary.get("lse_training_started"), "06B.lse_training_started")
    _require_false(packaging_boundary.get("lse_checkpoint_generated"), "06B.lse_checkpoint_generated")
    _require_false(packaging_boundary.get("push_performed"), "06B.push_performed")

    prerequisite_gate = _mapping(prerequisite_evidence, "accepted_gate")
    prerequisite_boundaries = _mapping(prerequisite_evidence, "boundaries")
    _require_true(prerequisite_gate.get("ready"), "06C.accepted_gate.ready")
    _require_true(prerequisite_gate.get("accepted_gate_passed"), "06C.accepted_gate_passed")
    _require_false(prerequisite_gate.get("training_started"), "06C.training_started")
    _require_false(prerequisite_boundaries.get("lse_training_started"), "06C.lse_training_started")
    _require_false(prerequisite_boundaries.get("lse_checkpoint_generated"), "06C.lse_checkpoint_generated")
    _require_true(
        prerequisite_boundaries.get("accepted_v5_artifact_unchanged"),
        "06C.accepted_v5_artifact_unchanged",
    )
    _require_false(prerequisite_boundaries.get("pushed"), "06C.pushed")
    _require_false(prerequisite_boundaries.get("pr_opened"), "06C.pr_opened")

    training_gate = _mapping(training_evidence, "accepted_gate")
    training_run = _mapping(training_evidence, "run")
    training_artifacts = _mapping(training_evidence, "artifacts")
    training_checkpoint = _mapping(training_artifacts, "best_checkpoint")
    training_receipt = _mapping(training_artifacts, "training_receipt")
    training_boundaries = _mapping(training_evidence, "boundaries")
    training_unlock = _mapping(training_evidence, "unlock")
    _require_true(training_gate.get("accepted_gate_passed"), "06D.accepted_gate_passed")
    _require_true(training_gate.get("ready"), "06D.accepted_gate.ready")
    _require_false(training_boundaries.get("final_content_accessed"), "06D.final_content_accessed")
    _require_false(training_boundaries.get("lse_checkpoint_tracked"), "06D.lse_checkpoint_tracked")
    _require_false(training_boundaries.get("push"), "06D.push")
    _require_false(training_boundaries.get("pr"), "06D.pr")
    _require_false(training_checkpoint.get("tracked"), "06D.best_checkpoint.tracked")

    lse_evidence_boundary = _mapping(accepted_lse_evidence, "boundary")
    lse_evidence_upstream = _mapping(accepted_lse_evidence, "upstream")
    _require_false(lse_evidence_boundary.get("training_started_in_06f"), "06F.training_started_in_06f")
    _require_false(lse_evidence_boundary.get("evaluation_started_in_06f"), "06F.evaluation_started_in_06f")
    _require_false(lse_evidence_boundary.get("final_content_accessed_in_06f"), "06F.final_content_accessed_in_06f")
    _require_false(lse_evidence_boundary.get("lse_checkpoint_generated_in_06f"), "06F.lse_checkpoint_generated_in_06f")
    _require_false(lse_evidence_boundary.get("pushed"), "06F.pushed")
    _require_false(lse_evidence_boundary.get("pr_opened"), "06F.pr_opened")
    if lse_evidence_boundary.get("tracked_pt_files") != 0:
        _fail("B2_PHASE_FINAL_CLOSURE_TRACKED_PT", "06F evidence tracked .pt must be zero")

    accepted_dlcm_identity = str(accepted_lse_manifest.get("accepted_dlcm_identity"))
    v5_deployment_identity = str(accepted_lse_manifest.get("v5_deployment_identity"))
    accepted_lse_identity = str(accepted_lse_manifest.get("accepted_lse_identity"))
    training_receipt_identity = str(accepted_lse_manifest.get("training_receipt_identity"))
    unlock_identity = str(accepted_lse_manifest.get("unlock_identity"))
    qualification_identity = str(accepted_lse_manifest.get("H_lse_qualification"))
    lse_checkpoint_sha = str(accepted_lse_manifest.get("accepted_lse_checkpoint_sha256"))
    selector_hash = str(accepted_lse_manifest.get("selector_signal_layout_hash"))

    _require_equal(gate_ids.get("accepted_v5_identity"), accepted_dlcm_identity, "06A.accepted_v5_identity")
    _require_equal(packaging_ids.get("accepted_identity"), accepted_dlcm_identity, "06B.accepted_identity")
    _require_equal(prerequisite_gate.get("accepted_identity"), accepted_dlcm_identity, "06C.accepted_identity")
    _require_equal(training_gate.get("accepted_identity"), accepted_dlcm_identity, "06D.accepted_identity")
    _require_equal(qualification_decision.get("accepted_identity"), accepted_dlcm_identity, "06E.accepted_identity")
    _require_equal(lse_evidence_upstream.get("accepted_dlcm_identity"), accepted_dlcm_identity, "06F.accepted_dlcm_identity")

    _require_equal(gate_ids.get("v5_deployment_identity"), v5_deployment_identity, "06A.v5_deployment_identity")
    _require_equal(packaging_ids.get("v5_deployment_identity"), v5_deployment_identity, "06B.v5_deployment_identity")
    _require_equal(prerequisite_gate.get("v5_deployment_identity"), v5_deployment_identity, "06C.v5_deployment_identity")
    _require_equal(training_gate.get("v5_deployment_identity"), v5_deployment_identity, "06D.v5_deployment_identity")
    _require_equal(qualification_decision.get("v5_deployment_identity"), v5_deployment_identity, "06E.v5_deployment_identity")
    _require_equal(lse_evidence_upstream.get("v5_deployment_identity"), v5_deployment_identity, "06F.v5_deployment_identity")

    _require_equal(qualification_decision.get("unlock_identity"), unlock_identity, "unlock_identity")
    _require_equal(training_unlock.get("unlock_identity"), unlock_identity, "06D.unlock_identity")
    _require_equal(lse_evidence_upstream.get("unlock_identity"), unlock_identity, "06F.unlock_identity")
    _require_equal(qualification_decision.get("training_receipt_identity"), training_receipt_identity, "06E.training_receipt_identity")
    _require_equal(training_receipt.get("receipt_identity"), training_receipt_identity, "06D.training_receipt_identity")
    _require_equal(lse_evidence_upstream.get("training_receipt_identity"), training_receipt_identity, "06F.training_receipt_identity")
    _require_equal(qualification_decision.get("H_lse_qualification"), qualification_identity, "H_lse_qualification")
    _require_equal(lse_evidence_upstream.get("H_lse_qualification"), qualification_identity, "06F.H_lse_qualification")
    _require_equal(qualification_decision.get("best_checkpoint_sha256"), lse_checkpoint_sha, "06E.best_checkpoint_sha256")
    _require_equal(training_checkpoint.get("sha256"), lse_checkpoint_sha, "06D.best_checkpoint_sha256")
    _require_equal(accepted_lse_receipt.get("accepted_lse_checkpoint_sha256"), lse_checkpoint_sha, "06F.receipt.checkpoint")
    _require_equal(accepted_lse_evidence.get("accepted_lse_checkpoint_sha256"), lse_checkpoint_sha, "06F.evidence.checkpoint")
    _require_equal(training_run.get("selector_signal_layout_hash"), selector_hash, "selector_signal_layout_hash")
    _require_equal(accepted_lse_evidence.get("selector_signal_layout_hash"), selector_hash, "06F.selector_signal_layout_hash")
    _require_equal(accepted_lse_receipt.get("accepted_lse_identity"), accepted_lse_identity, "06F.receipt.accepted_lse_identity")
    _require_equal(accepted_lse_evidence.get("accepted_lse_identity"), accepted_lse_identity, "06F.evidence.accepted_lse_identity")
    _require_equal(
        accepted_lse_receipt.get("receipt_identity"),
        accepted_lse_evidence.get("closure_receipt_identity"),
        "06F.closure_receipt_identity",
    )

    manifest = {
        "schema_version": "b2_07_phase_final_closure_manifest_v1",
        "status": "b2_phase_completed_locally",
        "git_sha": str(git_sha),
        "accepted_dlcm_identity": accepted_dlcm_identity,
        "v5_deployment_identity": v5_deployment_identity,
        "accepted_lse_identity": accepted_lse_identity,
        "H_lse_qualification": qualification_identity,
        "training_receipt_identity": training_receipt_identity,
        "unlock_identity": unlock_identity,
        "accepted_v5_checkpoint_sha256": reference_packaging_evidence.get("checkpoint_sha256"),
        "accepted_lse_checkpoint_sha256": lse_checkpoint_sha,
        "selector_signal_layout_hash": selector_hash,
        "calibration_nll": qualification_decision.get("calibration_nll"),
        "max_calibration_nll": qualification_decision.get("max_calibration_nll"),
        "evaluated_rows": qualification_decision.get("evaluated_rows"),
        "tracked_pt_count": int(tracked_pt_count),
        "training_started_in_b2_07": False,
        "evaluation_started_in_b2_07": False,
        "final_content_accessed_in_b2_07": False,
        "pushed": bool(pushed),
        "pr_opened": bool(pr_opened),
        "phase_chain": {
            "b2_06a": "accepted_gate_wired",
            "b2_06b": "accepted_v5_refs_packaged",
            "b2_06c": "lse_prerequisites_materialized",
            "b2_06d": "lse_training_completed",
            "b2_06e": "lse_qualified",
            "b2_06f": "accepted_lse_artifact_frozen",
        },
    }
    manifest["phase_final_closure_identity"] = canonical_json_sha256(manifest)
    return manifest
