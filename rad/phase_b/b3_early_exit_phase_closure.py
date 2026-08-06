"""B3-06 early-exit phase closure and paper-ready negative result integration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn


class B3EarlyExitPhaseClosureError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B3EarlyExitPhaseClosureError(code, detail)


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
        _fail("B3_EARLY_EXIT_PHASE_CLOSURE_JSON_INVALID", f"{path} must contain JSON object")
    return payload


def _require_schema(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("schema_version") != expected:
        _fail("B3_EARLY_EXIT_PHASE_CLOSURE_SCHEMA_MISMATCH", f"{label} schema mismatch")


def _require_equal(left: Any, right: Any, detail: str) -> None:
    if left != right:
        _fail("B3_EARLY_EXIT_PHASE_CLOSURE_IDENTITY_MISMATCH", detail)


def _nested(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        _fail("B3_EARLY_EXIT_PHASE_CLOSURE_FIELD_INVALID", f"{key} must be a mapping")
    return value


def _require_false(value: Any, detail: str) -> None:
    if value is not False:
        _fail("B3_EARLY_EXIT_PHASE_CLOSURE_BOUNDARY_VIOLATION", detail)


def _sum_counts(counts: Any, detail: str) -> int:
    if not isinstance(counts, Mapping):
        _fail("B3_EARLY_EXIT_PHASE_CLOSURE_FIELD_INVALID", detail)
    return sum(int(v) for v in counts.values())


def _check_flat_boundary(payload: Mapping[str, Any], label: str) -> None:
    for key in (
        "training_unlocked",
        "training_started",
        "evaluation_started",
        "final_content_accessed",
        "checkpoint_generated",
    ):
        if key in payload:
            _require_false(payload.get(key), f"{label}.{key} must be false")


def _check_nested_boundary(payload: Mapping[str, Any], key: str, label: str) -> None:
    if key not in payload:
        return
    boundary = _nested(payload, key)
    for boundary_key, value in boundary.items():
        if boundary_key in {"tracked_pt_files"}:
            if int(value) != 0:
                _fail("B3_EARLY_EXIT_PHASE_CLOSURE_TRACKED_PT", f"{label}.{boundary_key}")
        elif boundary_key in {
            "pushed",
            "pr_opened",
            "training_started",
            "evaluation_started",
            "final_content_accessed",
            "artifact_written",
            "checkpoint_generated",
            "training_started_in_b3_01",
            "evaluation_started_in_b3_01",
            "final_content_accessed_in_b3_01",
            "checkpoint_generated_in_b3_01",
            "artifact_written_in_b3_01",
            "exit_policy_training_started",
            "exit_policy_evaluation_started",
        }:
            _require_false(value, f"{label}.{boundary_key} must be false")


def build_early_exit_phase_closure(
    *,
    b3_01_evidence: Mapping[str, Any],
    b3_02_evidence: Mapping[str, Any],
    b3_03_evidence: Mapping[str, Any],
    b3_04_evidence: Mapping[str, Any],
    b3_05_evidence: Mapping[str, Any],
    tracked_pt_count: int,
) -> dict[str, Any]:
    if tracked_pt_count != 0:
        _fail("B3_EARLY_EXIT_PHASE_CLOSURE_TRACKED_PT", "tracked .pt files must remain zero")
    _require_schema(
        b3_01_evidence,
        "b3_01_early_exit_contract_preflight_evidence_v1",
        "B3-01 evidence",
    )
    _require_schema(
        b3_02_evidence,
        "b3_02_exit_prerequisite_materialization_evidence_v1",
        "B3-02 evidence",
    )
    _require_schema(
        b3_03_evidence,
        "b3_03_exit_policy_training_contract_evidence_v1",
        "B3-03 evidence",
    )
    _require_schema(
        b3_04_evidence,
        "b3_04_exit_target_positive_signal_contract_evidence_v1",
        "B3-04 evidence",
    )
    _require_schema(
        b3_05_evidence,
        "b3_05_early_exit_negative_result_evidence_v1",
        "B3-05 evidence",
    )

    b3_01_chain = _nested(b3_01_evidence, "accepted_chain")
    b3_02_chain = _nested(b3_02_evidence, "accepted_chain")
    for key in (
        "accepted_dlcm_identity",
        "accepted_lse_identity",
        "accepted_lse_checkpoint_sha256",
        "b2_phase_final_closure_identity",
    ):
        _require_equal(b3_01_chain.get(key), b3_02_chain.get(key), f"accepted chain mismatch: {key}")
    materialization = _nested(b3_02_evidence, "materialization")
    _check_flat_boundary(b3_03_evidence, "B3-03")
    _check_flat_boundary(b3_04_evidence, "B3-04")
    _check_flat_boundary(b3_05_evidence, "B3-05")
    _check_nested_boundary(b3_01_evidence, "boundary", "B3-01")
    _check_nested_boundary(b3_02_evidence, "boundary", "B3-02")

    positive_exit_targets = int(b3_05_evidence.get("positive_exit_targets", -1))
    positive_signal_count = int(b3_05_evidence.get("positive_signal_count", -1))
    target_exit_count = _sum_counts(b3_05_evidence.get("target_exits_by_depth"), "B3-05 target counts")
    depth_positive_count = _sum_counts(
        b3_05_evidence.get("candidate_positive_counts_by_depth"),
        "B3-05 positive counts",
    )
    if (
        b3_05_evidence.get("status") != "early_exit_line_closed_negative_result"
        or b3_05_evidence.get("decision") != "conservative_full_depth_fallback"
        or positive_exit_targets != 0
        or positive_signal_count != 0
        or target_exit_count != 0
        or depth_positive_count != 0
    ):
        _fail(
            "B3_EARLY_EXIT_PHASE_CLOSURE_NEGATIVE_RESULT_INVALID",
            "B3-05 must be a zero-positive negative result",
        )
    if b3_03_evidence.get("status") != "conservative_full_depth_fallback":
        _fail("B3_EARLY_EXIT_PHASE_CLOSURE_DECISION_INVALID", "B3-03 must be fallback")
    if b3_04_evidence.get("decision") != "no_positive_signal_under_conservative_contract":
        _fail("B3_EARLY_EXIT_PHASE_CLOSURE_DECISION_INVALID", "B3-04 must be no-positive")

    negative_row = {
        "component": "early_exit_policy",
        "candidate_depths": [12, 18],
        "fallback_depth": 24,
        "records": int(materialization.get("records", 0)),
        "target_exits_by_depth": dict(b3_05_evidence.get("target_exits_by_depth", {})),
        "candidate_positive_counts_by_depth": dict(
            b3_05_evidence.get("candidate_positive_counts_by_depth", {})
        ),
        "positive_exit_targets": positive_exit_targets,
        "positive_signal_count": positive_signal_count,
        "accepted_as_final_mechanism": False,
        "paper_interpretation": "negative_result_under_conservative_gate",
    }
    payload = {
        "schema_version": "b3_06_early_exit_phase_closure_manifest_v1",
        "phase": "B3-06 Early-Exit Phase Closure / Paper-Ready Negative Result Integration",
        "status": "early_exit_phase_closed_negative_result",
        "paper_position": "negative_result_and_future_work",
        "accepted_system_behavior": "full_depth_fallback",
        "primary_result": (
            "Early-exit was explored under accepted DLCM and LSE identities, but no legal "
            "positive exit signal was found under the conservative contract."
        ),
        "primary_identities": {
            "accepted_dlcm_identity": b3_01_chain.get("accepted_dlcm_identity"),
            "v5_deployment_identity": b3_01_chain.get("v5_deployment_identity"),
            "accepted_lse_identity": b3_01_chain.get("accepted_lse_identity"),
            "accepted_lse_checkpoint_sha256": b3_01_chain.get("accepted_lse_checkpoint_sha256"),
            "b2_phase_final_closure_identity": b3_01_chain.get("b2_phase_final_closure_identity"),
        },
        "phase_identities": {
            "b3_02_materialization_identity": materialization.get("materialization_identity"),
            "b3_03_training_contract_identity": b3_03_evidence.get("training_contract_identity"),
            "b3_04_positive_signal_contract_identity": b3_04_evidence.get(
                "positive_signal_contract_identity"
            ),
            "b3_05_line_closure_identity": b3_05_evidence.get("line_closure_identity"),
        },
        "negative_result_table": [negative_row],
        "claims": {
            "dynamic_fusion_abandoned": False,
            "lse_abandoned": False,
            "early_exit_accepted_mechanism": False,
            "early_exit_training_unlocked": False,
            "early_exit_checkpoint_generated": False,
            "full_depth_fallback_retained": True,
        },
        "evidence_documents": [
            "docs/phase_b/b3_01_early_exit_contract_architecture.md",
            "docs/phase_b/b3_01_early_exit_preflight_evidence.json",
            "docs/phase_b/b3_02_exit_prerequisite_materialization_manifest.json",
            "docs/phase_b/b3_02_exit_prerequisite_materialization_evidence.json",
            "docs/phase_b/b3_03_exit_policy_training_contract.json",
            "docs/phase_b/b3_03_exit_policy_training_contract_evidence.json",
            "docs/phase_b/b3_04_exit_target_positive_signal_contract.json",
            "docs/phase_b/b3_04_exit_target_positive_signal_contract_evidence.json",
            "docs/phase_b/b3_05_early_exit_line_closure_manifest.json",
            "docs/phase_b/b3_05_early_exit_negative_result_evidence.json",
            "docs/phase_b/b3_06_early_exit_phase_closure_manifest.json",
            "docs/phase_b/b3_06_early_exit_paper_results_summary.md",
            "docs/phase_b/b3_06_early_exit_evidence_index.md",
        ],
        "boundary": {
            "training_started_in_b3_06": False,
            "evaluation_started_in_b3_06": False,
            "final_content_accessed_in_b3_06": False,
            "model_artifact_generated_in_b3_06": False,
            "tracked_pt_files": int(tracked_pt_count),
            "pushed": False,
            "pr_opened": False,
        },
    }
    payload["phase_closure_identity"] = canonical_json_sha256(payload)
    return payload
