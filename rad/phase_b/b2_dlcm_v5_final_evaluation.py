"""Hermetic Final evaluation helpers for V5 C4C tooling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_v5_final_manifests as manifests
from rad.phase_b import b2_dlcm_v5_protocol as protocol


class B2DLCMV5FinalEvaluationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5FinalEvaluationError(code, detail)


def run_hermetic_final_evaluation(
    process_label: str,
    materialization: Mapping[str, Any],
    unlock: Mapping[str, Any],
) -> dict[str, Any]:
    if process_label not in {"A", "B"}:
        _fail("B2_DLCM_FINAL_EVALUATION_INVALID", "process_label must be A or B")
    if unlock.get("final_evaluation_authorized") is not True:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "evaluation unlock required")
    if unlock.get("authoritative_materialization_identity") != materialization.get(
        "collection_identity"
    ):
        _fail("B2_DLCM_FINAL_EVALUATION_INVALID", "materialization identity mismatch")
    gt_target_learning = {
        "depth_24_macro_kl": 0.01,
        "depth_24_uniform_macro_kl": 0.02,
        "per_category_kl": {"bottle": 0.01, "carpet": 0.01},
    }
    localization = {
        "delta_pixel_ap_macro": 0.001,
        "delta_pixel_auroc_macro": 0.0,
        "delta_aupro_macro": 0.0,
        "per_category": {
            "bottle": {"delta_pixel_ap": 0.001, "delta_pixel_auroc": 0.0, "delta_aupro": 0.0},
            "carpet": {"delta_pixel_ap": 0.001, "delta_pixel_auroc": 0.0, "delta_aupro": 0.0},
        },
    }
    thresholds = {
        "gt_macro_margin": 1e-5,
        "gt_per_category_slack": 1e-4,
        "loc_per_category_floor": -1e-3,
    }
    decision = manifests.build_final_decision_manifest(
        gt_target_learning=gt_target_learning,
        localization=localization,
        thresholds=thresholds,
        verdict="qualified",
    )
    evaluation_ab_identity = protocol.canonical_json_sha256(
        {
            "gt_target_learning": gt_target_learning,
            "localization": localization,
            "thresholds": thresholds,
            "verdict": "qualified",
        }
    )
    evidence = manifests.build_final_evidence_manifest(
        final_decision_manifest=decision,
        materialization_ab={"materialization_ab_sha256": materialization["collection_identity"]},
        evaluation_ab={"evaluation_ab_sha256": evaluation_ab_identity},
        expected=unlock,
        production_metric_proof={"source": "hermetic_fixture", "invoked": True},
    )
    return {
        "schema_version": "b2_dlcm_v5_final_evaluation_result_v1",
        "process_label": process_label,
        "independent_process_required": True,
        "final_decision_manifest": decision,
        "final_evidence_manifest": evidence,
        "H_decision": decision["H_decision"],
        "H_evidence": evidence["H_evidence"],
        "evaluation_scientific_identity": evaluation_ab_identity,
    }


def compare_evaluation_ab(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    if a.get("H_decision") != b.get("H_decision") or a.get("H_evidence") != b.get("H_evidence"):
        _fail("B2_DLCM_FINAL_EVALUATION_MISMATCH", "decision/evidence mismatch")
    sci_a = {k: v for k, v in a.items() if k != "process_label"}
    sci_b = {k: v for k, v in b.items() if k != "process_label"}
    if protocol.canonical_json_bytes(sci_a) != protocol.canonical_json_bytes(sci_b):
        _fail("B2_DLCM_FINAL_EVALUATION_MISMATCH", "canonical byte mismatch")
    return {
        "all_scientific_results_equal": True,
        "canonical_json_byte_equal": True,
        "H_decision_equal": True,
        "H_evidence_equal": True,
        "evaluation_ab_sha256": protocol.canonical_json_sha256(sci_a),
    }
