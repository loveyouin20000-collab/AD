"""Final decision, evidence, and accepted manifest builders for V5 C4C."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_v5_protocol as protocol


class B2DLCMV5FinalManifestError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5FinalManifestError(code, detail)


def build_final_decision_manifest(
    *,
    gt_target_learning: Mapping[str, Any],
    localization: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    verdict: str,
) -> dict[str, Any]:
    forbidden = ("development", "teacher", "path", "timestamp", "gpu_uuid")
    for group in (gt_target_learning, localization, thresholds):
        for key in group:
            if any(token in key.lower() for token in forbidden):
                _fail("B2_DLCM_FINAL_DECISION_INVALID", f"forbidden decision field {key}")
    payload = {
        "schema_version": "b2_dlcm_v5_final_decision_manifest_v1",
        "gt_target_learning": dict(gt_target_learning),
        "localization": dict(localization),
        "thresholds": dict(thresholds),
        "verdict": verdict,
    }
    payload["H_decision"] = protocol.canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "H_decision"}
    )
    return payload


def build_final_evidence_manifest(
    *,
    final_decision_manifest: Mapping[str, Any],
    materialization_ab: Mapping[str, Any],
    evaluation_ab: Mapping[str, Any],
    expected: Mapping[str, Any],
    production_metric_proof: Mapping[str, Any],
) -> dict[str, Any]:
    h_decision = str(final_decision_manifest["H_decision"])
    payload = {
        "schema_version": "b2_dlcm_v5_final_evidence_manifest_v1",
        "H_decision": h_decision,
        "calibration_ab_identity": str(expected["calibration_ab_identity"]),
        "development_qualified_identity": str(expected["development_qualified_identity"]),
        "final_roster_identity": str(expected["final_roster_identity"]),
        "tooling_contract_schema": str(expected["tooling_contract_schema"]),
        "tooling_baseline_commit": str(expected["tooling_baseline_commit"]),
        "tooling_baseline_tag": str(expected["tooling_baseline_tag"]),
        "materialization_ab": dict(materialization_ab),
        "evaluation_ab": dict(evaluation_ab),
        "production_metric_proof": dict(production_metric_proof),
    }
    payload["H_evidence"] = protocol.canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "H_evidence"}
    )
    return payload


def build_accepted_deployment_manifest(
    *,
    final_decision_manifest: Mapping[str, Any],
    final_evidence_manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if final_decision_manifest.get("verdict") != "qualified":
        _fail("B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN", "final verdict must be qualified")
    payload = {
        "schema_version": "b2_dlcm_v5_accepted_deployment_manifest_v1",
        "deployment_qualified": True,
        "v5_deployment_identity": str(expected["v5_deployment_identity"]),
        "beta_star_decimal": str(expected["beta_star_decimal"]),
        "calibration_ab_identity": str(expected["calibration_ab_identity"]),
        "H_decision": str(final_decision_manifest["H_decision"]),
        "H_evidence": str(final_evidence_manifest["H_evidence"]),
    }
    payload["accepted_identity"] = protocol.canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "accepted_identity"}
    )
    return payload
