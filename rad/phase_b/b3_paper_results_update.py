"""B3-07 paper results update closure built from frozen evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

ACCEPTED_DLCM_IDENTITY = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
V5_DEPLOYMENT_IDENTITY = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
ACCEPTED_LSE_IDENTITY = "3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa"
B2_PHASE_FINAL_CLOSURE_IDENTITY = "2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d"
B3_PHASE_CLOSURE_IDENTITY = "a984814c1821dbc6c0b2ee49fbf018be0c8b4f2fe226855f6b3e015eb89e05be"
B4_WEIGHT_EVIDENCE_IDENTITY = "68bcea45e1fe98ffbee9f9ea51a2b645916b4a623198f787ce8830b1b0f8fe79"
B4_FINAL_RELEASE_IDENTITY = "296191577c12aa42e2e4dbad3d34deaef67b04bbd34d3d0f52be20b9e1c99b93"


class B3PaperResultsUpdateError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B3PaperResultsUpdateError(code, detail)


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
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _fail("B3_PAPER_RESULTS_UPDATE_JSON_INVALID", f"{path} must contain a JSON object")
    return payload


def _require_schema(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("schema_version") != expected:
        _fail("B3_PAPER_RESULTS_UPDATE_SCHEMA_MISMATCH", f"{label} schema mismatch")


def _nested(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        _fail("B3_PAPER_RESULTS_UPDATE_FIELD_INVALID", f"{key} must be a mapping")
    return value


def _require_equal(actual: Any, expected: Any, detail: str) -> None:
    if actual != expected:
        _fail("B3_PAPER_RESULTS_UPDATE_IDENTITY_MISMATCH", detail)


def _require_false(value: Any, detail: str) -> None:
    if value is not False:
        _fail("B3_PAPER_RESULTS_UPDATE_BOUNDARY_VIOLATION", detail)


def _require_true(value: Any, code: str, detail: str) -> None:
    if value is not True:
        _fail(code, detail)


def _check_boundary(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    boundary = _nested(payload, "boundary")
    for key in keys:
        _require_false(boundary.get(key), f"{label}.{key} must be false")
    if int(boundary.get("tracked_pt_files", -1)) != 0:
        _fail("B3_PAPER_RESULTS_UPDATE_TRACKED_PT", f"{label}.tracked_pt_files must be zero")


def _validate_b3_negative_result(b3_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    claims = _nested(b3_manifest, "claims")
    if claims.get("early_exit_accepted_mechanism") is not False:
        _fail(
            "B3_PAPER_RESULTS_UPDATE_EARLY_EXIT_CLAIM_INVALID",
            "B3-06 must not accept early-exit",
        )
    _require_true(
        claims.get("full_depth_fallback_retained"),
        "B3_PAPER_RESULTS_UPDATE_EARLY_EXIT_CLAIM_INVALID",
        "B3-06 must retain full-depth fallback",
    )
    if claims.get("dynamic_fusion_abandoned") is not False or claims.get("lse_abandoned") is not False:
        _fail(
            "B3_PAPER_RESULTS_UPDATE_EARLY_EXIT_CLAIM_INVALID",
            "B3-06 must retain DLCM and LSE",
        )
    table = b3_manifest.get("negative_result_table")
    if not isinstance(table, list) or len(table) != 1 or not isinstance(table[0], Mapping):
        _fail(
            "B3_PAPER_RESULTS_UPDATE_EARLY_EXIT_RESULT_INVALID",
            "B3-06 must contain exactly one negative-result row",
        )
    row = table[0]
    if (
        row.get("candidate_depths") != [12, 18]
        or row.get("fallback_depth") != 24
        or row.get("positive_signal_count") != 0
        or row.get("positive_exit_targets") != 0
        or row.get("accepted_as_final_mechanism") is not False
    ):
        _fail(
            "B3_PAPER_RESULTS_UPDATE_EARLY_EXIT_RESULT_INVALID",
            "B3-06 negative-result row must preserve the conservative zero-signal outcome",
        )
    return row


def build_b3_paper_results_update_manifest(
    *,
    b2_manifest: Mapping[str, Any],
    b3_manifest: Mapping[str, Any],
    b4_weight_manifest: Mapping[str, Any],
    b4_release_manifest: Mapping[str, Any],
    tracked_pt_count: int,
) -> dict[str, Any]:
    if tracked_pt_count != 0:
        _fail("B3_PAPER_RESULTS_UPDATE_TRACKED_PT", "tracked .pt files must remain zero")
    _require_schema(b2_manifest, "b2_08_paper_results_evidence_index_manifest_v1", "B2-08")
    _require_schema(b3_manifest, "b3_06_early_exit_phase_closure_manifest_v1", "B3-06")
    _require_schema(b4_weight_manifest, "b4_01_dlcm_adaptive_weight_evidence_manifest_v1", "B4-01")
    _require_schema(b4_release_manifest, "b4_02_final_local_paper_release_manifest_v1", "B4-02")
    _check_boundary(
        b2_manifest,
        (
            "training_started_in_b2_08",
            "evaluation_started_in_b2_08",
            "final_content_accessed_in_b2_08",
            "model_artifact_generated_in_b2_08",
            "pushed",
            "pr_opened",
        ),
        "B2-08",
    )
    _check_boundary(
        b3_manifest,
        (
            "training_started_in_b3_06",
            "evaluation_started_in_b3_06",
            "final_content_accessed_in_b3_06",
            "model_artifact_generated_in_b3_06",
            "pushed",
            "pr_opened",
        ),
        "B3-06",
    )
    _check_boundary(
        b4_weight_manifest,
        (
            "training_started",
            "evaluation_started",
            "final_content_accessed",
            "model_artifact_generated",
            "pushed",
            "pr_opened",
        ),
        "B4-01",
    )
    _check_boundary(
        b4_release_manifest,
        (
            "training_started_in_release",
            "evaluation_started_in_release",
            "final_content_accessed_in_release",
            "model_artifact_generated_in_release",
            "pushed",
            "pr_opened",
        ),
        "B4-02",
    )

    b2_ids = _nested(b2_manifest, "primary_identities")
    b3_ids = _nested(b3_manifest, "primary_identities")
    b4_release_ids = _nested(b4_release_manifest, "bound_identities")
    for identities, label in ((b2_ids, "B2-08"), (b3_ids, "B3-06"), (b4_release_ids, "B4-02")):
        _require_equal(identities.get("accepted_dlcm_identity"), ACCEPTED_DLCM_IDENTITY, f"{label} accepted DLCM identity mismatch")
        _require_equal(identities.get("v5_deployment_identity"), V5_DEPLOYMENT_IDENTITY, f"{label} V5 deployment identity mismatch")
        _require_equal(identities.get("accepted_lse_identity"), ACCEPTED_LSE_IDENTITY, f"{label} accepted LSE identity mismatch")
    _require_equal(
        b2_manifest.get("source_phase_final_closure_identity"),
        B2_PHASE_FINAL_CLOSURE_IDENTITY,
        "B2-08 B2 closure identity mismatch",
    )
    _require_equal(
        b3_ids.get("b2_phase_final_closure_identity"),
        B2_PHASE_FINAL_CLOSURE_IDENTITY,
        "B3-06 B2 closure identity mismatch",
    )
    _require_equal(
        b4_release_ids.get("b2_phase_final_closure_identity"),
        B2_PHASE_FINAL_CLOSURE_IDENTITY,
        "B4-02 B2 closure identity mismatch",
    )
    _require_equal(b3_manifest.get("phase_closure_identity"), B3_PHASE_CLOSURE_IDENTITY, "B3-06 closure identity mismatch")
    _require_equal(b4_weight_manifest.get("accepted_dlcm_identity"), ACCEPTED_DLCM_IDENTITY, "B4-01 accepted DLCM identity mismatch")
    _require_equal(b4_weight_manifest.get("v5_deployment_identity"), V5_DEPLOYMENT_IDENTITY, "B4-01 V5 deployment identity mismatch")
    _require_equal(b4_weight_manifest.get("weight_evidence_identity"), B4_WEIGHT_EVIDENCE_IDENTITY, "B4-01 evidence identity mismatch")
    _require_equal(b4_release_ids.get("b3_06_phase_closure_identity"), B3_PHASE_CLOSURE_IDENTITY, "B4-02 B3 closure identity mismatch")
    _require_equal(b4_release_ids.get("b4_01_weight_evidence_identity"), B4_WEIGHT_EVIDENCE_IDENTITY, "B4-02 B4 weight identity mismatch")
    _require_equal(b4_release_manifest.get("final_release_identity"), B4_FINAL_RELEASE_IDENTITY, "B4-02 final release identity mismatch")

    dlcm = _nested(b2_manifest, "dlcm")
    _require_equal(dlcm.get("beta_star_decimal"), "0.54", "B2-08 beta* mismatch")
    _require_equal(b4_weight_manifest.get("beta_star_decimal"), "0.54", "B4-01 beta* mismatch")
    lse = _nested(b2_manifest, "lse_qualification")
    if lse.get("verdict") != "qualified":
        _fail("B3_PAPER_RESULTS_UPDATE_LSE_INVALID", "B2-08 LSE must be qualified")
    _require_true(
        b4_weight_manifest.get("sample_adaptive_variation_observed"),
        "B3_PAPER_RESULTS_UPDATE_WEIGHT_EVIDENCE_INVALID",
        "B4-01 adaptive weight evidence is required",
    )
    if b4_weight_manifest.get("uniform_equivalent_at_tolerance") is not False:
        _fail(
            "B3_PAPER_RESULTS_UPDATE_WEIGHT_EVIDENCE_INVALID",
            "B4-01 weights must not be uniform equivalent",
        )
    negative_result = _validate_b3_negative_result(b3_manifest)
    release_claims = _nested(b4_release_manifest, "primary_claims")
    _require_true(
        release_claims.get("dlcm_sample_adaptive_fusion_supported"),
        "B3_PAPER_RESULTS_UPDATE_RELEASE_CLAIM_INVALID",
        "B4-02 must support DLCM sample-adaptive fusion",
    )
    _require_true(
        release_claims.get("dlcm_final_accepted"),
        "B3_PAPER_RESULTS_UPDATE_RELEASE_CLAIM_INVALID",
        "B4-02 must retain accepted DLCM",
    )
    _require_true(
        release_claims.get("lse_qualified"),
        "B3_PAPER_RESULTS_UPDATE_RELEASE_CLAIM_INVALID",
        "B4-02 must retain qualified LSE",
    )
    if release_claims.get("early_exit_accepted_mechanism") is not False:
        _fail(
            "B3_PAPER_RESULTS_UPDATE_RELEASE_CLAIM_INVALID",
            "B4-02 must not accept early-exit",
        )
    _require_true(
        release_claims.get("early_exit_negative_result"),
        "B3_PAPER_RESULTS_UPDATE_RELEASE_CLAIM_INVALID",
        "B4-02 must retain the early-exit negative result",
    )

    weight_summary = _nested(b4_weight_manifest, "deployment_weight_summary")
    source_documents = sorted(
        set(
            list(b2_manifest.get("evidence_documents", []))
            + list(b3_manifest.get("evidence_documents", []))
            + [
                "docs/phase_b/b4_01_dlcm_adaptive_weight_evidence_manifest.json",
                "docs/phase_b/b4_01_dlcm_adaptive_weight_evidence.md",
                "docs/phase_b/b4_02_final_local_paper_release_manifest.json",
                "docs/phase_b/b4_02_final_paper_results_summary.md",
            ]
        )
    )
    payload: dict[str, Any] = {
        "schema_version": "b3_07_paper_results_update_manifest_v1",
        "phase": "B3-07 Paper Results Update",
        "status": "paper_results_update_frozen_locally",
        "update_decision": "paper_results_update_ready",
        "bound_identities": {
            "accepted_dlcm_identity": ACCEPTED_DLCM_IDENTITY,
            "v5_deployment_identity": V5_DEPLOYMENT_IDENTITY,
            "accepted_lse_identity": ACCEPTED_LSE_IDENTITY,
            "b2_phase_final_closure_identity": B2_PHASE_FINAL_CLOSURE_IDENTITY,
            "b3_06_phase_closure_identity": B3_PHASE_CLOSURE_IDENTITY,
            "b4_01_weight_evidence_identity": B4_WEIGHT_EVIDENCE_IDENTITY,
            "b4_02_final_release_identity": B4_FINAL_RELEASE_IDENTITY,
        },
        "paper_claims": {
            "dlcm_sample_adaptive_fusion_supported": True,
            "dlcm_final_accepted": True,
            "lse_qualified": True,
            "early_exit_negative_result": True,
            "early_exit_accepted_mechanism": False,
            "full_depth_fallback_retained": True,
        },
        "result_summary": {
            "beta_star_decimal": "0.54",
            "lse_calibration_nll": lse.get("calibration_nll"),
            "lse_required_depths": lse.get("required_depths"),
            "adaptive_weight_calibration_records": b4_weight_manifest.get("calibration_records"),
            "adaptive_weight_max_sample_linf_delta_from_uniform": weight_summary.get("max_sample_linf_delta_from_uniform"),
            "early_exit_candidate_depths": negative_result.get("candidate_depths"),
            "early_exit_fallback_depth": negative_result.get("fallback_depth"),
            "early_exit_positive_signal_count": negative_result.get("positive_signal_count"),
        },
        "source_documents": source_documents,
        "boundary": {
            "training_started": False,
            "evaluation_started": False,
            "final_content_accessed": False,
            "model_artifact_generated": False,
            "tracked_pt_files": int(tracked_pt_count),
            "pushed": False,
            "pr_opened": False,
        },
    }
    payload["update_identity"] = canonical_json_sha256(payload)
    return payload
