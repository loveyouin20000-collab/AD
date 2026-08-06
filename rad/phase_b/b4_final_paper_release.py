"""B4-02 final local paper release closure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn


class B4FinalPaperReleaseError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B4FinalPaperReleaseError(code, detail)


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
        _fail("B4_FINAL_RELEASE_JSON_INVALID", f"{path} must contain JSON object")
    return payload


def _require_schema(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("schema_version") != expected:
        _fail("B4_FINAL_RELEASE_SCHEMA_MISMATCH", f"{label} schema mismatch")


def _nested(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        _fail("B4_FINAL_RELEASE_FIELD_INVALID", f"{key} must be a mapping")
    return value


def _require_false(value: Any, detail: str) -> None:
    if value is not False:
        _fail("B4_FINAL_RELEASE_BOUNDARY_VIOLATION", detail)


def _require_equal(left: Any, right: Any, detail: str) -> None:
    if left != right:
        _fail("B4_FINAL_RELEASE_IDENTITY_MISMATCH", detail)


def _check_release_boundary(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    boundary = _nested(payload, "boundary")
    for key in keys:
        _require_false(boundary.get(key), f"{label}.{key} must be false")
    tracked = boundary.get("tracked_pt_files")
    if tracked is not None and int(tracked) != 0:
        _fail("B4_FINAL_RELEASE_TRACKED_PT", f"{label}.tracked_pt_files must be zero")


def build_final_paper_release_manifest(
    *,
    b2_manifest: Mapping[str, Any],
    b3_manifest: Mapping[str, Any],
    b4_weight_manifest: Mapping[str, Any],
    tracked_pt_count: int,
) -> dict[str, Any]:
    if tracked_pt_count != 0:
        _fail("B4_FINAL_RELEASE_TRACKED_PT", "tracked .pt files must remain zero")
    _require_schema(b2_manifest, "b2_08_paper_results_evidence_index_manifest_v1", "B2-08")
    _require_schema(b3_manifest, "b3_06_early_exit_phase_closure_manifest_v1", "B3-06")
    _require_schema(
        b4_weight_manifest,
        "b4_01_dlcm_adaptive_weight_evidence_manifest_v1",
        "B4-01 weight evidence",
    )
    _check_release_boundary(
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
    _check_release_boundary(
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
    _check_release_boundary(
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
    b2_ids = _nested(b2_manifest, "primary_identities")
    b3_ids = _nested(b3_manifest, "primary_identities")
    _require_equal(
        b2_ids.get("accepted_dlcm_identity"),
        b3_ids.get("accepted_dlcm_identity"),
        "B2/B3 accepted DLCM identity mismatch",
    )
    _require_equal(
        b2_ids.get("accepted_dlcm_identity"),
        b4_weight_manifest.get("accepted_dlcm_identity"),
        "B2/B4 accepted DLCM identity mismatch",
    )
    _require_equal(
        b2_ids.get("v5_deployment_identity"),
        b3_ids.get("v5_deployment_identity"),
        "B2/B3 V5 deployment identity mismatch",
    )
    _require_equal(
        b2_ids.get("v5_deployment_identity"),
        b4_weight_manifest.get("v5_deployment_identity"),
        "B2/B4 V5 deployment identity mismatch",
    )
    _require_equal(
        b2_ids.get("accepted_lse_identity"),
        b3_ids.get("accepted_lse_identity"),
        "B2/B3 accepted LSE identity mismatch",
    )
    b3_claims = _nested(b3_manifest, "claims")
    if b3_claims.get("early_exit_accepted_mechanism") is not False:
        _fail("B4_FINAL_RELEASE_EARLY_EXIT_CLAIM_INVALID", "early-exit must remain negative")
    if b3_claims.get("dynamic_fusion_abandoned") is not False or b3_claims.get("lse_abandoned") is not False:
        _fail("B4_FINAL_RELEASE_PRIMARY_CLAIM_INVALID", "DLCM/LSE must not be abandoned")
    if b4_weight_manifest.get("sample_adaptive_variation_observed") is not True:
        _fail("B4_FINAL_RELEASE_WEIGHT_EVIDENCE_INVALID", "adaptive weight evidence missing")
    if b4_weight_manifest.get("uniform_equivalent_at_tolerance") is not False:
        _fail("B4_FINAL_RELEASE_WEIGHT_EVIDENCE_INVALID", "weights must not be uniform equivalent")
    lse = _nested(b2_manifest, "lse_qualification")
    if lse.get("verdict") != "qualified":
        _fail("B4_FINAL_RELEASE_LSE_INVALID", "LSE must be qualified")

    b2_docs = list(b2_manifest.get("evidence_documents", []))
    b3_docs = list(b3_manifest.get("evidence_documents", []))
    b4_docs = [
        "docs/phase_b/b4_01_dlcm_adaptive_weight_evidence_manifest.json",
        "docs/phase_b/b4_01_dlcm_adaptive_weight_evidence.md",
        "docs/phase_b/b4_01_dlcm_adaptive_weight_trace.csv",
    ]
    release_docs = [
        "docs/phase_b/b4_02_final_local_paper_release_manifest.json",
        "docs/phase_b/b4_02_global_paper_evidence_index.md",
        "docs/phase_b/b4_02_final_paper_results_summary.md",
        "docs/phase_b/b4_02_release_checklist.md",
    ]
    payload = {
        "schema_version": "b4_02_final_local_paper_release_manifest_v1",
        "phase": "B4-02 Final Local Paper Release Closure",
        "status": "final_local_paper_release_frozen",
        "release_decision": "ready_for_push_pr_decision",
        "bound_identities": {
            "accepted_dlcm_identity": b2_ids.get("accepted_dlcm_identity"),
            "v5_deployment_identity": b2_ids.get("v5_deployment_identity"),
            "accepted_lse_identity": b2_ids.get("accepted_lse_identity"),
            "b2_phase_final_closure_identity": b2_manifest.get(
                "source_phase_final_closure_identity"
            ),
            "b3_06_phase_closure_identity": b3_manifest.get("phase_closure_identity"),
            "b4_01_weight_evidence_identity": b4_weight_manifest.get(
                "weight_evidence_identity"
            ),
        },
        "artifact_hashes": dict(b2_manifest.get("artifact_hashes", {})),
        "primary_claims": {
            "dlcm_sample_adaptive_fusion_supported": True,
            "dlcm_final_accepted": True,
            "lse_qualified": True,
            "early_exit_negative_result": True,
            "early_exit_accepted_mechanism": False,
            "full_depth_fallback_retained": True,
        },
        "paper_result_rows": [
            {
                "claim": "DLCM dynamic layer fusion",
                "status": "accepted",
                "evidence": "B2-08 + B4-01",
            },
            {
                "claim": "LSE layer-sufficiency validation",
                "status": "qualified",
                "evidence": "B2-06E/B2-08",
            },
            {
                "claim": "Early-exit",
                "status": "negative_result_future_work",
                "evidence": "B3-06",
            },
        ],
        "adaptive_weight_summary": {
            "calibration_records": b4_weight_manifest.get("calibration_records"),
            "sample_adaptive_variation_observed": b4_weight_manifest.get(
                "sample_adaptive_variation_observed"
            ),
            "uniform_equivalent_at_tolerance": b4_weight_manifest.get(
                "uniform_equivalent_at_tolerance"
            ),
            "deployment_max_sample_linf_delta_from_uniform": _nested(
                b4_weight_manifest, "deployment_weight_summary"
            ).get("max_sample_linf_delta_from_uniform"),
        },
        "evidence_documents": sorted(set(b2_docs + b3_docs + b4_docs + release_docs)),
        "boundary": {
            "training_started_in_release": False,
            "evaluation_started_in_release": False,
            "final_content_accessed_in_release": False,
            "model_artifact_generated_in_release": False,
            "tracked_pt_files": int(tracked_pt_count),
            "pushed": False,
            "pr_opened": False,
        },
    }
    payload["final_release_identity"] = canonical_json_sha256(payload)
    return payload
