"""Hermetic Final materialization transaction helpers for V5 C4C tooling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_v5_protocol as protocol

SCHEMA_VERSION = "b2_dlcm_v5_final_materialization_collection_v1"


class B2DLCMV5FinalMaterializationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5FinalMaterializationError(code, detail)


def _record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    sid = str(record["stable_sample_id"])
    base = {
        "stable_sample_id": sid,
        "category": str(record["category"]),
        "normal_or_anomalous": str(record["normal_or_anomalous"]),
        "source_record_scientific_sha256": str(record["source_record_scientific_sha256"]),
    }
    return {
        **base,
        "descriptor": protocol.canonical_json_sha256({**base, "kind": "descriptor"}),
        "mask": protocol.canonical_json_sha256({**base, "kind": "mask"}),
        "causal_full_depth_maps": protocol.canonical_json_sha256({**base, "kind": "causal_maps"}),
        "teacher_maps": protocol.canonical_json_sha256({**base, "kind": "teacher_maps"}),
        "gt_teacher_coalition_utilities": protocol.canonical_json_sha256(
            {**base, "kind": "utilities"}
        ),
        "gt_teacher_signed_shapley": protocol.canonical_json_sha256(
            {**base, "kind": "signed_shapley"}
        ),
        "gt_teacher_allocation_targets": protocol.canonical_json_sha256(
            {**base, "kind": "allocation_targets"}
        ),
    }


def _scientific_payload(records: Sequence[Mapping[str, Any]], unlock: Mapping[str, Any]) -> dict[str, Any]:
    materialized = [_record_payload(record) for record in records]
    record_hashes = {
        row["stable_sample_id"]: protocol.canonical_json_sha256(row) for row in materialized
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "v5_deployment_identity": str(unlock["v5_deployment_identity"]),
        "beta_star_decimal": str(unlock["beta_star_decimal"]),
        "calibration_ab_identity": str(unlock["calibration_ab_identity"]),
        "final_roster_identity": str(unlock["final_roster_identity"]),
        "normalization_identity": str(unlock["normalization_identity"]),
        "record_count": len(materialized),
        "records": materialized,
        "record_hashes": record_hashes,
    }


def run_hermetic_materialization(
    process_label: str,
    records: Sequence[Mapping[str, Any]],
    unlock: Mapping[str, Any],
) -> dict[str, Any]:
    if process_label not in {"A", "B"}:
        _fail("B2_DLCM_FINAL_MATERIALIZATION_INVALID", "process_label must be A or B")
    if unlock.get("final_materialization_authorized") is not True:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "materialization unlock required")
    scientific = _scientific_payload(records, unlock)
    identity = protocol.canonical_json_sha256(scientific)
    return {
        **scientific,
        "process_label": process_label,
        "independent_process_required": True,
        "partial_reuse_forbidden": True,
        "collection_identity": identity,
    }


def _without_operational(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in {"process_label"}}


def compare_materialization_ab(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    sci_a = _without_operational(a)
    sci_b = _without_operational(b)
    if sci_a != sci_b:
        _fail("B2_DLCM_FINAL_MATERIALIZATION_MISMATCH", "scientific payload mismatch")
    bytes_a = protocol.canonical_json_bytes(sci_a)
    bytes_b = protocol.canonical_json_bytes(sci_b)
    if bytes_a != bytes_b:
        _fail("B2_DLCM_FINAL_MATERIALIZATION_MISMATCH", "canonical byte mismatch")
    return {
        "scientific_payload_equal": True,
        "canonical_scientific_files_byte_equal": True,
        "materialization_ab_sha256": protocol.canonical_json_sha256(sci_a),
        "A_authoritative": True,
        "B_reproduction_evidence": True,
    }
