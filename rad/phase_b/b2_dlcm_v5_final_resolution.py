"""Authorized stable-ID resolution for B2-05C4C V5 Final tooling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_v5_protocol as protocol


class B2DLCMV5FinalResolutionError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5FinalResolutionError(code, detail)


def _source_identity(record: Mapping[str, Any]) -> str:
    existing = record.get("source_record_scientific_sha256")
    if isinstance(existing, str) and len(existing) == 64:
        return existing
    scientific = {
        "stable_sample_id": record["stable_sample_id"],
        "category": record["category"],
        "normal_or_anomalous": record.get("normal_or_anomalous"),
        "image_label": record.get("image_label"),
    }
    return protocol.canonical_json_sha256(scientific)


def resolve_stable_ids(
    source_manifest: Mapping[str, Any],
    roster: Mapping[str, Any],
    *,
    authorized: bool,
) -> dict[str, Any]:
    if not authorized:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "resolution requires valid unlock")
    if roster.get("final_content_resolved") is not False or roster.get("paths_present") is not False:
        _fail("B2_DLCM_FINAL_ROSTER_MISMATCH", "roster must be identity-only before resolution")
    source_rows = source_manifest.get("records")
    roster_rows = roster.get("records")
    if not isinstance(source_rows, list) or not isinstance(roster_rows, list):
        _fail("B2_DLCM_FINAL_SOURCE_MANIFEST_INVALID", "source and roster require records")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        if not isinstance(row, Mapping):
            _fail("B2_DLCM_FINAL_SOURCE_MANIFEST_INVALID", "source record must be object")
        sid = str(row.get("stable_sample_id", ""))
        if sid in by_id:
            _fail("B2_DLCM_FINAL_SOURCE_MANIFEST_INVALID", f"duplicate stable id {sid}")
        by_id[sid] = row
    resolved: list[dict[str, Any]] = []
    for raw in roster_rows:
        if not isinstance(raw, Mapping):
            _fail("B2_DLCM_FINAL_ROSTER_MISMATCH", "roster record must be object")
        sid = str(raw["stable_sample_id"])
        src = by_id.get(sid)
        if src is None:
            _fail("B2_DLCM_FINAL_STABLE_ID_NOT_FOUND", f"stable id {sid} missing")
        if src.get("category") != raw.get("category"):
            _fail("B2_DLCM_FINAL_SOURCE_RECORD_MISMATCH", f"category mismatch for {sid}")
        expected_label = 0 if raw.get("normal_or_anomalous") == "normal" else 1
        if int(src.get("image_label", expected_label)) != expected_label:
            _fail("B2_DLCM_FINAL_SOURCE_RECORD_MISMATCH", f"label mismatch for {sid}")
        if _source_identity(src) != raw.get("source_record_scientific_sha256"):
            _fail("B2_DLCM_FINAL_SOURCE_RECORD_MISMATCH", f"source identity mismatch for {sid}")
        image_identity = src.get("image_identity")
        if not isinstance(image_identity, str) or not image_identity:
            _fail("B2_DLCM_FINAL_SOURCE_RECORD_MISMATCH", f"image missing for {sid}")
        resolved.append(
            {
                "stable_sample_id": sid,
                "category": str(raw["category"]),
                "normal_or_anomalous": str(raw["normal_or_anomalous"]),
                "image_identity": image_identity,
                "mask_identity": src.get("mask_identity"),
                "source_record_scientific_sha256": str(raw["source_record_scientific_sha256"]),
            }
        )
    return {
        "stable_ids_resolved": True,
        "resolution_scientific_identity_included": False,
        "records": resolved,
    }
