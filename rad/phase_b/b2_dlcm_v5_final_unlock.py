"""B2-05C4C V5 Final execution plan and unlock guards."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

from rad.phase_b import b2_dlcm_v5_protocol as protocol

PLAN_SCHEMA_VERSION = "b2_dlcm_v5_final_execution_plan_v1"
MATERIALIZATION_UNLOCK_SCHEMA_VERSION = "b2_dlcm_v5_final_materialization_unlock_v1"
EVALUATION_UNLOCK_SCHEMA_VERSION = "b2_dlcm_v5_final_evaluation_unlock_v1"

PLAN_EXPECTED_FIELDS = (
    "v5_deployment_identity",
    "beta_star_decimal",
    "calibration_ab_identity",
    "development_qualified_identity",
    "final_roster_identity",
    "source_master_manifest_identity",
    "normalization_identity",
    "tooling_commit",
    "tooling_tag",
)

UNLOCK_EXPECTED_FIELDS = (
    *PLAN_EXPECTED_FIELDS,
    "accepted_v5_final_execution_plan_scientific_sha256",
)


class B2DLCMV5FinalUnlockError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV5FinalUnlockError(code, detail)


def _require_mapping(value: Any, *, code: str, detail: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code, detail)
    return value


def _require_expected(expected: Mapping[str, Any]) -> None:
    missing = [field for field in PLAN_EXPECTED_FIELDS if not expected.get(field)]
    if missing:
        _fail("B2_DLCM_FINAL_UNLOCK_INVALID", f"missing expected fields: {missing}")
    if expected.get("beta_star_decimal") != "0.54":
        _fail("B2_DLCM_FINAL_UNLOCK_INVALID", "beta* must remain 0.54")
    if expected.get("worktree_clean") is False:
        _fail("B2_DLCM_FINAL_UNLOCK_INVALID", "worktree must be clean")
    head = expected.get("head_commit")
    tooling = expected.get("tooling_commit")
    if head is not None and tooling is not None and str(head) != str(tooling):
        _fail("B2_DLCM_FINAL_UNLOCK_INVALID", "HEAD must equal tooling commit")


def build_final_execution_plan(
    *,
    config: Mapping[str, Any],
    repo_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Build the path-free, runtime-free C4C Final execution plan."""

    _require_expected(config)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "stage": "b2_05c4c",
        "scope": "final_execution_tooling_only",
        "c4_not_c5": True,
        "real_final_content_accessed": False,
        "v5_deployment_identity": str(config["v5_deployment_identity"]),
        "beta_star_decimal": str(config["beta_star_decimal"]),
        "calibration_ab_identity": str(config["calibration_ab_identity"]),
        "development_qualified_identity": str(config["development_qualified_identity"]),
        "final_roster_identity": str(config["final_roster_identity"]),
        "source_master_manifest_identity": str(config["source_master_manifest_identity"]),
        "normalization_identity": str(config["normalization_identity"]),
        "tooling_commit": str(config["tooling_commit"]),
        "tooling_tag": str(config["tooling_tag"]),
        "repo_head": str(repo_identity.get("head", config["tooling_commit"])),
        "materialization_protocol": dict(config.get("materialization_protocol", {})),
        "evaluation_protocol": dict(config.get("evaluation_protocol", {})),
        "final_gates": dict(config.get("final_gates", {})),
        "schemas": {
            "materialization_unlock": MATERIALIZATION_UNLOCK_SCHEMA_VERSION,
            "evaluation_unlock": EVALUATION_UNLOCK_SCHEMA_VERSION,
            "decision": "b2_dlcm_v5_final_decision_manifest_v1",
            "evidence": "b2_dlcm_v5_final_evidence_manifest_v1",
            "accepted": "b2_dlcm_v5_accepted_deployment_manifest_v1",
        },
    }


def final_execution_plan_sha256(plan: Mapping[str, Any]) -> str:
    return protocol.canonical_json_sha256(dict(plan))


def dry_run_status(*, plan_sha256: str) -> dict[str, Any]:
    return {
        "mode": "dry_run",
        "accepted_v5_final_execution_plan_scientific_sha256": plan_sha256,
        "real_final_content_accessed": False,
        "stable_ids_resolved": False,
        "materialization_started": False,
        "evaluation_started": False,
        "accepted_written": False,
        "run_directory_created": False,
        "artifact_written": False,
    }


def build_materialization_unlock(*, expected: Mapping[str, Any]) -> dict[str, Any]:
    _require_expected(expected)
    payload = {
        "schema_version": MATERIALIZATION_UNLOCK_SCHEMA_VERSION,
        "single_use": True,
        "consumed": False,
        "final_materialization_authorized": True,
        "v5_deployment_identity": str(expected["v5_deployment_identity"]),
        "beta_star_decimal": str(expected["beta_star_decimal"]),
        "calibration_ab_identity": str(expected["calibration_ab_identity"]),
        "development_qualified_identity": str(expected["development_qualified_identity"]),
        "final_roster_identity": str(expected["final_roster_identity"]),
        "source_master_manifest_identity": str(expected["source_master_manifest_identity"]),
        "normalization_identity": str(expected["normalization_identity"]),
        "tooling_commit": str(expected["tooling_commit"]),
        "tooling_tag": str(expected["tooling_tag"]),
        "accepted_v5_final_execution_plan_scientific_sha256": str(
            expected["accepted_v5_final_execution_plan_scientific_sha256"]
        ),
        "head_commit": str(expected.get("head_commit", expected["tooling_commit"])),
        "worktree_clean": bool(expected.get("worktree_clean", True)),
        "config_identity": str(expected.get("config_identity", "")),
    }
    payload["unlock_scientific_sha256"] = protocol.canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "unlock_scientific_sha256"}
    )
    return payload


def _validate_common_unlock(unlock: Mapping[str, Any], *, expected: Mapping[str, Any]) -> None:
    _require_expected(expected)
    if unlock.get("single_use") is not True:
        _fail("B2_DLCM_FINAL_UNLOCK_INVALID", "single_use must be true")
    if unlock.get("consumed") is True:
        _fail("B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_USED", "unlock already consumed")
    for key in UNLOCK_EXPECTED_FIELDS:
        if str(unlock.get(key, "")) != str(expected[key]):
            _fail("B2_DLCM_FINAL_UNLOCK_INVALID", f"{key} mismatch")
    if unlock.get("head_commit") != expected.get("head_commit", expected["tooling_commit"]):
        _fail("B2_DLCM_FINAL_UNLOCK_INVALID", "head_commit mismatch")
    if unlock.get("worktree_clean") is not True:
        _fail("B2_DLCM_FINAL_UNLOCK_INVALID", "worktree must be clean")
    claimed = unlock.get("unlock_scientific_sha256")
    recomputed = protocol.canonical_json_sha256(
        {k: v for k, v in unlock.items() if k != "unlock_scientific_sha256"}
    )
    if claimed != recomputed:
        _fail("B2_DLCM_FINAL_UNLOCK_INVALID", "unlock scientific hash mismatch")


def validate_materialization_unlock(
    unlock: Mapping[str, Any] | None,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if unlock is None:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "materialization unlock required")
    value = dict(_require_mapping(unlock, code="B2_DLCM_FINAL_UNLOCK_INVALID", detail="unlock must be object"))
    if value.get("schema_version") != MATERIALIZATION_UNLOCK_SCHEMA_VERSION:
        _fail("B2_DLCM_FINAL_UNLOCK_INVALID", "materialization unlock schema mismatch")
    if value.get("final_materialization_authorized") is not True:
        _fail("B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN", "materialization authorization missing")
    _validate_common_unlock(value, expected=expected)
    return value


def consume_materialization_unlock(unlock: Mapping[str, Any]) -> dict[str, Any]:
    if unlock.get("consumed") is True:
        _fail("B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_USED", "unlock already consumed")
    out = dict(unlock)
    out["consumed"] = True
    out["unlock_scientific_sha256"] = protocol.canonical_json_sha256(
        {k: v for k, v in out.items() if k != "unlock_scientific_sha256"}
    )
    return out


def build_evaluation_unlock(
    *,
    expected: Mapping[str, Any],
    materialization_identity: str,
) -> dict[str, Any]:
    unlock = build_materialization_unlock(expected=expected)
    unlock["schema_version"] = EVALUATION_UNLOCK_SCHEMA_VERSION
    unlock["final_materialization_authorized"] = False
    unlock["final_evaluation_authorized"] = True
    unlock["authoritative_materialization_identity"] = materialization_identity
    unlock["unlock_scientific_sha256"] = protocol.canonical_json_sha256(
        {k: v for k, v in unlock.items() if k != "unlock_scientific_sha256"}
    )
    return unlock
