"""B2-05C1 V2 protocol: error codes, unlocks, identity layering, gate schemas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

ERROR_CODES: tuple[str, ...] = (
    "B2_DLCM_V2_REAL_TRAINING_NOT_ENABLED",
    "B2_DLCM_V2_CONTRACT_MISMATCH",
    "B2_DLCM_FINAL_ROSTER_INSUFFICIENT",
    "B2_DLCM_FINAL_ROSTER_OVERLAP",
    "B2_DLCM_FINAL_ROSTER_SOURCE_INVALID",
    "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN",
    "B2_DLCM_DEVELOPMENT_UNQUALIFIED",
    "B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_REQUIRED",
    "B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_USED",
    "B2_DLCM_FINAL_MATERIALIZATION_MISMATCH",
    "B2_DLCM_FINAL_EVALUATION_UNLOCK_REQUIRED",
    "B2_DLCM_FINAL_EVALUATION_MISMATCH",
    "B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID",
    "B2_DLCM_FINAL_DECISION_INVALID",
    "B2_DLCM_FINAL_EVIDENCE_INVALID",
    "B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN",
)

SCHEMA_VERSION = "b2_dlcm_v2_protocol_v1"
FORBIDDEN_BYPASS_FLAGS = (
    "force_unlock",
    "bypass_gates",
    "skip_development",
    "allow_final_without_development",
    "ignore_auxiliary_diagnostics",
)


class B2DLCMV2ProtocolError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV2ProtocolError(code, detail)


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def reject_bypass_flags(config: Mapping[str, Any]) -> None:
    for flag in FORBIDDEN_BYPASS_FLAGS:
        if flag in config and config[flag]:
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", f"bypass flag forbidden: {flag}")


def require_real_training_enabled(config: Mapping[str, Any], *, dry_run: bool) -> None:
    reject_bypass_flags(config)
    if dry_run:
        return
    if config.get("real_training_enabled") is not True:
        _fail(
            "B2_DLCM_V2_REAL_TRAINING_NOT_ENABLED",
            "real training disabled by contract",
        )


def forbid_final_content_access(*, unlocked: bool, context: str) -> None:
    if not unlocked:
        _fail(
            "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN",
            f"final content access forbidden before unlock ({context})",
        )


def build_materialization_unlock(
    *,
    development_go: bool,
    development_evidence_sha256: str,
    implementation_commit: str,
) -> dict[str, Any]:
    if not development_go:
        _fail("B2_DLCM_DEVELOPMENT_UNQUALIFIED", "development must pass before materialization unlock")
    payload = {
        "schema_version": "b2_dlcm_v2_final_materialization_unlock_v1",
        "materialization_unlocked": True,
        "consumed": False,
        "development_go": True,
        "development_evidence_sha256": development_evidence_sha256,
        "implementation_commit": implementation_commit,
    }
    payload["unlock_scientific_sha256"] = canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "unlock_scientific_sha256"}
    )
    return payload


def consume_materialization_unlock(unlock: Mapping[str, Any]) -> dict[str, Any]:
    if unlock.get("materialization_unlocked") is not True:
        _fail("B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_REQUIRED", "unlock missing/false")
    if unlock.get("consumed") is True:
        _fail("B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_USED", "unlock already consumed")
    claimed = unlock.get("unlock_scientific_sha256")
    recomputed = canonical_json_sha256(
        {k: v for k, v in unlock.items() if k != "unlock_scientific_sha256"}
    )
    if claimed != recomputed:
        _fail("B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_REQUIRED", "unlock hash mismatch")
    out = dict(unlock)
    out["consumed"] = True
    out["unlock_scientific_sha256"] = canonical_json_sha256(
        {k: v for k, v in out.items() if k != "unlock_scientific_sha256"}
    )
    return out


def require_evaluation_unlock(unlock: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if unlock is None or unlock.get("evaluation_unlocked") is not True:
        _fail("B2_DLCM_FINAL_EVALUATION_UNLOCK_REQUIRED", "final evaluation unlock required")
    return unlock


def assert_ab_equality(*, label: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    ha = canonical_json_sha256(a)
    hb = canonical_json_sha256(b)
    if ha != hb:
        code = (
            "B2_DLCM_FINAL_MATERIALIZATION_MISMATCH"
            if label == "materialization"
            else "B2_DLCM_FINAL_EVALUATION_MISMATCH"
        )
        _fail(code, f"{label} A/B scientific hash mismatch")
    return ha


def build_h_decision(
    *,
    gt_target_learning: Mapping[str, Any],
    localization: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    verdict: str,
) -> dict[str, Any]:
    forbidden = ("development", "teacher", "aux", "development_go")
    for key in list(gt_target_learning) + list(localization) + list(thresholds):
        lowered = key.lower()
        if any(tok in lowered for tok in forbidden):
            _fail("B2_DLCM_FINAL_DECISION_INVALID", f"development/teacher field in H_decision: {key}")
    payload = {
        "schema_version": "b2_dlcm_v2_h_decision_v1",
        "gt_target_learning": dict(gt_target_learning),
        "localization": dict(localization),
        "thresholds": dict(thresholds),
        "verdict": verdict,
    }
    payload["H_decision"] = canonical_json_sha256(payload)
    return payload


def build_h_evidence(
    *,
    h_decision: str,
    development_go: bool,
    auxiliary_diagnostics_sha256: str,
    materialization_ab_sha256: str,
    evaluation_ab_sha256: str,
    production_metric_proof: Mapping[str, Any],
    coverage_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not auxiliary_diagnostics_sha256:
        _fail("B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID", "diagnostics hash required for evidence")
    payload = {
        "schema_version": "b2_dlcm_v2_h_evidence_v1",
        "H_decision": h_decision,
        "development_go": bool(development_go),
        "auxiliary_diagnostics_sha256": auxiliary_diagnostics_sha256,
        "materialization_ab_sha256": materialization_ab_sha256,
        "evaluation_ab_sha256": evaluation_ab_sha256,
        "production_metric_proof": dict(production_metric_proof),
        "coverage_provenance": dict(coverage_provenance),
    }
    payload["H_evidence"] = canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "H_evidence"}
    )
    return payload


def build_h_accepted(
    *,
    h_deploy: str,
    h_decision: str,
    h_evidence: str,
    h_selection: str,
    upstream: Mapping[str, Any],
    v2_contract_sha256: str,
    final_passed: bool,
) -> dict[str, Any]:
    if not final_passed:
        _fail("B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN", "accepted identity forbidden before final pass")
    payload = {
        "schema_version": "b2_dlcm_v2_h_accepted_v1",
        "H_deploy": h_deploy,
        "H_decision": h_decision,
        "H_evidence": h_evidence,
        "H_selection": h_selection,
        "upstream": dict(upstream),
        "v2_contract_sha256": v2_contract_sha256,
    }
    payload["H_accepted"] = canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "H_accepted"}
    )
    return payload


def persist_json_atomic(path: Path | str, payload: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_json_receipt(path: Path | str) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", f"missing artifact {path}")
    receipt = Path(str(path) + ".sha256")
    if not receipt.is_file():
        receipt = path.with_suffix(path.suffix + ".sha256")
    if not receipt.is_file():
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", f"missing receipt for {path}")
    claimed = receipt.read_text(encoding="utf-8").strip().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if claimed != actual:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", f"receipt mismatch for {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup_partial_artifacts(paths: Sequence[Path | str]) -> None:
    for raw in paths:
        path = Path(raw)
        if path.exists():
            path.unlink()
        side = path.with_suffix(path.suffix + ".sha256")
        if side.exists():
            side.unlink()
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
