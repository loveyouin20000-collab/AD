"""B2-06A fail-closed LSE preflight gate bound to accepted V5 artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import yaml


class B2LSEAcceptedGateError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2LSEAcceptedGateError(code, detail)


@dataclass(frozen=True)
class LSEPreflightConfig:
    path: Path
    repo_root: Path
    values: Mapping[str, Any]


def _as_path(value: Any, *, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        _fail("B2_LSE_ACCEPTED_GATE_CONFIG_INVALID", "path value required")
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def load_lse_preflight_config(path: Path | str, *, repo_root: Path | str | None = None) -> LSEPreflightConfig:
    config_path = Path(path)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    if not config_path.is_absolute():
        config_path = root / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        _fail("B2_LSE_ACCEPTED_GATE_CONFIG_INVALID", "config must be a mapping")
    lse = raw.get("lse")
    if not isinstance(lse, Mapping):
        _fail("B2_LSE_ACCEPTED_GATE_CONFIG_INVALID", "config missing lse mapping")
    return LSEPreflightConfig(path=config_path, repo_root=root, values=dict(lse))


def _load_json(path: Path, *, missing_code: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(missing_code, f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _fail("B2_LSE_ACCEPTED_GATE_INVALID", f"{path} must contain JSON object")
    return payload


def _require_field(payload: Mapping[str, Any], key: str, expected: Any, *, code: str) -> None:
    if payload.get(key) != expected:
        _fail(code, f"{key} mismatch")


def _required_lse_value(config: LSEPreflightConfig, key: str, *, code: str) -> Any:
    value = config.values.get(key)
    if value in (None, ""):
        _fail(code, f"missing lse.{key}")
    return value


def _accepted_reference_root(config: LSEPreflightConfig, accepted_path: Path) -> Path:
    configured = config.values.get("accepted_reference_root")
    if configured:
        return _as_path(configured, repo_root=config.repo_root).resolve()
    return (accepted_path.parent / "accepted_refs").resolve()


def _require_checkpoint_accepted_bound(config: LSEPreflightConfig, accepted_path: Path) -> Path:
    checkpoint = _as_path(_required_lse_value(config, "dlcm_checkpoint", code="B2_LSE_CHECKPOINT_REQUIRED"), repo_root=config.repo_root)
    reference_root = _accepted_reference_root(config, accepted_path)
    try:
        checkpoint.resolve().relative_to(reference_root)
    except ValueError:
        _fail(
            "B2_LSE_CHECKPOINT_NOT_ACCEPTED_BOUND",
            "DLCM checkpoint must be under the accepted artifact reference root",
        )
    return checkpoint


def validate_accepted_manifest_chain(config: LSEPreflightConfig) -> dict[str, Any]:
    accepted_path = _as_path(
        _required_lse_value(config, "accepted_manifest", code="B2_LSE_ACCEPTED_MANIFEST_REQUIRED"),
        repo_root=config.repo_root,
    )
    decision_path = _as_path(
        _required_lse_value(config, "final_decision_manifest", code="B2_LSE_DECISION_MANIFEST_REQUIRED"),
        repo_root=config.repo_root,
    )
    evidence_path = _as_path(
        _required_lse_value(config, "final_evidence_manifest", code="B2_LSE_EVIDENCE_MANIFEST_REQUIRED"),
        repo_root=config.repo_root,
    )
    accepted = _load_json(accepted_path, missing_code="B2_LSE_ACCEPTED_MANIFEST_REQUIRED")
    decision = _load_json(decision_path, missing_code="B2_LSE_DECISION_MANIFEST_REQUIRED")
    evidence = _load_json(evidence_path, missing_code="B2_LSE_EVIDENCE_MANIFEST_REQUIRED")
    if accepted.get("schema_version") != "b2_dlcm_v5_accepted_deployment_manifest_v1":
        _fail("B2_LSE_ACCEPTED_MANIFEST_INVALID", "accepted manifest schema mismatch")
    if accepted.get("deployment_qualified") is not True:
        _fail("B2_LSE_ACCEPTED_MANIFEST_INVALID", "deployment_qualified must be true")
    if decision.get("verdict") != "qualified":
        _fail("B2_LSE_FINAL_DECISION_UNQUALIFIED", "Final decision must be qualified")
    expected_accepted = _required_lse_value(
        config,
        "expected_accepted_identity",
        code="B2_LSE_ACCEPTED_IDENTITY_REQUIRED",
    )
    _require_field(
        accepted,
        "accepted_identity",
        expected_accepted,
        code="B2_LSE_ACCEPTED_IDENTITY_MISMATCH",
    )
    expected_deploy = _required_lse_value(
        config,
        "expected_v5_deployment_identity",
        code="B2_LSE_DEPLOYMENT_IDENTITY_REQUIRED",
    )
    _require_field(
        accepted,
        "v5_deployment_identity",
        expected_deploy,
        code="B2_LSE_DEPLOYMENT_IDENTITY_MISMATCH",
    )
    expected_decision = _required_lse_value(config, "expected_H_decision", code="B2_LSE_DECISION_IDENTITY_REQUIRED")
    expected_evidence = _required_lse_value(config, "expected_H_evidence", code="B2_LSE_EVIDENCE_IDENTITY_REQUIRED")
    _require_field(accepted, "H_decision", expected_decision, code="B2_LSE_DECISION_IDENTITY_MISMATCH")
    _require_field(accepted, "H_evidence", expected_evidence, code="B2_LSE_EVIDENCE_IDENTITY_MISMATCH")
    _require_field(decision, "H_decision", accepted["H_decision"], code="B2_LSE_DECISION_IDENTITY_MISMATCH")
    _require_field(evidence, "H_decision", accepted["H_decision"], code="B2_LSE_DECISION_IDENTITY_MISMATCH")
    _require_field(evidence, "H_evidence", accepted["H_evidence"], code="B2_LSE_EVIDENCE_IDENTITY_MISMATCH")
    if str(accepted.get("beta_star_decimal")) != str(config.values.get("expected_beta_star_decimal", "0.54")):
        _fail("B2_LSE_BETA_IDENTITY_MISMATCH", "beta* mismatch")
    if config.values.get("expected_calibration_ab_identity") is not None:
        _require_field(
            accepted,
            "calibration_ab_identity",
            config.values["expected_calibration_ab_identity"],
            code="B2_LSE_CALIBRATION_IDENTITY_MISMATCH",
        )
    checkpoint = _require_checkpoint_accepted_bound(config, accepted_path)
    return {
        "accepted_path": accepted_path,
        "decision_path": decision_path,
        "evidence_path": evidence_path,
        "accepted": accepted,
        "decision": decision,
        "evidence": evidence,
        "checkpoint": checkpoint,
    }


def run_lse_preflight(config: LSEPreflightConfig) -> dict[str, Any]:
    chain = validate_accepted_manifest_chain(config)
    prerequisites = {
        "dlcm_checkpoint": chain["checkpoint"],
        "train_gain_targets": _as_path(_required_lse_value(config, "train_gain_targets", code="B2_LSE_GAIN_TARGETS_REQUIRED"), repo_root=config.repo_root),
        "calibration_gain_targets": _as_path(_required_lse_value(config, "calibration_gain_targets", code="B2_LSE_GAIN_TARGETS_REQUIRED"), repo_root=config.repo_root),
        "train_cache": _as_path(_required_lse_value(config, "train_cache", code="B2_LSE_TEACHER_CACHE_REQUIRED"), repo_root=config.repo_root),
        "calibration_cache": _as_path(_required_lse_value(config, "calibration_cache", code="B2_LSE_TEACHER_CACHE_REQUIRED"), repo_root=config.repo_root),
        "descriptor_stats": _as_path(_required_lse_value(config, "descriptor_stats", code="B2_LSE_DESCRIPTOR_STATS_REQUIRED"), repo_root=config.repo_root),
    }
    missing = [name for name, path in prerequisites.items() if not path.exists()]
    return {
        "schema_version": "b2_lse_accepted_gate_preflight_v1",
        "accepted_gate_passed": True,
        "training_started": False,
        "ready": not missing,
        "missing_prerequisites": missing,
        "accepted_identity": chain["accepted"]["accepted_identity"],
        "v5_deployment_identity": chain["accepted"]["v5_deployment_identity"],
        "H_decision": chain["accepted"]["H_decision"],
        "H_evidence": chain["accepted"]["H_evidence"],
        "dlcm_checkpoint": str(chain["checkpoint"]),
    }
