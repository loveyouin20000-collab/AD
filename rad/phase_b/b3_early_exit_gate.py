"""B3-01 fail-closed early-exit preflight gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import yaml


class B3EarlyExitGateError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B3EarlyExitGateError(code, detail)


@dataclass(frozen=True)
class EarlyExitPreflightConfig:
    path: Path
    repo_root: Path
    values: Mapping[str, Any]


def _as_path(value: Any, *, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        _fail("B3_EARLY_EXIT_CONFIG_INVALID", "path value required")
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def load_early_exit_preflight_config(
    path: Path | str, *, repo_root: Path | str | None = None
) -> EarlyExitPreflightConfig:
    config_path = Path(path)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    if not config_path.is_absolute():
        config_path = root / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        _fail("B3_EARLY_EXIT_CONFIG_INVALID", "config must be a mapping")
    early_exit = raw.get("early_exit")
    if not isinstance(early_exit, Mapping):
        _fail("B3_EARLY_EXIT_CONFIG_INVALID", "config missing early_exit mapping")
    return EarlyExitPreflightConfig(path=config_path, repo_root=root, values=dict(early_exit))


def _load_json(path: Path, *, missing_code: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(missing_code, f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _fail("B3_EARLY_EXIT_MANIFEST_INVALID", f"{path} must contain JSON object")
    return payload


def _required(config: EarlyExitPreflightConfig, key: str, *, code: str) -> Any:
    value = config.values.get(key)
    if value in (None, ""):
        _fail(code, f"missing early_exit.{key}")
    return value


def _require_equal(actual: Any, expected: Any, *, code: str, key: str) -> None:
    if actual != expected:
        _fail(code, f"{key} mismatch")


def _require_false(value: Any, *, code: str, key: str) -> None:
    if value is not False:
        _fail(code, f"{key} must be false")


def _require_true(value: Any, *, code: str, key: str) -> None:
    if value is not True:
        _fail(code, f"{key} must be true")


def _accepted_reference_root(config: EarlyExitPreflightConfig, manifest_path: Path) -> Path:
    configured = config.values.get("accepted_lse_reference_root")
    if configured:
        return _as_path(configured, repo_root=config.repo_root).resolve()
    return (manifest_path.parent / "accepted_refs").resolve()


def _require_lse_checkpoint_accepted_bound(
    config: EarlyExitPreflightConfig, manifest_path: Path
) -> Path:
    checkpoint = _as_path(
        _required(
            config,
            "lse_checkpoint",
            code="B3_EARLY_EXIT_LSE_CHECKPOINT_REQUIRED",
        ),
        repo_root=config.repo_root,
    )
    reference_root = _accepted_reference_root(config, manifest_path)
    try:
        checkpoint.resolve().relative_to(reference_root)
    except ValueError:
        _fail(
            "B3_EARLY_EXIT_LSE_CHECKPOINT_NOT_ACCEPTED_BOUND",
            "LSE checkpoint must be under the accepted LSE reference root",
        )
    return checkpoint


def validate_accepted_b2_chain(config: EarlyExitPreflightConfig) -> dict[str, Any]:
    lse_manifest_path = _as_path(
        _required(
            config,
            "accepted_lse_manifest",
            code="B3_EARLY_EXIT_ACCEPTED_LSE_MANIFEST_REQUIRED",
        ),
        repo_root=config.repo_root,
    )
    b2_closure_path = _as_path(
        _required(
            config,
            "b2_phase_final_closure_manifest",
            code="B3_EARLY_EXIT_B2_CLOSURE_MANIFEST_REQUIRED",
        ),
        repo_root=config.repo_root,
    )
    lse = _load_json(
        lse_manifest_path,
        missing_code="B3_EARLY_EXIT_ACCEPTED_LSE_MANIFEST_REQUIRED",
    )
    b2 = _load_json(
        b2_closure_path,
        missing_code="B3_EARLY_EXIT_B2_CLOSURE_MANIFEST_REQUIRED",
    )
    if lse.get("schema_version") != "b2_06f_lse_accepted_artifact_manifest_v1":
        _fail("B3_EARLY_EXIT_ACCEPTED_LSE_MANIFEST_INVALID", "accepted LSE schema mismatch")
    if b2.get("schema_version") != "b2_07_phase_final_closure_manifest_v1":
        _fail("B3_EARLY_EXIT_B2_CLOSURE_MANIFEST_INVALID", "B2 closure schema mismatch")
    _require_true(
        lse.get("accepted_artifact_generated"),
        code="B3_EARLY_EXIT_ACCEPTED_LSE_MANIFEST_INVALID",
        key="accepted_artifact_generated",
    )
    _require_true(
        lse.get("lse_qualified"),
        code="B3_EARLY_EXIT_ACCEPTED_LSE_MANIFEST_INVALID",
        key="lse_qualified",
    )
    _require_false(
        lse.get("training_started"),
        code="B3_EARLY_EXIT_ACCEPTED_LSE_BOUNDARY_VIOLATION",
        key="lse.training_started",
    )
    _require_false(
        lse.get("evaluation_started"),
        code="B3_EARLY_EXIT_ACCEPTED_LSE_BOUNDARY_VIOLATION",
        key="lse.evaluation_started",
    )
    _require_equal(
        lse.get("accepted_dlcm_identity"),
        _required(
            config,
            "expected_accepted_dlcm_identity",
            code="B3_EARLY_EXIT_ACCEPTED_DLCM_IDENTITY_REQUIRED",
        ),
        code="B3_EARLY_EXIT_ACCEPTED_DLCM_IDENTITY_MISMATCH",
        key="accepted_dlcm_identity",
    )
    _require_equal(
        lse.get("v5_deployment_identity"),
        _required(
            config,
            "expected_v5_deployment_identity",
            code="B3_EARLY_EXIT_V5_DEPLOYMENT_IDENTITY_REQUIRED",
        ),
        code="B3_EARLY_EXIT_V5_DEPLOYMENT_IDENTITY_MISMATCH",
        key="v5_deployment_identity",
    )
    _require_equal(
        lse.get("accepted_lse_identity"),
        _required(
            config,
            "expected_accepted_lse_identity",
            code="B3_EARLY_EXIT_ACCEPTED_LSE_IDENTITY_REQUIRED",
        ),
        code="B3_EARLY_EXIT_ACCEPTED_LSE_IDENTITY_MISMATCH",
        key="accepted_lse_identity",
    )
    _require_equal(
        b2.get("phase_final_closure_identity"),
        _required(
            config,
            "expected_b2_phase_final_closure_identity",
            code="B3_EARLY_EXIT_B2_CLOSURE_IDENTITY_REQUIRED",
        ),
        code="B3_EARLY_EXIT_B2_CLOSURE_IDENTITY_MISMATCH",
        key="phase_final_closure_identity",
    )
    for key in ("accepted_dlcm_identity", "v5_deployment_identity", "accepted_lse_identity"):
        _require_equal(
            b2.get(key),
            lse.get(key),
            code="B3_EARLY_EXIT_B2_CHAIN_IDENTITY_MISMATCH",
            key=key,
        )
    _require_equal(
        b2.get("accepted_lse_checkpoint_sha256"),
        lse.get("accepted_lse_checkpoint_sha256"),
        code="B3_EARLY_EXIT_B2_CHAIN_IDENTITY_MISMATCH",
        key="accepted_lse_checkpoint_sha256",
    )
    if b2.get("tracked_pt_count") != 0:
        _fail("B3_EARLY_EXIT_TRACKED_PT", "B2 closure tracked .pt count must be zero")
    _require_false(
        b2.get("training_started_in_b2_07"),
        code="B3_EARLY_EXIT_B2_CLOSURE_BOUNDARY_VIOLATION",
        key="training_started_in_b2_07",
    )
    _require_false(
        b2.get("evaluation_started_in_b2_07"),
        code="B3_EARLY_EXIT_B2_CLOSURE_BOUNDARY_VIOLATION",
        key="evaluation_started_in_b2_07",
    )
    _require_false(
        b2.get("final_content_accessed_in_b2_07"),
        code="B3_EARLY_EXIT_B2_CLOSURE_BOUNDARY_VIOLATION",
        key="final_content_accessed_in_b2_07",
    )
    checkpoint = _require_lse_checkpoint_accepted_bound(config, lse_manifest_path)
    return {
        "accepted_lse_manifest_path": lse_manifest_path,
        "b2_closure_manifest_path": b2_closure_path,
        "accepted_lse_manifest": lse,
        "b2_closure_manifest": b2,
        "lse_checkpoint": checkpoint,
    }


def _depths(config: EarlyExitPreflightConfig) -> tuple[tuple[int, ...], int]:
    early = config.values.get("early_depths")
    full = config.values.get("full_depth")
    if not isinstance(early, list) or not all(isinstance(x, int) for x in early):
        _fail("B3_EARLY_EXIT_DEPTH_CONFIG_INVALID", "early_depths must be integer list")
    if full != 24:
        _fail("B3_EARLY_EXIT_DEPTH_CONFIG_INVALID", "full_depth must be 24")
    if tuple(early) != (12, 18):
        _fail("B3_EARLY_EXIT_DEPTH_CONFIG_INVALID", "early_depths must be [12, 18]")
    return tuple(early), int(full)


def run_early_exit_preflight(config: EarlyExitPreflightConfig) -> dict[str, Any]:
    chain = validate_accepted_b2_chain(config)
    early_depths, full_depth = _depths(config)
    prerequisites = {
        "lse_checkpoint": chain["lse_checkpoint"],
        "exit_target_manifest": _as_path(
            _required(
                config,
                "exit_target_manifest",
                code="B3_EARLY_EXIT_TARGET_MANIFEST_REQUIRED",
            ),
            repo_root=config.repo_root,
        ),
        "latency_profile": _as_path(
            _required(config, "latency_profile", code="B3_EARLY_EXIT_LATENCY_PROFILE_REQUIRED"),
            repo_root=config.repo_root,
        ),
        "calibration_trace": _as_path(
            _required(
                config,
                "calibration_trace",
                code="B3_EARLY_EXIT_CALIBRATION_TRACE_REQUIRED",
            ),
            repo_root=config.repo_root,
        ),
    }
    missing = [name for name, path in prerequisites.items() if not path.exists()]
    lse = chain["accepted_lse_manifest"]
    b2 = chain["b2_closure_manifest"]
    return {
        "schema_version": "b3_early_exit_accepted_lse_preflight_v1",
        "accepted_gate_passed": True,
        "ready": not missing,
        "missing_prerequisites": missing,
        "training_started": False,
        "evaluation_started": False,
        "final_content_accessed": False,
        "artifact_written": False,
        "early_depths": list(early_depths),
        "full_depth": full_depth,
        "accepted_dlcm_identity": lse["accepted_dlcm_identity"],
        "v5_deployment_identity": lse["v5_deployment_identity"],
        "accepted_lse_identity": lse["accepted_lse_identity"],
        "accepted_lse_checkpoint_sha256": lse["accepted_lse_checkpoint_sha256"],
        "b2_phase_final_closure_identity": b2["phase_final_closure_identity"],
        "lse_checkpoint": str(chain["lse_checkpoint"]),
    }
