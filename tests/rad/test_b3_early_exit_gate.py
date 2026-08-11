from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rad.phase_b import b3_early_exit_gate as gate

ACCEPTED_DLCM = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
V5_DEPLOY = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
ACCEPTED_LSE = "3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa"
B2_CLOSURE = "2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d"
LSE_CKPT = "e6e5a4dbd7471ef9e52430eab9533f8edda57ca76ead2ffbed034044805b1c98"


def _lse_manifest(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": "b2_06f_lse_accepted_artifact_manifest_v1",
        "accepted_artifact_generated": True,
        "lse_qualified": True,
        "accepted_dlcm_identity": ACCEPTED_DLCM,
        "v5_deployment_identity": V5_DEPLOY,
        "accepted_lse_identity": ACCEPTED_LSE,
        "accepted_lse_checkpoint": str(tmp_path / "accepted_refs" / "lse_best.pt"),
        "accepted_lse_checkpoint_sha256": LSE_CKPT,
        "training_started": False,
        "evaluation_started": False,
    }


def _b2_closure() -> dict[str, object]:
    return {
        "schema_version": "b2_07_phase_final_closure_manifest_v1",
        "status": "b2_phase_completed_locally",
        "phase_final_closure_identity": B2_CLOSURE,
        "accepted_dlcm_identity": ACCEPTED_DLCM,
        "v5_deployment_identity": V5_DEPLOY,
        "accepted_lse_identity": ACCEPTED_LSE,
        "accepted_lse_checkpoint_sha256": LSE_CKPT,
        "training_started_in_b2_07": False,
        "evaluation_started_in_b2_07": False,
        "final_content_accessed_in_b2_07": False,
        "tracked_pt_count": 0,
        "pushed": False,
        "pr_opened": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifests(tmp_path: Path) -> None:
    _write_json(tmp_path / "accepted_lse_manifest.json", _lse_manifest(tmp_path))
    _write_json(tmp_path / "b2_07_phase_final_closure_manifest.json", _b2_closure())


def _config(tmp_path: Path, **overrides: object) -> Path:
    early_exit: dict[str, object] = {
        "accepted_lse_manifest": str(tmp_path / "accepted_lse_manifest.json"),
        "b2_phase_final_closure_manifest": str(tmp_path / "b2_07_phase_final_closure_manifest.json"),
        "expected_accepted_dlcm_identity": ACCEPTED_DLCM,
        "expected_v5_deployment_identity": V5_DEPLOY,
        "expected_accepted_lse_identity": ACCEPTED_LSE,
        "expected_b2_phase_final_closure_identity": B2_CLOSURE,
        "accepted_lse_reference_root": str(tmp_path / "accepted_refs"),
        "lse_checkpoint": str(tmp_path / "accepted_refs" / "lse_best.pt"),
        "early_depths": [12, 18],
        "full_depth": 24,
        "exit_target_manifest": str(tmp_path / "targets" / "exit_targets.json"),
        "latency_profile": str(tmp_path / "profiles" / "latency.json"),
        "calibration_trace": str(tmp_path / "traces" / "calibration.jsonl"),
        "output_dir": str(tmp_path / "artifacts" / "early_exit"),
    }
    early_exit.update(overrides)
    path = tmp_path / "early_exit.yaml"
    path.write_text(yaml.safe_dump({"early_exit": early_exit}), encoding="utf-8")
    return path


def test_missing_accepted_lse_manifest_fails_closed(tmp_path: Path) -> None:
    cfg = gate.load_early_exit_preflight_config(_config(tmp_path))
    with pytest.raises(gate.B3EarlyExitGateError) as exc:
        gate.run_early_exit_preflight(cfg)
    assert exc.value.code == "B3_EARLY_EXIT_ACCEPTED_LSE_MANIFEST_REQUIRED"


def test_wrong_accepted_lse_identity_fails_closed(tmp_path: Path) -> None:
    _write_manifests(tmp_path)
    manifest = _lse_manifest(tmp_path)
    manifest["accepted_lse_identity"] = "0" * 64
    _write_json(tmp_path / "accepted_lse_manifest.json", manifest)
    cfg = gate.load_early_exit_preflight_config(_config(tmp_path))
    with pytest.raises(gate.B3EarlyExitGateError) as exc:
        gate.run_early_exit_preflight(cfg)
    assert exc.value.code == "B3_EARLY_EXIT_ACCEPTED_LSE_IDENTITY_MISMATCH"


def test_manual_lse_checkpoint_without_accepted_binding_fails_closed(tmp_path: Path) -> None:
    _write_manifests(tmp_path)
    cfg = gate.load_early_exit_preflight_config(
        _config(tmp_path, lse_checkpoint="/tmp/manual/lse_best.pt")
    )
    with pytest.raises(gate.B3EarlyExitGateError) as exc:
        gate.run_early_exit_preflight(cfg)
    assert exc.value.code == "B3_EARLY_EXIT_LSE_CHECKPOINT_NOT_ACCEPTED_BOUND"


def test_valid_chain_reports_missing_prerequisites_without_training(tmp_path: Path) -> None:
    _write_manifests(tmp_path)
    cfg = gate.load_early_exit_preflight_config(_config(tmp_path))
    report = gate.run_early_exit_preflight(cfg)
    assert report["accepted_gate_passed"] is True
    assert report["training_started"] is False
    assert report["evaluation_started"] is False
    assert report["ready"] is False
    assert set(report["missing_prerequisites"]) == {
        "lse_checkpoint",
        "exit_target_manifest",
        "latency_profile",
        "calibration_trace",
    }
