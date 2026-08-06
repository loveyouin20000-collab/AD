from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rad.phase_b import b3_early_exit_prerequisites as prereq

ACCEPTED_DLCM = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
ACCEPTED_LSE = "3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa"
B2_CLOSURE = "2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d"
LSE_CKPT_SHA = "e6e5a4dbd7471ef9e52430eab9533f8edda57ca76ead2ffbed034044805b1c98"
V5_DEPLOY = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sample_id": "b",
            "depth": 18,
            "target_gain": 0.2,
            "target_sufficient": 1.0,
            "pred_mean": 0.3,
            "pred_suf_prob": 0.7,
        },
        {
            "sample_id": "a",
            "depth": 12,
            "target_gain": 0.5,
            "target_sufficient": 0.0,
            "pred_mean": 0.4,
            "pred_suf_prob": 0.2,
        },
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _accepted_lse_manifest(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": "b2_06f_lse_accepted_artifact_manifest_v1",
        "accepted_artifact_generated": True,
        "lse_qualified": True,
        "accepted_dlcm_identity": ACCEPTED_DLCM,
        "v5_deployment_identity": V5_DEPLOY,
        "accepted_lse_identity": ACCEPTED_LSE,
        "accepted_lse_checkpoint": str(tmp_path / "accepted_refs" / "lse_best.pt"),
        "accepted_lse_checkpoint_sha256": LSE_CKPT_SHA,
        "training_started": False,
        "evaluation_started": False,
    }


def _b2_closure() -> dict[str, object]:
    return {
        "schema_version": "b2_07_phase_final_closure_manifest_v1",
        "phase_final_closure_identity": B2_CLOSURE,
        "accepted_dlcm_identity": ACCEPTED_DLCM,
        "accepted_lse_identity": ACCEPTED_LSE,
        "v5_deployment_identity": V5_DEPLOY,
        "accepted_lse_checkpoint_sha256": LSE_CKPT_SHA,
        "training_started_in_b2_07": False,
        "evaluation_started_in_b2_07": False,
        "final_content_accessed_in_b2_07": False,
        "tracked_pt_count": 0,
    }


def _config(tmp_path: Path) -> Path:
    cfg = {
        "early_exit": {
            "accepted_lse_manifest": str(tmp_path / "accepted_lse_manifest.json"),
            "b2_phase_final_closure_manifest": str(tmp_path / "b2_closure.json"),
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
        }
    }
    path = tmp_path / "early_exit.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    config = _config(tmp_path)
    _write_json(tmp_path / "accepted_lse_manifest.json", _accepted_lse_manifest(tmp_path))
    _write_json(tmp_path / "b2_closure.json", _b2_closure())
    (tmp_path / "accepted_refs").mkdir()
    (tmp_path / "accepted_refs" / "lse_best.pt").write_bytes(b"checkpoint")
    predictions = tmp_path / "cal_predictions.jsonl"
    _write_jsonl(predictions)
    return config, predictions


def test_materialize_exit_prerequisites_writes_three_bound_outputs(tmp_path: Path) -> None:
    config, predictions = _inputs(tmp_path)

    manifest = prereq.materialize_exit_prerequisites(
        config_path=config,
        calibration_predictions_path=predictions,
        repo_root=tmp_path,
    )

    assert manifest["schema_version"] == "b3_02_exit_prerequisite_materialization_manifest_v1"
    assert manifest["accepted_lse_identity"] == ACCEPTED_LSE
    assert manifest["b2_phase_final_closure_identity"] == B2_CLOSURE
    assert manifest["records"] == 2
    assert manifest["training_started"] is False
    assert manifest["evaluation_started"] is False
    assert Path(manifest["exit_target_manifest"]).is_file()
    assert Path(manifest["latency_profile"]).is_file()
    assert Path(manifest["calibration_trace"]).is_file()


def test_materialize_exit_prerequisites_fails_closed_when_predictions_missing(tmp_path: Path) -> None:
    config, predictions = _inputs(tmp_path)
    predictions.unlink()

    with pytest.raises(prereq.B3ExitPrerequisiteError) as exc:
        prereq.materialize_exit_prerequisites(
            config_path=config,
            calibration_predictions_path=predictions,
            repo_root=tmp_path,
        )

    assert exc.value.code == "B3_EXIT_PREREQ_CALIBRATION_PREDICTIONS_REQUIRED"
