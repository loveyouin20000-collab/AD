"""B2-06A LSE accepted V5 preflight gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rad.phase_b import b2_lse_accepted_gate as gate

V5_DEPLOY = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
CAL_AB = "cae406c91ec392ffd7cc6d48ec2f0c94ab78d78f905cbfe904287842a7a7278a"
H_DECISION = "6fb60a82d01f987930070aeee75639524512ad481064369b2f06ac99f96ae0a8"
H_EVIDENCE = "bbc3708a8ddcd3b2965ec9e758af1a7bf30a360cdbbc5ff86be911cfbe872e02"
ACCEPTED = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"


def _accepted() -> dict[str, object]:
    return {
        "schema_version": "b2_dlcm_v5_accepted_deployment_manifest_v1",
        "deployment_qualified": True,
        "v5_deployment_identity": V5_DEPLOY,
        "beta_star_decimal": "0.54",
        "calibration_ab_identity": CAL_AB,
        "H_decision": H_DECISION,
        "H_evidence": H_EVIDENCE,
        "accepted_identity": ACCEPTED,
    }


def _decision() -> dict[str, object]:
    return {
        "schema_version": "b2_dlcm_v5_final_decision_manifest_v1",
        "H_decision": H_DECISION,
        "verdict": "qualified",
        "gt_target_learning": {
            "depth_24_macro_kl": 0.01,
            "depth_24_uniform_macro_kl": 0.02,
            "per_category_kl": {"bottle": 0.01, "carpet": 0.01},
        },
        "localization": {
            "delta_pixel_ap_macro": 0.001,
            "delta_pixel_auroc_macro": 0.0,
            "delta_aupro_macro": 0.0,
            "per_category": {
                "bottle": {"delta_pixel_ap": 0.001},
                "carpet": {"delta_pixel_ap": 0.001},
            },
        },
        "thresholds": {"gt_macro_margin": 1e-5},
    }


def _evidence() -> dict[str, object]:
    return {
        "schema_version": "b2_dlcm_v5_final_evidence_manifest_v1",
        "H_decision": H_DECISION,
        "H_evidence": H_EVIDENCE,
        "calibration_ab_identity": CAL_AB,
        "development_qualified_identity": "bedf478a173fd0e1f307cf1446edc638bad10a52c6c34f1ad0e72496da0159cd",
        "final_roster_identity": "267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213",
        "tooling_contract_schema": "b2_dlcm_v5_final_execution_contract_v1",
        "tooling_baseline_commit": "bb246ed156a8141344f793cfaeed35d8b4bc16c3",
        "tooling_baseline_tag": "b2-dlcm-uniform-anchored-final-tooling-v2",
        "materialization_ab": {"materialization_ab_sha256": "f" * 64},
        "evaluation_ab": {"evaluation_ab_sha256": "e" * 64},
        "production_metric_proof": {"source": "hermetic_fixture", "invoked": True},
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config(tmp_path: Path, **overrides: object) -> Path:
    lse: dict[str, object] = {
        "accepted_manifest": str(tmp_path / "accepted_deployment_manifest.json"),
            "final_decision_manifest": str(tmp_path / "final_decision_manifest.json"),
            "final_evidence_manifest": str(tmp_path / "final_evidence_manifest.json"),
            "expected_accepted_identity": ACCEPTED,
            "expected_v5_deployment_identity": V5_DEPLOY,
            "expected_H_decision": H_DECISION,
            "expected_H_evidence": H_EVIDENCE,
            "dlcm_checkpoint": str(tmp_path / "accepted_refs" / "canonical_deployment_candidate_v5.pt"),
            "train_gain_targets": str(tmp_path / "gain" / "mvtec_train.pt"),
            "calibration_gain_targets": str(tmp_path / "gain" / "mvtec_calibration.pt"),
            "train_cache": str(tmp_path / "cache" / "train"),
            "calibration_cache": str(tmp_path / "cache" / "calibration"),
            "descriptor_stats": str(tmp_path / "stats" / "mvtec_seed111.json"),
    }
    lse.update(overrides)
    payload: dict[str, object] = {"lse": lse}
    path = tmp_path / "lse_b2.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_manifests(tmp_path: Path) -> None:
    _write_json(tmp_path / "accepted_deployment_manifest.json", _accepted())
    _write_json(tmp_path / "final_decision_manifest.json", _decision())
    _write_json(tmp_path / "final_evidence_manifest.json", _evidence())


def test_missing_accepted_manifest_fails_closed(tmp_path: Path) -> None:
    cfg = gate.load_lse_preflight_config(_config(tmp_path))
    with pytest.raises(gate.B2LSEAcceptedGateError) as exc:
        gate.run_lse_preflight(cfg)
    assert exc.value.code == "B2_LSE_ACCEPTED_MANIFEST_REQUIRED"


def test_wrong_accepted_identity_fails_closed(tmp_path: Path) -> None:
    _write_manifests(tmp_path)
    bad = _accepted()
    bad["accepted_identity"] = "0" * 64
    _write_json(tmp_path / "accepted_deployment_manifest.json", bad)
    cfg = gate.load_lse_preflight_config(_config(tmp_path))
    with pytest.raises(gate.B2LSEAcceptedGateError) as exc:
        gate.run_lse_preflight(cfg)
    assert exc.value.code == "B2_LSE_ACCEPTED_IDENTITY_MISMATCH"


def test_manual_c4b_checkpoint_path_without_accepted_binding_fails_closed(tmp_path: Path) -> None:
    _write_manifests(tmp_path)
    manual = "/root/autodl-tmp/AD-phase-b2-dlcm-uniform-anchored-calibration/artifacts/phase_b/b2_dlcm_uniform_anchored_calibration/authoritative-run-20260805-081937/canonical_deployment_candidate_v5.pt"
    cfg = gate.load_lse_preflight_config(_config(tmp_path, dlcm_checkpoint=manual))
    with pytest.raises(gate.B2LSEAcceptedGateError) as exc:
        gate.run_lse_preflight(cfg)
    assert exc.value.code == "B2_LSE_CHECKPOINT_NOT_ACCEPTED_BOUND"


def test_valid_accepted_manifest_reports_missing_prerequisites_without_training(tmp_path: Path) -> None:
    _write_manifests(tmp_path)
    cfg = gate.load_lse_preflight_config(_config(tmp_path))
    report = gate.run_lse_preflight(cfg)
    assert report["accepted_gate_passed"] is True
    assert report["training_started"] is False
    assert report["ready"] is False
    assert set(report["missing_prerequisites"]) == {
        "dlcm_checkpoint",
        "train_gain_targets",
        "calibration_gain_targets",
        "train_cache",
        "calibration_cache",
        "descriptor_stats",
    }
