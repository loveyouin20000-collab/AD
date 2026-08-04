"""Tests for B2-05B official enablement wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from rad.phase_b import b2_dlcm_official as subject
from tests.rad.b2_dlcm_fixtures import ACCEPTED_UPSTREAM

DESC_A = Path(
    "/root/autodl-tmp/AD-phase-b2-descriptor-real-extraction/"
    "artifacts/phase_b/b2_descriptor_artifacts/authoritative-run-a-20260729-013956"
)
CONTRIB_A = Path(
    "/root/autodl-tmp/AD-phase-b2-contribution-target-materialization/"
    "artifacts/phase_b/b2_contribution_targets/authoritative-run-a-20260804-030431"
)
DESC_B = Path(
    "/root/autodl-tmp/AD-phase-b2-descriptor-real-extraction/"
    "artifacts/phase_b/b2_descriptor_artifacts/authoritative-run-b-20260729-014404"
)
CONTRIB_B = Path(
    "/root/autodl-tmp/AD-phase-b2-contribution-target-materialization/"
    "artifacts/phase_b/b2_contribution_targets/authoritative-run-b-20260804-032200"
)
REPO = Path(__file__).resolve().parents[2]
OFFICIAL_CFG = REPO / "configs/phase_b/b2_dlcm_training_official_v1.json"


pytestmark = pytest.mark.skipif(
    not DESC_A.is_dir() or not CONTRIB_A.is_dir(),
    reason="accepted upstream runs missing",
)


def _load_official_config() -> dict:
    import json

    return json.loads(OFFICIAL_CFG.read_text(encoding="utf-8"))


def test_official_config_pins() -> None:
    cfg = _load_official_config()
    assert cfg["contract_stage"] == "b2_05b"
    assert cfg["real_training_enabled"] is True
    assert cfg["official_training_enabled"] is True
    assert cfg["expected_training_contract_tag"] == "b2-dlcm-training-contract-v1"
    assert cfg["expected_training_contract_commit"] == (
        "b580715f3dbfce3e4a03fe4073a57f99e8027f25"
    )
    assert cfg["expected_plan_sha_required"] is True
    assert cfg["repository_identity_gate_enabled"] is True


def test_plan_hash_stable_and_excludes_operational_fields() -> None:
    from rad.phase_b import b2_dlcm_training as training

    cfg = _load_official_config()
    verified = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=DESC_A / "final_manifest.json",
        descriptor_root=DESC_A,
        contribution_target_manifest=CONTRIB_A / "final_manifest.json",
        contribution_target_root=CONTRIB_A,
        accepted_upstream=ACCEPTED_UPSTREAM,
        evaluation_unlocked=False,
    )
    payload = subject.scientific_training_plan_payload(config=cfg, verified=verified)
    text = str(payload)
    for banned in (
        "output_root",
        "timestamp",
        "hostname",
        "gpu_uuid",
        "real_training_enabled",
        "repository_identity_gate_enabled",
    ):
        assert banned not in text
    sha1 = subject.compute_accepted_dlcm_training_plan_scientific_sha256(
        config=cfg, verified=verified
    )
    sha2 = subject.compute_accepted_dlcm_training_plan_scientific_sha256(
        config=cfg, verified=verified
    )
    assert sha1 == sha2
    assert len(sha1) == 64


def test_plan_sha_mismatch_combinations() -> None:
    cfg = _load_official_config()
    cfg = dict(cfg)
    cfg["expected_accepted_training_plan_sha256"] = "a" * 64
    with pytest.raises(subject.B2DLCMOfficialError) as exc:
        subject.require_plan_sha_agreement(
            config=cfg, recomputed="b" * 64, cli_expected="b" * 64
        )
    assert exc.value.code == "B2_DLCM_PLAN_SHA_MISMATCH"
    with pytest.raises(subject.B2DLCMOfficialError):
        subject.require_plan_sha_agreement(
            config=cfg, recomputed="a" * 64, cli_expected="c" * 64
        )


def test_official_dry_run_no_writes(tmp_path: Path) -> None:
    cfg = _load_official_config()
    out = tmp_path / "must_not_exist"
    result = subject.official_dry_run(
        config=cfg,
        descriptor_manifest=DESC_A / "final_manifest.json",
        descriptor_root=DESC_A,
        contribution_target_manifest=CONTRIB_A / "final_manifest.json",
        contribution_target_root=CONTRIB_A,
        output_root=out,
        seed=17,
        repo_root=None,
    )
    assert result["artifact_written"] is False
    assert result["run_directory_created"] is False
    assert result["real_training_started"] is False
    assert result["evaluation_records_loaded"] == 0
    assert result["training_records"] == 16
    assert not out.exists()
    assert len(result["accepted_dlcm_training_plan_scientific_sha256"]) == 64


@pytest.mark.skipif(not DESC_B.is_dir() or not CONTRIB_B.is_dir(), reason="run B missing")
def test_dry_run_a_b_plan_sha_equal() -> None:
    from rad.phase_b import b2_dlcm_training as training

    cfg = _load_official_config()
    va = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=DESC_A / "final_manifest.json",
        descriptor_root=DESC_A,
        contribution_target_manifest=CONTRIB_A / "final_manifest.json",
        contribution_target_root=CONTRIB_A,
        accepted_upstream=ACCEPTED_UPSTREAM,
        evaluation_unlocked=False,
    )
    vb = training.load_verified_b2_dlcm_training_inputs(
        descriptor_manifest=DESC_B / "final_manifest.json",
        descriptor_root=DESC_B,
        contribution_target_manifest=CONTRIB_B / "final_manifest.json",
        contribution_target_root=CONTRIB_B,
        accepted_upstream=ACCEPTED_UPSTREAM,
        evaluation_unlocked=False,
    )
    sha_a = subject.compute_accepted_dlcm_training_plan_scientific_sha256(config=cfg, verified=va)
    sha_b = subject.compute_accepted_dlcm_training_plan_scientific_sha256(config=cfg, verified=vb)
    assert sha_a == sha_b
