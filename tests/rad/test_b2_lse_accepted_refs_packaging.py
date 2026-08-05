"""B2-06B accepted V5 reference packaging tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rad.phase_b import b2_lse_accepted_refs as refs

ACCEPTED = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
V5_DEPLOY = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
CAL_AB = "cae406c91ec392ffd7cc6d48ec2f0c94ab78d78f905cbfe904287842a7a7278a"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _accepted_manifest(path: Path, *, accepted_identity: str = ACCEPTED) -> Path:
    _write_json(
        path,
        {
            "schema_version": "b2_dlcm_v5_accepted_deployment_manifest_v1",
            "deployment_qualified": True,
            "accepted_identity": accepted_identity,
            "v5_deployment_identity": V5_DEPLOY,
            "beta_star_decimal": "0.54",
            "calibration_ab_identity": CAL_AB,
            "H_decision": "6fb60a82d01f987930070aeee75639524512ad481064369b2f06ac99f96ae0a8",
            "H_evidence": "bbc3708a8ddcd3b2965ec9e758af1a7bf30a360cdbbc5ff86be911cfbe872e02",
        },
    )
    return path


def _calibration_manifest(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": "b2_dlcm_v5_calibration_manifest_v1",
            "scientific_identity": CAL_AB,
            "selected": {"beta_decimal": "0.54"},
            "v5_contract_identity": {"architecture_contract_version": "b2_dlcm_architecture_v5"},
        },
    )
    return path


def test_package_checkpoint_into_accepted_refs_writes_receipt(tmp_path: Path) -> None:
    source = tmp_path / "c4b" / "canonical_deployment_candidate_v5.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint-bytes")
    accepted = _accepted_manifest(tmp_path / "final" / "accepted_deployment_manifest.json")
    calibration = _calibration_manifest(tmp_path / "c4b" / "calibration_a_manifest.json")

    receipt = refs.package_accepted_checkpoint_reference(
        accepted_manifest=accepted,
        source_checkpoint=source,
        source_calibration_manifest=calibration,
        expected_accepted_identity=ACCEPTED,
        expected_v5_deployment_identity=V5_DEPLOY,
        expected_calibration_ab_identity=CAL_AB,
    )

    packaged = accepted.parent / "accepted_refs" / "canonical_deployment_candidate_v5.pt"
    receipt_path = accepted.parent / "accepted_refs" / "b2_06b_accepted_reference_packaging_receipt.json"
    assert packaged.read_bytes() == b"checkpoint-bytes"
    assert receipt_path.is_file()
    assert receipt["accepted_identity"] == ACCEPTED
    assert receipt["v5_deployment_identity"] == V5_DEPLOY
    assert receipt["checkpoint_sha256"] == refs.sha256_file(source)
    assert receipt["accepted_identity_changed"] is False


def test_wrong_accepted_identity_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "c4b" / "canonical_deployment_candidate_v5.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint-bytes")
    accepted = _accepted_manifest(tmp_path / "final" / "accepted_deployment_manifest.json", accepted_identity="0" * 64)
    calibration = _calibration_manifest(tmp_path / "c4b" / "calibration_a_manifest.json")

    with pytest.raises(refs.B2AcceptedRefsError) as exc:
        refs.package_accepted_checkpoint_reference(
            accepted_manifest=accepted,
            source_checkpoint=source,
            source_calibration_manifest=calibration,
            expected_accepted_identity=ACCEPTED,
            expected_v5_deployment_identity=V5_DEPLOY,
            expected_calibration_ab_identity=CAL_AB,
        )

    assert exc.value.code == "B2_ACCEPTED_REFS_ACCEPTED_IDENTITY_MISMATCH"


def test_missing_source_checkpoint_fails_closed(tmp_path: Path) -> None:
    accepted = _accepted_manifest(tmp_path / "final" / "accepted_deployment_manifest.json")
    calibration = _calibration_manifest(tmp_path / "c4b" / "calibration_a_manifest.json")

    with pytest.raises(refs.B2AcceptedRefsError) as exc:
        refs.package_accepted_checkpoint_reference(
            accepted_manifest=accepted,
            source_checkpoint=tmp_path / "canonical_deployment_candidate_v5.pt",
            source_calibration_manifest=calibration,
            expected_accepted_identity=ACCEPTED,
            expected_v5_deployment_identity=V5_DEPLOY,
            expected_calibration_ab_identity=CAL_AB,
        )

    assert exc.value.code == "B2_ACCEPTED_REFS_SOURCE_CHECKPOINT_MISSING"
