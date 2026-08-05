"""Hermetic C4C tests for V5 Final tooling closure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rad.phase_b import b2_dlcm_v5_final_evaluation as final_eval
from rad.phase_b import b2_dlcm_v5_final_loader as final_loader
from rad.phase_b import b2_dlcm_v5_final_manifests as final_manifests
from rad.phase_b import b2_dlcm_v5_final_materialization as final_mat
from rad.phase_b import b2_dlcm_v5_final_resolution as final_res
from rad.phase_b import b2_dlcm_v5_final_unlock as final_unlock
from rad.phase_b import b2_dlcm_v5_protocol as protocol

V5_DEPLOY = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
CAL_AB = "cae406c91ec392ffd7cc6d48ec2f0c94ab78d78f905cbfe904287842a7a7278a"
ROSTER = "267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213"
PLAN = "a" * 64
TOOLING_COMMIT = "d4cef9c1234567890abcdef1234567890abcdef"
TOOLING_TAG = "b2-dlcm-uniform-anchored-final-tooling-v1"
SOURCE = "335337d43dccea9e368393d2616972ad2217e490b91468d11a7e7f04fd688ee4"
DEV = "bedf478a173fd0e1f307cf1446edc638bad10a52c6c34f1ad0e72496da0159cd"
NORM = "f77975a94acf87a14b0753aabc9aad6777943ee4e4958b0a2083701cf4528594"


def _expected() -> dict[str, object]:
    return {
        "v5_deployment_identity": V5_DEPLOY,
        "beta_star_decimal": "0.54",
        "calibration_ab_identity": CAL_AB,
        "development_qualified_identity": DEV,
        "final_roster_identity": ROSTER,
        "source_master_manifest_identity": SOURCE,
        "normalization_identity": NORM,
        "tooling_commit": TOOLING_COMMIT,
        "tooling_tag": TOOLING_TAG,
        "accepted_v5_final_execution_plan_scientific_sha256": PLAN,
        "head_commit": TOOLING_COMMIT,
        "worktree_clean": True,
        "config_identity": "b" * 64,
    }


def _valid_unlock() -> dict[str, object]:
    return final_unlock.build_materialization_unlock(expected=_expected())


def _roster() -> dict[str, object]:
    rows = []
    for idx, (category, label) in enumerate(
        [
            ("bottle", "normal"),
            ("bottle", "anomalous"),
            ("carpet", "normal"),
            ("carpet", "anomalous"),
        ],
        start=1,
    ):
        sid = f"{idx:064x}"
        rows.append(
            {
                "stable_sample_id": sid,
                "category": category,
                "normal_or_anomalous": label,
                "source_record_scientific_sha256": protocol.canonical_json_sha256(
                    {"stable_sample_id": sid, "category": category, "label": label}
                ),
                "selection_rank": idx,
            }
        )
    return {
        "roster_scientific_sha256": ROSTER,
        "records": rows,
        "final_content_resolved": False,
        "paths_present": False,
    }


def _source_manifest(roster: dict[str, object]) -> dict[str, object]:
    rows = []
    for row in roster["records"]:  # type: ignore[index]
        record = dict(row)
        record.update(
            {
                "image_identity": f"mvtec/{record['stable_sample_id']}.png",
                "mask_identity": None
                if record["normal_or_anomalous"] == "normal"
                else f"mvtec/{record['stable_sample_id']}_mask.png",
                "image_label": 0 if record["normal_or_anomalous"] == "normal" else 1,
            }
        )
        rows.append(record)
    return {
        "source_master_manifest_scientific_sha256": SOURCE,
        "records": rows,
    }


def test_no_unlock_forbids_final_content_access() -> None:
    with pytest.raises(final_unlock.B2DLCMV5FinalUnlockError) as exc:
        final_unlock.validate_materialization_unlock(None, expected=_expected())
    assert exc.value.code == "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN"


def test_valid_unlock_authorizes_and_consumes_once() -> None:
    unlock = _valid_unlock()
    validated = final_unlock.validate_materialization_unlock(unlock, expected=_expected())
    assert validated["final_materialization_authorized"] is True
    consumed = final_unlock.consume_materialization_unlock(validated)
    assert consumed["consumed"] is True
    with pytest.raises(final_unlock.B2DLCMV5FinalUnlockError) as exc:
        final_unlock.validate_materialization_unlock(consumed, expected=_expected())
    assert exc.value.code == "B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_USED"


def test_dry_run_plan_ab_equal_and_no_output() -> None:
    plan_a = final_unlock.build_final_execution_plan(config=_expected(), repo_identity={"head": TOOLING_COMMIT})
    plan_b = final_unlock.build_final_execution_plan(config=_expected(), repo_identity={"head": TOOLING_COMMIT})
    sha_a = final_unlock.final_execution_plan_sha256(plan_a)
    sha_b = final_unlock.final_execution_plan_sha256(plan_b)
    assert sha_a == sha_b
    status = final_unlock.dry_run_status(plan_sha256=sha_a)
    assert status["real_final_content_accessed"] is False
    assert status["stable_ids_resolved"] is False
    assert status["materialization_started"] is False
    assert status["evaluation_started"] is False
    assert status["accepted_written"] is False
    assert status["run_directory_created"] is False
    assert status["artifact_written"] is False


def test_resolution_requires_authorization_and_exact_lookup() -> None:
    roster = _roster()
    source = _source_manifest(roster)
    with pytest.raises(final_res.B2DLCMV5FinalResolutionError) as exc:
        final_res.resolve_stable_ids(source, roster, authorized=False)
    assert exc.value.code == "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN"
    resolved = final_res.resolve_stable_ids(source, roster, authorized=True)
    assert resolved["stable_ids_resolved"] is True
    assert resolved["resolution_scientific_identity_included"] is False
    assert len(resolved["records"]) == 4
    missing = dict(source)
    missing["records"] = list(source["records"])[:-1]  # type: ignore[index]
    with pytest.raises(final_res.B2DLCMV5FinalResolutionError) as exc2:
        final_res.resolve_stable_ids(missing, roster, authorized=True)
    assert exc2.value.code == "B2_DLCM_FINAL_STABLE_ID_NOT_FOUND"


def test_materialization_ab_byte_equal_and_mismatch_detection() -> None:
    resolved = final_res.resolve_stable_ids(_source_manifest(_roster()), _roster(), authorized=True)
    unlock = _valid_unlock()
    mat_a = final_mat.run_hermetic_materialization("A", resolved["records"], unlock)
    mat_b = final_mat.run_hermetic_materialization("B", resolved["records"], unlock)
    comparison = final_mat.compare_materialization_ab(mat_a, mat_b)
    assert comparison["scientific_payload_equal"] is True
    assert comparison["canonical_scientific_files_byte_equal"] is True
    bad = dict(mat_b)
    bad["record_hashes"] = dict(mat_b["record_hashes"])
    bad["record_hashes"]["0" * 64] = "1" * 64
    with pytest.raises(final_mat.B2DLCMV5FinalMaterializationError) as exc:
        final_mat.compare_materialization_ab(mat_a, bad)
    assert exc.value.code == "B2_DLCM_FINAL_MATERIALIZATION_MISMATCH"


def test_evaluation_ab_decision_evidence_and_loader() -> None:
    resolved = final_res.resolve_stable_ids(_source_manifest(_roster()), _roster(), authorized=True)
    mat = final_mat.run_hermetic_materialization("A", resolved["records"], _valid_unlock())
    unlock = final_unlock.build_evaluation_unlock(expected=_expected(), materialization_identity=mat["collection_identity"])
    eval_a = final_eval.run_hermetic_final_evaluation("A", mat, unlock)
    eval_b = final_eval.run_hermetic_final_evaluation("B", mat, unlock)
    comparison = final_eval.compare_evaluation_ab(eval_a, eval_b)
    assert comparison["H_decision_equal"] is True
    assert comparison["H_evidence_equal"] is True
    with pytest.raises(final_loader.B2DLCMV5FinalLoaderError) as exc:
        final_loader.verify_accepted_v5_final_manifest(None, expected=_expected())
    assert exc.value.code == "B2_DLCM_NOT_ACCEPTED"
    accepted = final_manifests.build_accepted_deployment_manifest(
        final_decision_manifest=eval_a["final_decision_manifest"],
        final_evidence_manifest=eval_a["final_evidence_manifest"],
        expected=_expected(),
    )
    final_loader.verify_accepted_v5_final_manifest(accepted, expected=_expected())
    assert accepted["deployment_qualified"] is True


def test_old_fail_closed_evidence_is_preserved() -> None:
    p = Path("docs/phase_b/b2_05c4_uniform_anchored_manifest.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["status"] == "B2-05C4 stopped fail-closed"
