"""Hermetic tests for B2-05C4C Final execution plan binding repair."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v5_final_unlock as final_unlock

V5_DEPLOY = "c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd"
CAL_AB = "cae406c91ec392ffd7cc6d48ec2f0c94ab78d78f905cbfe904287842a7a7278a"
DEV = "bedf478a173fd0e1f307cf1446edc638bad10a52c6c34f1ad0e72496da0159cd"
ROSTER = "267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213"
SOURCE = "335337d43dccea9e368393d2616972ad2217e490b91468d11a7e7f04fd688ee4"
NORM = "f77975a94acf87a14b0753aabc9aad6777943ee4e4958b0a2083701cf4528594"
BASELINE = "8db5d85000000000000000000000000000000000"
V2_TAG = "b2-dlcm-uniform-anchored-final-tooling-v2"
DOCS_HEAD = "b847b1bd07fdd208d1f41bb793c0f8bbf033dddc"
PIN_HEAD = "1111111111111111111111111111111111111111"
NEXT_HEAD = "2222222222222222222222222222222222222222"


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": "b2_dlcm_v5_final_execution_official_v1",
        "contract_stage": "b2_05c4c",
        "v5_deployment_identity": V5_DEPLOY,
        "beta_star_decimal": "0.54",
        "calibration_ab_identity": CAL_AB,
        "development_qualified_identity": DEV,
        "final_roster_identity": ROSTER,
        "source_master_manifest_identity": SOURCE,
        "normalization_identity": NORM,
        "tooling_contract_schema": "b2_dlcm_v5_final_execution_contract_v1",
        "tooling_baseline_commit": BASELINE,
        "tooling_baseline_tag": V2_TAG,
        "expected_accepted_v5_final_execution_plan_sha256": None,
        "materialization_protocol": {
            "processes": ["A", "B"],
            "independent_process_required": True,
            "canonical_scientific_files_byte_equal_required": True,
            "partial_reuse_forbidden": True,
        },
        "evaluation_protocol": {
            "processes": ["A", "B"],
            "independent_process_required": True,
            "decision_evidence_hash_equal_required": True,
        },
        "final_gates": {
            "gt_macro_margin": 0.00001,
            "gt_per_category_slack": 0.0001,
            "delta_pixel_ap_macro_min": 0.0,
            "delta_pixel_auroc_macro_min": -0.0001,
            "delta_aupro_macro_min": -0.0001,
            "loc_per_category_floor": -0.001,
        },
    }
    config.update(overrides)
    return config


def _sha(config: dict[str, object], *, head: str) -> str:
    plan = final_unlock.build_final_execution_plan(config=config, repo_identity={"head": head})
    return final_unlock.final_execution_plan_sha256(plan)


def test_plan_sha_is_stable_across_docs_and_config_commits_after_tooling_baseline() -> None:
    baseline_config = _config()
    baseline_sha = _sha(baseline_config, head=BASELINE)
    docs_sha = _sha(baseline_config, head=DOCS_HEAD)
    pin_sha = _sha(baseline_config, head=PIN_HEAD)

    assert docs_sha == baseline_sha
    assert pin_sha == baseline_sha


def test_expected_plan_pin_is_not_part_of_scientific_identity() -> None:
    unpinned = _config()
    sha = _sha(unpinned, head=BASELINE)
    pinned = _config(expected_accepted_v5_final_execution_plan_sha256=sha)
    repinned = _config(expected_accepted_v5_final_execution_plan_sha256="f" * 64)

    assert _sha(pinned, head=PIN_HEAD) == sha
    assert _sha(repinned, head=NEXT_HEAD) == sha


def test_observed_execution_head_and_branch_are_excluded_from_plan_payload() -> None:
    plan = final_unlock.build_final_execution_plan(
        config=_config(),
        repo_identity={"head": NEXT_HEAD, "branch": "phase-b2-dlcm-v5-final-execution"},
    )

    assert "repo_head" not in plan
    assert "head_commit" not in plan
    assert "observed_branch" not in plan
    assert "tooling_baseline_commit" in plan
    assert plan["tooling_baseline_commit"] == BASELINE


def test_missing_or_invalid_tooling_baseline_fails_closed() -> None:
    no_baseline = _config(tooling_baseline_commit="")
    with pytest.raises(final_unlock.B2DLCMV5FinalUnlockError) as exc:
        final_unlock.build_final_execution_plan(config=no_baseline, repo_identity={"head": BASELINE})
    assert exc.value.code == "B2_DLCM_FINAL_TOOLING_BASELINE_INVALID"

    with pytest.raises(final_unlock.B2DLCMV5FinalUnlockError) as exc2:
        final_unlock.validate_repository_gate(
            repo_identity={
                "head": NEXT_HEAD,
                "tooling_baseline_commit": BASELINE,
                "tooling_baseline_tag": V2_TAG,
                "head_is_descendant_of_tooling_tag": True,
                "production_tooling_diff_since_tag_empty": False,
            },
        )
    assert exc2.value.code == "B2_DLCM_FINAL_TOOLING_BASELINE_DIRTY"
