from __future__ import annotations

import pytest

from rad.phase_b import b3_exit_training_contract as contract

ACCEPTED_DLCM = "0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116"
ACCEPTED_LSE = "3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa"
B2_CLOSURE = "2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d"
MATERIALIZATION = "d77c2cd604c7d3eca4cc9f0649bc8e6ef1b84985a31ec470dd6dd7ef1f43e5b8"


def _manifest(target_exits: dict[str, int] | None = None) -> dict[str, object]:
    return {
        "schema_version": "b3_02_exit_prerequisite_materialization_manifest_v1",
        "materialization_identity": MATERIALIZATION,
        "accepted_dlcm_identity": ACCEPTED_DLCM,
        "accepted_lse_identity": ACCEPTED_LSE,
        "b2_phase_final_closure_identity": B2_CLOSURE,
        "records": 16,
        "counts_by_depth": {"12": 8, "18": 8},
        "target_exits_by_depth": target_exits or {"12": 0, "18": 0},
        "training_started": False,
        "evaluation_started": False,
        "final_content_accessed": False,
        "checkpoint_generated": False,
    }


def test_no_positive_exit_targets_do_not_unlock_training() -> None:
    decision = contract.build_exit_training_contract(
        prerequisite_manifest=_manifest(),
        git_sha="abc123",
        tracked_pt_count=0,
    )

    assert decision["schema_version"] == "b3_03_exit_policy_training_contract_v1"
    assert decision["decision"] == "conservative_full_depth_fallback"
    assert decision["training_unlocked"] is False
    assert decision["training_started"] is False
    assert decision["checkpoint_generated"] is False
    assert decision["fallback_depth"] == 24
    assert decision["reason"] == "no_positive_exit_targets"


def test_positive_exit_targets_require_future_training_unlock() -> None:
    decision = contract.build_exit_training_contract(
        prerequisite_manifest=_manifest({"12": 1, "18": 0}),
        git_sha="abc123",
        tracked_pt_count=0,
    )

    assert decision["decision"] == "training_contract_ready_pending_unlock"
    assert decision["training_unlocked"] is False
    assert decision["training_started"] is False


def test_tracked_pt_files_fail_closed() -> None:
    with pytest.raises(contract.B3ExitTrainingContractError) as exc:
        contract.build_exit_training_contract(
            prerequisite_manifest=_manifest(),
            git_sha="abc123",
            tracked_pt_count=1,
        )

    assert exc.value.code == "B3_EXIT_TRAINING_CONTRACT_TRACKED_PT"


def test_prior_training_boundary_violation_fails_closed() -> None:
    manifest = _manifest()
    manifest["training_started"] = True

    with pytest.raises(contract.B3ExitTrainingContractError) as exc:
        contract.build_exit_training_contract(
            prerequisite_manifest=manifest,
            git_sha="abc123",
            tracked_pt_count=0,
        )

    assert exc.value.code == "B3_EXIT_TRAINING_CONTRACT_BOUNDARY_VIOLATION"
