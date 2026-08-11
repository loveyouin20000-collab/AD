"""Hermetic fixtures for B2-05C2 V3 DLCM contract tests."""

from __future__ import annotations

from typing import Any

from rad.phase_b import b2_dlcm_v3 as v3
from tests.rad import b2_dlcm_fixtures as v1_fix

FIXTURE_ARTIFACT_KIND = "test_fixture"
ACCEPTED_UPSTREAM = dict(v1_fix.ACCEPTED_UPSTREAM)

V3_CONTRACT_TEMPLATE: dict[str, Any] = {
    "schema_version": "b2_dlcm_category_robust_contract_v3",
    "contract_stage": "b2_05c2a",
    "real_training_enabled": False,
    "smoothmax_tau": 0.05,
    "sampler_contract_version": "b2_dlcm_category_balanced_sampler_v1",
    "training_categories": ["bottle", "carpet"],
    "per_category_per_batch": 2,
    "authoritative_v2_contract_tag": "b2-dlcm-decoupled-contract-v2",
    "authoritative_v2_contract_commit": v3.V2_CONTRACT_COMMIT,
    "authoritative_base_tag": "b2-dlcm-unqualified-evidence-v1",
    "authoritative_base_commit": "43d856f5ff771957f9f39d0909b1bc87d6b7081b",
    "adopted_final_roster_scientific_sha256": v3.ADOPTED_ROSTER_SCIENTIFIC,
    "accepted_upstream": ACCEPTED_UPSTREAM,
    "candidate_layers": [6, 12, 18, 24],
    "prediction_depths": [12, 18, 24],
    "descriptor_dimension": 18,
    "layer_embedding_dimension": 8,
    "depth_embedding_dimension": 8,
    "hidden_dimension": 64,
    "dropout_probability": 0.1,
    "teacher_allocation_loss_weight": 0.25,
    "gt_signed_loss_weight": 0.25,
    "teacher_signed_loss_weight": 0.0625,
    "huber_delta": 1.0,
    "ranking_weight": 0.25,
    "ranking_tie_tolerance": 1e-6,
    "depth_weights": {"12": 1 / 3, "18": 1 / 3, "24": 1 / 3},
    "seeds": [17, 29, 43],
}


def contract_config(**overrides: Any) -> dict[str, Any]:
    payload = dict(V3_CONTRACT_TEMPLATE)
    payload["accepted_upstream"] = dict(ACCEPTED_UPSTREAM)
    payload.update(overrides)
    payload["artifact_kind"] = FIXTURE_ARTIFACT_KIND
    return payload
