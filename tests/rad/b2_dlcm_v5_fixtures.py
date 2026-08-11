"""Hermetic fixtures for B2-05C4 V5 uniform-anchored calibration tests."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from rad.phase_b import b2_dlcm_v5 as v5
from tests.rad import b2_dlcm_fixtures as v1_fix

FIXTURE_ARTIFACT_KIND = "test_fixture"
ACCEPTED_UPSTREAM = dict(v1_fix.ACCEPTED_UPSTREAM)

V5_CONTRACT_TEMPLATE: dict[str, Any] = {
    "schema_version": "b2_dlcm_uniform_anchored_contract_v5",
    "contract_stage": "b2_05c4a",
    "real_training_enabled": False,
    "calibration_enabled": False,
    "development_enabled": False,
    "final_materialization_enabled": False,
    "final_evaluation_enabled": False,
    "beta_grid_size": 101,
    "loo_depth": 24,
    "canonical_seed": 17,
    "adopted_final_roster_scientific_sha256": v5.ADOPTED_ROSTER_SCIENTIFIC,
    "authoritative_v4_unqualified_tag": v5.V4_UNQUALIFIED_TAG,
    "authoritative_v4_unqualified_commit": v5.V4_UNQUALIFIED_COMMIT,
    "accepted_upstream": ACCEPTED_UPSTREAM,
    "training_categories": ["bottle", "carpet"],
    "candidate_layers": [6, 12, 18, 24],
    "prediction_depths": [12, 18, 24],
    "descriptor_dimension": 18,
}


def contract_config(**overrides: Any) -> dict[str, Any]:
    payload = dict(V5_CONTRACT_TEMPLATE)
    payload["accepted_upstream"] = dict(ACCEPTED_UPSTREAM)
    payload.update(overrides)
    payload["artifact_kind"] = FIXTURE_ARTIFACT_KIND
    return payload


def _simplex_from_logits(logits: list[float]) -> torch.Tensor:
    t = torch.tensor(logits, dtype=torch.float32)
    return F.softmax(t, dim=0)


def make_calibration_records(*, n_players: int = 4) -> list[dict[str, Any]]:
    """4 bottle + 4 carpet hermetic Calibration records at depth 24."""

    # Construct targets/dynamic weights so some intermediate betas are eligible
    # and LOO/selection are deterministic.
    bottle_specs = [
        ([2.0, 0.5, 0.1, 0.1], [2.2, 0.4, 0.05, 0.05]),
        ([1.8, 0.6, 0.2, 0.1], [2.0, 0.5, 0.1, 0.05]),
        ([1.5, 0.8, 0.3, 0.2], [1.7, 0.7, 0.2, 0.1]),
        ([1.2, 1.0, 0.4, 0.2], [1.4, 0.9, 0.3, 0.1]),
    ]
    carpet_specs = [
        ([0.4, 0.4, 1.5, 0.8], [0.35, 0.35, 1.7, 0.9]),
        ([0.5, 0.3, 1.4, 0.9], [0.4, 0.3, 1.6, 1.0]),
        ([0.6, 0.4, 1.2, 1.0], [0.5, 0.35, 1.4, 1.1]),
        ([0.5, 0.5, 1.1, 1.1], [0.45, 0.45, 1.25, 1.2]),
    ]
    records: list[dict[str, Any]] = []
    for i, (p_logits, w_logits) in enumerate(bottle_specs):
        p = _simplex_from_logits(p_logits[:n_players])
        w = _simplex_from_logits(w_logits[:n_players])
        records.append(
            {
                "stable_sample_id": f"bottle_cal_{i:02d}",
                "category": "bottle",
                "depth": 24,
                "p_gt": p,
                "dynamic_weights": w,
            }
        )
    for i, (p_logits, w_logits) in enumerate(carpet_specs):
        p = _simplex_from_logits(p_logits[:n_players])
        w = _simplex_from_logits(w_logits[:n_players])
        records.append(
            {
                "stable_sample_id": f"carpet_cal_{i:02d}",
                "category": "carpet",
                "depth": 24,
                "p_gt": p,
                "dynamic_weights": w,
            }
        )
    return records


def make_ineligible_records(*, n_players: int = 3) -> list[dict[str, Any]]:
    """Records where dynamic weights are far from GT so only maybe beta~0 could pass;
    constructed so NO beta is eligible (uniform itself fails macro margin)."""

    # Use identical p and a very peaked wrong dynamic; uniform will still lose macro
    # margin requirement (KL(β) <= KL_uni - 1e-5) for all β including 0, because
    # at beta=0, KL(β)=KL_uni so macro inequality fails strictly.
    records: list[dict[str, Any]] = []
    for cat in ("bottle", "carpet"):
        for i in range(4):
            p = _simplex_from_logits([3.0, 0.1, 0.1][:n_players])
            w = _simplex_from_logits([0.1, 3.0, 0.1][:n_players])
            records.append(
                {
                    "stable_sample_id": f"{cat}_bad_{i:02d}",
                    "category": cat,
                    "depth": 24,
                    "p_gt": p,
                    "dynamic_weights": w,
                }
            )
    return records
