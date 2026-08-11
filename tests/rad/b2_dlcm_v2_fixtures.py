"""Hermetic fixtures for B2-05C1 V2 DLCM contract tests.

Test-only. Never imported by production CLIs. Always ``artifact_kind == test_fixture``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rad.phase_b import b2_dlcm_v2 as v2
from tests.rad import b2_dlcm_fixtures as v1_fix

FIXTURE_ARTIFACT_KIND = "test_fixture"
ACCEPTED_UPSTREAM = dict(v1_fix.ACCEPTED_UPSTREAM)

V2_CONTRACT_TEMPLATE: dict[str, Any] = {
    "schema_version": "b2_dlcm_decoupled_training_contract_v2",
    "contract_stage": "b2_05c1a",
    "real_training_enabled": False,
    "authoritative_base_tag": "b2-dlcm-unqualified-evidence-v1",
    "authoritative_base_commit": v2.V1_EVIDENCE_COMMIT,
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
    payload = dict(V2_CONTRACT_TEMPLATE)
    payload["accepted_upstream"] = dict(ACCEPTED_UPSTREAM)
    payload.update(overrides)
    payload["artifact_kind"] = FIXTURE_ARTIFACT_KIND
    return payload


def hermetic_identity_candidates(*, per_group: int = 6) -> list[dict[str, Any]]:
    """Identity-only candidates for roster tests (no paths)."""

    rows: list[dict[str, Any]] = []
    for cat in ("bottle", "carpet"):
        for label in ("normal", "anomalous"):
            for idx in range(per_group):
                sid = f"{cat}-{label}-{idx:02d}-{'a' * 32}"
                rows.append(
                    {
                        "stable_sample_id": sid,
                        "category": cat,
                        "normal_or_anomalous": label,
                        "source_record_scientific_sha256": "b" * 64,
                    }
                )
    return rows


def exclusion_ids_from_candidates(
    candidates: list[Mapping[str, Any]],
    *,
    take: int = 32,
) -> list[str]:
    return [str(row["stable_sample_id"]) for row in candidates[:take]]
