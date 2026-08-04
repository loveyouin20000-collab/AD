"""Hermetic fixtures for B2-05A DLCM training contract tests.

Test-only. No real MVTec/teacher/descriptor artifacts. Always marked
``artifact_kind == "test_fixture"``. Never imported by production CLIs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rad.phase_b import b2_dlcm as dlcm

FIXTURE_ARTIFACT_KIND = "test_fixture"
FIXTURE_CANDIDATE_LAYERS = (6, 12, 18, 24)
FIXTURE_PREDICTION_DEPTHS = (12, 18, 24)
FIXTURE_SPLIT_COUNTS = {"training": 16, "calibration": 8, "evaluation": 8}
FIXTURE_CATEGORIES = ("bottle", "cable", "capsule", "carpet")

# Accepted upstream identity pins from the B2-05A charter (bound by config).
ACCEPTED_UPSTREAM = {
    "accepted_input_contribution_plan_scientific_sha256": (
        "c3034f54b2e8cc99bffa31d5165ce595625263736d747b5f9db0b97072da7bb0"
    ),
    "gt_map_calibration_scientific_sha256": (
        "c2b6fdcac1c05fc1b879c35df8aaccf98484c4683e727e77a0101aa3c1d19ffc"
    ),
    "contribution_target_sample_coverage_sha256": (
        "df282f04205b6e182082c69be69ec585e62bb4d8ef7dbf742637f44a6b5843ed"
    ),
    "contribution_target_collection_scientific_sha256": (
        "69c9a678391a0e73d919e7cfe8f3a99a5e0916d465c14f2e82ff75b6ae3c9d9a"
    ),
    "shapley_normalization_scientific_sha256": (
        "21ed2d6f2fef0f62006d7817cc086180ac054640045ec6fcf5b58dac120cc522"
    ),
    "training_target_coverage_sha256": (
        "0bc032724b76d584f97719066fe5dce2ebb7b3023c76c92edebee2d14052c165"
    ),
    "calibration_target_coverage_sha256": (
        "9e2e7e6ed11428dddd9b83595a27b57f132cb548e63d21e2a1753d4f82e57f63"
    ),
    "evaluation_target_coverage_sha256": (
        "881e1887a2986c9a654fd6a690512807c88a8faa1828339eb7b680618817829d"
    ),
    "descriptor_collection_scientific_sha256": (
        "eb967822725e730ee2eb8afa3a5c8e28b4657141aa920d6a688ab370c70c6dd9"
    ),
    "descriptor_sample_coverage_sha256": (
        "27d064db21b5c699503be32e414d579bd1aa7158f1d9b141de26555fc79bc6df"
    ),
    "descriptor_normalization_scientific_sha256": (
        "f77975a94acf87a14b0753aabc9aad6777943ee4e4958b0a2083701cf4528594"
    ),
    "descriptor_normalization_training_coverage_sha256": (
        "e940f46bf696d326f8b982f15b8639f81e4548ec31a9b09634729811337e4c90"
    ),
}


def _stable_id(index: int) -> str:
    return f"fixture-{index:02d}"


def _split_for_index(index: int) -> str:
    if index < 16:
        return "training"
    if index < 24:
        return "calibration"
    return "evaluation"


@dataclass(frozen=True)
class HermeticDLCMRecord:
    stable_sample_id: str
    split: str
    category: str
    descriptors: Mapping[int, torch.Tensor]  # depth -> [n_d, 18] float32
    p_gt: Mapping[int, torch.Tensor]
    p_t: Mapping[int, torch.Tensor]
    phi_gt: Mapping[int, torch.Tensor]
    phi_t: Mapping[int, torch.Tensor]
    anomaly_maps: Mapping[int, torch.Tensor]  # depth -> [n_d, H, W]
    mask: torch.Tensor  # [H, W]


def _unit_simplex(n: int, tilt: float) -> torch.Tensor:
    raw = torch.arange(1, n + 1, dtype=torch.float64) + float(tilt)
    raw = raw / raw.sum()
    return raw.to(torch.float32)


def build_hermetic_dlcm_fixture(*, map_hw: tuple[int, int] = (8, 8)) -> list[HermeticDLCMRecord]:
    """32 aligned records with exact 16/8/8 split and valid players at every depth."""

    records: list[HermeticDLCMRecord] = []
    h, w = map_hw
    for index in range(32):
        split = _split_for_index(index)
        category = FIXTURE_CATEGORIES[index % len(FIXTURE_CATEGORIES)]
        descriptors: dict[int, torch.Tensor] = {}
        p_gt: dict[int, torch.Tensor] = {}
        p_t: dict[int, torch.Tensor] = {}
        phi_gt: dict[int, torch.Tensor] = {}
        phi_t: dict[int, torch.Tensor] = {}
        anomaly_maps: dict[int, torch.Tensor] = {}
        for depth in FIXTURE_PREDICTION_DEPTHS:
            players = dlcm.players_for_depth(FIXTURE_CANDIDATE_LAYERS, depth)
            n = len(players)
            # Deterministic standardized-looking descriptors in [-1, 1].
            base = torch.arange(n * 18, dtype=torch.float32)
            desc = ((base + index * 0.01 + depth * 0.001) % 17.0) / 8.5 - 1.0
            descriptors[depth] = desc.view(n, 18).contiguous()
            p_gt[depth] = _unit_simplex(n, tilt=0.1 * index + 0.01 * depth)
            p_t[depth] = _unit_simplex(n, tilt=0.07 * index + 0.02 * depth)
            # Signed targets with some spread for ranking.
            phi = torch.linspace(-1.0, 1.0, n, dtype=torch.float32) * (1.0 + 0.01 * index)
            phi_gt[depth] = phi
            phi_t[depth] = phi.flip(0) * 0.5
            maps = torch.zeros(n, h, w, dtype=torch.float32)
            for li in range(n):
                maps[li].fill_(0.1 * (li + 1) + 0.001 * index)
            anomaly_maps[depth] = maps
        mask = torch.zeros(h, w, dtype=torch.float32)
        if index % 2 == 1:
            mask[2:5, 2:5] = 1.0
        records.append(
            HermeticDLCMRecord(
                stable_sample_id=_stable_id(index),
                split=split,
                category=category,
                descriptors=descriptors,
                p_gt=p_gt,
                p_t=p_t,
                phi_gt=phi_gt,
                phi_t=phi_t,
                anomaly_maps=anomaly_maps,
                mask=mask,
            )
        )
    counts = {name: sum(1 for r in records if r.split == name) for name in FIXTURE_SPLIT_COUNTS}
    assert counts == dict(FIXTURE_SPLIT_COUNTS)
    return records


def fixture_normalization_artifact() -> dict[str, Any]:
    """Frozen float64 normalization stats matching B2-03B schema fields."""

    mean = [0.0] * 18
    std = [1.0] * 18
    return {
        "artifact_kind": FIXTURE_ARTIFACT_KIND,
        "normalization_contract_version": "b2_descriptor_normalization_v1",
        "feature_order": [f"f{i:02d}" for i in range(18)],
        "axis_order": "feature",
        "mean": mean,
        "std": std,
        "count": 16,
        "zero_variance": [False] * 18,
        "descriptor_normalization_scientific_sha256": ACCEPTED_UPSTREAM[
            "descriptor_normalization_scientific_sha256"
        ],
        "descriptor_normalization_training_coverage_sha256": ACCEPTED_UPSTREAM[
            "descriptor_normalization_training_coverage_sha256"
        ],
    }


def records_by_split(records: Sequence[HermeticDLCMRecord]) -> dict[str, list[HermeticDLCMRecord]]:
    out: dict[str, list[HermeticDLCMRecord]] = {"training": [], "calibration": [], "evaluation": []}
    for record in records:
        out[record.split].append(record)
    return out


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
