"""Helpers for CPU CI policy and bootstrap fixtures in tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BALANCED_PROFILE: dict[str, Any] = {
    "name": "balanced",
    "gain_threshold": 0.02,
    "kappa": 0.5,
    "map_uncertainty_threshold": 0.5,
    "image_confidence_margin": 0.4,
    "stability_threshold": 1.0,
    "require_map_uncertainty": True,
    "require_image_confidence": True,
    "require_stability": False,
}

_SPLIT_ROWS: list[dict[str, Any]] = [
    {
        "sample_id": "bottle/train/good/000.png",
        "image_path": "bottle/train/good/000.png",
        "mask_path": "",
        "category": "bottle",
        "label": 0,
        "split": "train",
    },
    {
        "sample_id": "bottle/train/good/001.png",
        "image_path": "bottle/train/good/001.png",
        "mask_path": "",
        "category": "bottle",
        "label": 0,
        "split": "calibration",
    },
]

_DESCRIPTOR_MEDIAN = [0.0] * 18
_DESCRIPTOR_IQR = [1.0] * 18


def minimal_policy_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "test_fixture",
        "eligible_for_evaluation": False,
        "profiles": {"balanced": dict(BALANCED_PROFILE)},
    }


def write_minimal_policy_fixture(path: Path) -> None:
    """Write a valid frozen-policy JSON for dry-run CLI tests."""
    path.write_text(json.dumps(minimal_policy_payload(), indent=2) + "\n", encoding="utf-8")


def write_split_manifest_jsonl(path: Path) -> None:
    """Write a schema-valid split manifest without test_fixture markers."""
    lines = [json.dumps(row, sort_keys=True) for row in _SPLIT_ROWS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_descriptor_stats_json(path: Path) -> None:
    """Write a schema-valid descriptor-stats JSON without test_fixture markers."""
    payload = {
        "schema_version": 1,
        "clamp": [-8.0, 8.0],
        "eps": 1e-6,
        "median": list(_DESCRIPTOR_MEDIAN),
        "iqr": list(_DESCRIPTOR_IQR),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
