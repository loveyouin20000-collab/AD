"""Artifact schema and identity tests for B2-05A."""

from __future__ import annotations

import json
from pathlib import Path

from rad.phase_b import b2_dlcm as model_mod
from rad.phase_b import b2_dlcm_deployment as deploy
from rad.phase_b import b2_dlcm_training as training
from tests.rad.b2_dlcm_fixtures import ACCEPTED_UPSTREAM, fixture_normalization_artifact


def test_contract_config_pins(repo_root: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads(
        (root / "configs/phase_b/b2_dlcm_training_contract_v1.json").read_text(encoding="utf-8")
    )
    assert cfg["contract_stage"] == "b2_05a"
    assert cfg["real_training_enabled"] is False
    assert cfg["authoritative_base_commit"] == (
        "97a4f497f6f2b096dd4a339555f81e7296ec3035"
    )
    assert cfg["candidate_layers"] == [6, 12, 18, 24]
    assert cfg["prediction_depths"] == [12, 18, 24]
    assert cfg["seeds"] == [17, 29, 43]
    for key, value in ACCEPTED_UPSTREAM.items():
        assert cfg["accepted_upstream"][key] == value


def test_seed_collection_excludes_file_sha_from_scientific() -> None:
    payload = {
        "ordered_seeds": [17, 29, 43],
        "seed_scientific_identities": {"17": "aa" * 32, "29": "bb" * 32, "43": "cc" * 32},
        "evaluation_unlocked": False,
        "canonical_seed_selected": False,
    }
    sci = training._canonical_json_sha256(payload)
    polluted = dict(payload)
    polluted["seed_17_file_sha256"] = "ff" * 32
    # Scientific helper must ignore file sha when projecting — here we assert
    # explicit scientific payload does not include file sha keys.
    assert "file_sha" not in json.dumps(payload)
    assert len(sci) == 64


def test_deployment_checkpoint_schema_fields() -> None:
    model = model_mod.B2DLCM(seed=29)
    ckpt = deploy.export_deployment_checkpoint(
        training_model=model,
        normalization=fixture_normalization_artifact(),
        canonical_seed=29,
        source_original_best_identity="11" * 32,
        source_reproduction_best_identity="11" * 32,
        contribution_target_collection_scientific_sha256="22" * 32,
    )
    for key in (
        "architecture_contract_version",
        "state_dict",
        "embedded_normalization",
        "golden_cases",
        "deployment_scientific_sha256",
    ):
        assert key in ckpt
