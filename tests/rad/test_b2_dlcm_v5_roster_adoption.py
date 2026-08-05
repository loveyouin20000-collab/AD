"""V5 roster adoption unit tests (no committed adoption manifest required)."""

from __future__ import annotations

from pathlib import Path

from rad.phase_b import b2_dlcm_v5 as v5
from rad.phase_b import b2_dlcm_v5_roster_adoption as adoption

REPO = Path(__file__).resolve().parents[2]


def test_adoption_manifest_no_reselection() -> None:
    manifest = adoption.build_adoption_manifest(
        repo_root=REPO,
        implementation_commit="deadbeef" * 5,
    )
    roster = adoption.load_and_verify_c1_roster(REPO)
    adoption.assert_adoption_matches_roster(manifest, roster)
    assert manifest["source_roster_scientific_sha256"] == v5.ADOPTED_ROSTER_SCIENTIFIC
    assert manifest["selection_reused_without_change"] is True
    assert manifest["final_content_resolved"] is False
    assert manifest["paths_present"] is False
    assert len(manifest["ordered_stable_sample_ids"]) == 16
    assert manifest["proofs"]["accepted_manifest_present"] is False
    assert manifest["proofs"]["final_content_access_forbidden"] is True
    assert "v5_contract_identity" in manifest
