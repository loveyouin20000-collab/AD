"""V5 roster adoption unit tests (no committed adoption manifest required)."""

from __future__ import annotations

import shutil
from pathlib import Path

from rad.phase_b import b2_dlcm_v5 as v5
from rad.phase_b import b2_dlcm_v5_roster_adoption as adoption

REPO = Path(__file__).resolve().parents[2]


def _historical_root(tmp_path: Path) -> Path:
    phase_b = tmp_path / "docs" / "phase_b"
    phase_b.mkdir(parents=True)
    roster = REPO / "docs" / "phase_b" / "b2_05c1_final_evaluation_roster.json"
    shutil.copy2(roster, phase_b / roster.name)
    shutil.copy2(Path(str(roster) + ".sha256"), phase_b / f"{roster.name}.sha256")
    return tmp_path


def test_adoption_manifest_no_reselection(tmp_path: Path) -> None:
    historical_root = _historical_root(tmp_path)
    manifest = adoption.build_adoption_manifest(
        repo_root=historical_root,
        implementation_commit="deadbeef" * 5,
    )
    roster = adoption.load_and_verify_c1_roster(historical_root)
    adoption.assert_adoption_matches_roster(manifest, roster)
    assert manifest["source_roster_scientific_sha256"] == v5.ADOPTED_ROSTER_SCIENTIFIC
    assert manifest["selection_reused_without_change"] is True
    assert manifest["final_content_resolved"] is False
    assert manifest["paths_present"] is False
    assert len(manifest["ordered_stable_sample_ids"]) == 16
    assert manifest["proofs"]["accepted_manifest_present"] is False
    assert manifest["proofs"]["final_content_access_forbidden"] is True
    assert "v5_contract_identity" in manifest
