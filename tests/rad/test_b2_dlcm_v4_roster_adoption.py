"""Roster adoption tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rad.phase_b import b2_dlcm_v4 as v3
from rad.phase_b import b2_dlcm_v4_roster_adoption as adoption

REPO = Path(__file__).resolve().parents[2]


def test_adoption_binds_identity_and_ids() -> None:
    roster = adoption.load_and_verify_c1_roster(REPO)
    assert roster["roster_scientific_sha256"] == v3.ADOPTED_ROSTER_SCIENTIFIC
    assert roster["final_content_resolved"] is False
    assert roster["paths_present"] is False
    manifest = adoption.build_adoption_manifest(
        repo_root=REPO,
        implementation_commit="deadbeef" * 5,
    )
    assert manifest["selection_reused_without_change"] is True
    assert manifest["source_roster_scientific_sha256"] == v3.ADOPTED_ROSTER_SCIENTIFIC
    assert len(manifest["ordered_stable_sample_ids"]) == 16
    adoption.assert_adoption_matches_roster(manifest, roster)


def test_adoption_detects_id_mutation() -> None:
    roster = adoption.load_and_verify_c1_roster(REPO)
    manifest = adoption.build_adoption_manifest(
        repo_root=REPO,
        implementation_commit="c" * 40,
    )
    bad = dict(manifest)
    ids = list(bad["ordered_stable_sample_ids"])
    ids[0] = "mutated-id"
    bad["ordered_stable_sample_ids"] = ids
    with pytest.raises(adoption.B2DLCMV4RosterAdoptionError) as exc:
        adoption.assert_adoption_matches_roster(bad, roster)
    assert exc.value.code == "B2_DLCM_ROSTER_ADOPTION_MISMATCH"
