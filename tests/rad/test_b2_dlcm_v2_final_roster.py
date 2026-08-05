"""RED/GREEN tests for B2-05C1 untouched final roster."""

from __future__ import annotations

import pytest

from rad.phase_b import b2_dlcm_v2_final_roster as subject
from rad.phase_b import b2_dlcm_v2_protocol as protocol
from tests.rad.b2_dlcm_v2_fixtures import hermetic_identity_candidates


def test_hermetic_roster_deterministic_16_no_paths() -> None:
    candidates = hermetic_identity_candidates(per_group=6)
    excluded = [c["stable_sample_id"] for c in candidates[:2]]  # exclude some, not all
    # Build enough remaining: per_group=6, exclude 2 total across groups may still leave 4+
    roster = subject.hermetic_roster_from_records(
        candidates=candidates,
        excluded_ids=excluded,
        implementation_commit="abc1234deadbeef",
        source_manifest_scientific_sha256="f" * 64,
    )
    assert roster["counts"]["total"] == 16
    assert roster["counts"]["overlap"] == 0
    assert roster["paths_present"] is False
    assert roster["final_content_resolved"] is False
    subject.assert_roster_no_paths(roster)
    ids = [r["stable_sample_id"] for r in roster["records"]]
    assert len(ids) == 16
    assert len(set(ids)) == 16
    assert not (set(ids) & set(excluded))
    # Determinism
    again = subject.hermetic_roster_from_records(
        candidates=list(reversed(candidates)),
        excluded_ids=excluded,
        implementation_commit="abc1234deadbeef",
        source_manifest_scientific_sha256="f" * 64,
    )
    assert [r["stable_sample_id"] for r in again["records"]] == ids


def test_insufficient_candidates_fail_closed() -> None:
    candidates = hermetic_identity_candidates(per_group=3)
    with pytest.raises(
        subject.B2DLCMV2FinalRosterError, match="B2_DLCM_FINAL_ROSTER_INSUFFICIENT"
    ):
        subject.hermetic_roster_from_records(
            candidates=candidates,
            excluded_ids=[],
            implementation_commit="abc1234",
            source_manifest_scientific_sha256="f" * 64,
        )


def test_overlap_fail_closed() -> None:
    candidates = hermetic_identity_candidates(per_group=5)
    # Exclude nothing but force overlap by putting selected ids into excluded after sort:
    # take first 4 of each group which are idx 00-03; exclude those.
    excluded = [
        c["stable_sample_id"]
        for c in candidates
        if c["stable_sample_id"].endswith("00-" + "a" * 32)
        or c["stable_sample_id"].endswith("01-" + "a" * 32)
        or c["stable_sample_id"].endswith("02-" + "a" * 32)
        or c["stable_sample_id"].endswith("03-" + "a" * 32)
    ]
    # After excluding 00-03, only idx 04 remains → insufficient rather than overlap.
    with pytest.raises(
        subject.B2DLCMV2FinalRosterError, match="B2_DLCM_FINAL_ROSTER_INSUFFICIENT"
    ):
        subject.hermetic_roster_from_records(
            candidates=candidates,
            excluded_ids=excluded,
            implementation_commit="abc1234",
            source_manifest_scientific_sha256="f" * 64,
        )


def test_content_access_forbidden_before_unlock() -> None:
    with pytest.raises(
        protocol.B2DLCMV2ProtocolError, match="B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN"
    ):
        protocol.forbid_final_content_access(unlocked=False, context="roster_path_probe")


def test_public_fields_only() -> None:
    candidates = hermetic_identity_candidates(per_group=4)
    roster = subject.hermetic_roster_from_records(
        candidates=candidates,
        excluded_ids=[],
        implementation_commit="abc1234",
        source_manifest_scientific_sha256="f" * 64,
    )
    for row in roster["records"]:
        assert set(row) == set(subject.PUBLIC_FIELDS)
        assert row["selection_rank"] in {1, 2, 3, 4}
