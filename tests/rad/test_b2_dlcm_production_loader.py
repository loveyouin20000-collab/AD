"""RED/GREEN tests for B2 DLCM production upstream input verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rad.phase_b import b2_dlcm_training as subject
from tests.rad.b2_dlcm_fixtures import ACCEPTED_UPSTREAM

DESC_RUN = Path(
    "/root/autodl-tmp/AD-phase-b2-descriptor-real-extraction/"
    "artifacts/phase_b/b2_descriptor_artifacts/authoritative-run-a-20260729-013956"
)
CONTRIB_RUN = Path(
    "/root/autodl-tmp/AD-phase-b2-contribution-target-materialization/"
    "artifacts/phase_b/b2_contribution_targets/authoritative-run-a-20260804-030431"
)


def _is_accessible_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_accessible_dir(DESC_RUN) or not _is_accessible_dir(CONTRIB_RUN),
    reason="accepted upstream artifact runs not present on this host",
)


def _paths() -> dict[str, Path]:
    return {
        "descriptor_manifest": DESC_RUN / "final_manifest.json",
        "descriptor_root": DESC_RUN,
        "contribution_target_manifest": CONTRIB_RUN / "final_manifest.json",
        "contribution_target_root": CONTRIB_RUN,
    }


def test_production_loader_calls_verify_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    import rad.phase_b.b2_contribution_targets as contrib_mod
    import rad.phase_b.b2_descriptor_artifacts as desc_mod

    real_desc_verify = desc_mod.verify_descriptor_artifact_collection
    real_contrib_verify = contrib_mod.verify_contribution_target_collection
    real_desc_receipt = desc_mod.verify_final_manifest_receipt
    real_contrib_receipt = contrib_mod.verify_final_manifest_receipt

    def wrap(name: str, fn: Any) -> Any:
        def _inner(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            return fn(*args, **kwargs)

        return _inner

    monkeypatch.setattr(
        desc_mod, "verify_final_manifest_receipt", wrap("desc_receipt", real_desc_receipt)
    )
    monkeypatch.setattr(
        desc_mod,
        "verify_descriptor_artifact_collection",
        wrap("desc_verify", real_desc_verify),
    )
    monkeypatch.setattr(
        contrib_mod,
        "verify_final_manifest_receipt",
        wrap("contrib_receipt", real_contrib_receipt),
    )
    monkeypatch.setattr(
        contrib_mod,
        "verify_contribution_target_collection",
        wrap("contrib_verify", real_contrib_verify),
    )

    paths = _paths()
    bundle = subject.load_verified_b2_dlcm_training_inputs(
        accepted_upstream=ACCEPTED_UPSTREAM,
        evaluation_unlocked=False,
        **paths,
    )
    assert "desc_receipt" in calls and "desc_verify" in calls
    assert "contrib_receipt" in calls and "contrib_verify" in calls
    assert len(bundle.training_records) == 16
    assert len(bundle.calibration_records) == 8
    assert len(bundle.evaluation_record_ids) == 8
    assert bundle.teacher_forward_count == 0


def test_production_loader_rejects_identity_mismatch() -> None:
    paths = _paths()
    bad = dict(ACCEPTED_UPSTREAM)
    bad["descriptor_collection_scientific_sha256"] = "0" * 64
    with pytest.raises(subject.B2DLCMTrainingError) as exc:
        subject.load_verified_b2_dlcm_training_inputs(
            accepted_upstream=bad,
            evaluation_unlocked=False,
            **paths,
        )
    assert exc.value.code == "B2_DLCM_UPSTREAM_IDENTITY_MISMATCH"


def test_production_loader_locks_evaluation_content() -> None:
    paths = _paths()
    bundle = subject.load_verified_b2_dlcm_training_inputs(
        accepted_upstream=ACCEPTED_UPSTREAM,
        evaluation_unlocked=False,
        **paths,
    )
    with pytest.raises(subject.B2DLCMTrainingError) as exc:
        bundle.require_evaluation_records()
    assert exc.value.code == "B2_DLCM_EVAL_LOCKED"


def test_production_loader_rejects_non_production_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed when contribution artifact_kind is not production."""
    import rad.phase_b.b2_contribution_targets as contrib_mod

    paths = _paths()
    real = contrib_mod.verify_contribution_target_collection

    def _fake(*, config: Any, run_dir: Any) -> Any:
        verified = real(config=config, run_dir=run_dir)
        # Mutate a shallow copy of manifest kind via monkeypatched require.

        class _Bad:
            run_dir = verified.run_dir
            manifest = dict(verified.manifest) | {"artifact_kind": "test_fixture"}
            records_by_id = verified.records_by_id
            calibration_artifact = verified.calibration_artifact
            normalization = verified.normalization
            teacher_forward_count = verified.teacher_forward_count

        return _Bad()

    monkeypatch.setattr(contrib_mod, "verify_contribution_target_collection", _fake)
    with pytest.raises(subject.B2DLCMTrainingError) as exc:
        subject.load_verified_b2_dlcm_training_inputs(
            accepted_upstream=ACCEPTED_UPSTREAM,
            evaluation_unlocked=False,
            **paths,
        )
    assert exc.value.code in {
        "B2_DLCM_ARTIFACT_KIND_INVALID",
        "B2_TARGET_ARTIFACT_KIND_INVALID",
    }


def test_empty_manifest_rejected_by_loader(tmp_path: Path) -> None:
    (tmp_path / "desc").mkdir()
    (tmp_path / "tgt").mkdir()
    desc_m = tmp_path / "desc.json"
    tgt_m = tmp_path / "tgt.json"
    desc_m.write_text("{}", encoding="utf-8")
    tgt_m.write_text("{}", encoding="utf-8")
    with pytest.raises(subject.B2DLCMTrainingError):
        subject.load_verified_b2_dlcm_training_inputs(
            descriptor_manifest=desc_m,
            descriptor_root=tmp_path / "desc",
            contribution_target_manifest=tgt_m,
            contribution_target_root=tmp_path / "tgt",
            accepted_upstream=ACCEPTED_UPSTREAM,
            evaluation_unlocked=False,
        )
