"""B1-05: accepted task-level categories must be complete real datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from rad.qualification import b1_cuda_equivalence as subject
from tests.rad.b2_hermetic import populate_b1_task_level_roots


@pytest.fixture
def task_level_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    mvtec, visa, sample = populate_b1_task_level_roots(tmp_path)
    monkeypatch.setattr(subject, "MVTEC_ROOT", mvtec)
    monkeypatch.setattr(subject, "VISA_ROOT", visa)
    monkeypatch.setattr(subject, "MVTEC_SAMPLE_FIXTURE_ROOT", sample)
    return mvtec, visa, sample


def test_mvtec_sample_is_rejected_as_accepted_task_level_category(
    task_level_roots: tuple[Path, Path, Path],
) -> None:
    audits = subject.audit_task_level_category_provenance()
    mvtec = next(item for item in audits if item["dataset"] == "mvtec")
    assert mvtec["canonical_category"] == "bottle"
    assert mvtec["accepted_for_b1_task_gate"] is True
    assert mvtec["invalid_predecessor"]["accepted_for_b1_task_gate"] is False
    assert "sample" in mvtec["invalid_predecessor"]["path"]


def test_task_level_samples_exclude_fixture_sample_paths(
    task_level_roots: tuple[Path, Path, Path],
) -> None:
    records = subject.load_task_level_category_samples()
    assert records, "expected real-category task-level samples"
    for sample_id, image_path, _mask, _label, category in records:
        path_l = str(image_path).lower()
        assert "/mvtec/sample/" not in path_l
        assert "fixture" not in path_l
        assert "/tests/" not in path_l
        assert category in {"mvtec/bottle", "visa/candle"}
        assert "mvtec_sample/" not in sample_id
