from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from rad.data.adapters.mvtec import MVTecAdapter
from rad.data.adapters.registry import get_adapter
from rad.errors import DatasetIntegrityError


def _write_rgb(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def _write_mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 4), value).save(path)


def _build_mvtec_fixture(root: Path) -> None:
    _write_rgb(root / "bottle" / "test" / "good" / "000.png", (1, 2, 3))
    _write_rgb(root / "bottle" / "test" / "good" / "001.png", (4, 5, 6))
    _write_rgb(root / "bottle" / "test" / "broken_large" / "000.png", (7, 8, 9))
    _write_mask(root / "bottle" / "ground_truth" / "broken_large" / "000_mask.png")
    _write_rgb(root / "cable" / "test" / "good" / "000.png", (11, 12, 13))
    # Non-category noise directory must be ignored.
    (root / "OpenDataLab___MVTecAD").mkdir()


def test_mvtec_records_are_deterministic_and_stable(tmp_path: Path) -> None:
    _build_mvtec_fixture(tmp_path)
    adapter = MVTecAdapter(tmp_path)

    first = adapter.records("test")
    second = adapter.records("test")

    ids = [r.sample_id for r in first]
    assert ids == sorted(ids)
    assert ids == [r.sample_id for r in second]
    assert ids == [
        "bottle/test/broken_large/000.png",
        "bottle/test/good/000.png",
        "bottle/test/good/001.png",
        "cable/test/good/000.png",
    ]


def test_mvtec_category_filter_preserves_default_behavior_and_limits_enumeration(
    tmp_path: Path,
) -> None:
    _build_mvtec_fixture(tmp_path)
    adapter = MVTecAdapter(tmp_path)
    assert adapter.records("test", categories=None) == adapter.records("test")
    bottle = adapter.records("test", categories=("bottle",))
    assert [record.category for record in bottle] == ["bottle", "bottle", "bottle"]


def test_mvtec_category_filter_does_not_touch_unrequested_broken_category(
    tmp_path: Path,
) -> None:
    _build_mvtec_fixture(tmp_path)
    _write_rgb(tmp_path / "cable" / "test" / "broken" / "001.png")
    adapter = MVTecAdapter(tmp_path)
    with pytest.raises(DatasetIntegrityError, match="mask"):
        adapter.records("test")
    bottle = adapter.records("test", categories=("bottle",))
    assert {record.category for record in bottle} == {"bottle"}


def test_mvtec_normal_mask_is_none_and_anomaly_mask_resolves(tmp_path: Path) -> None:
    _build_mvtec_fixture(tmp_path)
    adapter = MVTecAdapter(tmp_path)
    records = {r.sample_id: r for r in adapter.records("test")}

    normal = records["bottle/test/good/000.png"]
    anomalous = records["bottle/test/broken_large/000.png"]

    assert normal.dataset == "mvtec"
    assert normal.category == "bottle"
    assert normal.split == "test"
    assert normal.image_label == 0
    assert normal.mask_path is None
    assert adapter.open_mask(normal) is None

    assert anomalous.image_label == 1
    assert anomalous.mask_path is not None
    mask = adapter.open_mask(anomalous)
    assert mask is not None
    assert mask.mode == "L"


def test_mvtec_open_image_is_rgb(tmp_path: Path) -> None:
    _build_mvtec_fixture(tmp_path)
    adapter = MVTecAdapter(tmp_path)
    record = adapter.records("test")[0]
    image = adapter.open_image(record)
    assert image.mode == "RGB"
    assert image.size == (4, 4)


def test_mvtec_missing_image_fails(tmp_path: Path) -> None:
    _build_mvtec_fixture(tmp_path)
    adapter = MVTecAdapter(tmp_path)
    record = next(r for r in adapter.records("test") if r.sample_id.endswith("good/000.png"))
    record.image_path.unlink()

    with pytest.raises(DatasetIntegrityError, match="missing image"):
        adapter.open_image(record)


def test_mvtec_anomalous_missing_mask_fails(tmp_path: Path) -> None:
    _build_mvtec_fixture(tmp_path)
    (tmp_path / "bottle" / "ground_truth" / "broken_large" / "000_mask.png").unlink()

    with pytest.raises(DatasetIntegrityError, match="mask"):
        MVTecAdapter(tmp_path).records("test")


def test_mvtec_duplicate_sample_id_fails(tmp_path: Path) -> None:
    root = tmp_path / "dup_root"
    _write_rgb(root / "bottle" / "test" / "good" / "000.png")
    (root / "bottle_alias").symlink_to(root / "bottle")

    with pytest.raises(DatasetIntegrityError, match="duplicate"):
        MVTecAdapter(root).records("test")


def test_get_adapter_returns_live_mvtec_adapter(tmp_path: Path) -> None:
    _build_mvtec_fixture(tmp_path)
    adapter = get_adapter("mvtec", tmp_path)
    assert isinstance(adapter, MVTecAdapter)
    assert len(adapter.records("test")) == 4
