from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from rad.data.adapters.registry import get_adapter
from rad.data.adapters.visa import VisAAdapter
from rad.errors import DatasetIntegrityError


def _write_rgb(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _write_mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), value).save(path)


def _build_visa_meta_fixture(root: Path) -> None:
    _write_rgb(root / "candle" / "Data" / "Images" / "Normal" / "0790.JPG", (1, 1, 1))
    _write_rgb(root / "candle" / "Data" / "Images" / "Anomaly" / "000.JPG", (2, 2, 2))
    _write_mask(root / "candle" / "Data" / "Masks" / "Anomaly" / "000.png")
    _write_rgb(root / "cashew" / "Data" / "Images" / "Normal" / "0001.JPG", (3, 3, 3))

    meta = {
        "Test": {  # non-canonical key; adapter must normalize
            "candle": [
                {
                    "img_path": "candle/Data/Images/Anomaly/000.JPG",
                    "mask_path": "candle/Data/Masks/Anomaly/000.png",
                    "cls_name": "candle",
                    "specie_name": "",
                    "anomaly": "1",  # string label; must normalize to int
                },
                {
                    "img_path": "candle/Data/Images/Normal/0790.JPG",
                    "mask_path": "",
                    "cls_name": "candle",
                    "specie_name": "",
                    "anomaly": 0,
                },
            ],
            "cashew": [
                {
                    "image": "cashew/Data/Images/Normal/0001.JPG",  # alias key
                    "mask": None,
                    "cls_name": "cashew",
                    "label": "normal",
                },
            ],
        }
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_visa_metadata_normalization_and_deterministic_ids(tmp_path: Path) -> None:
    _build_visa_meta_fixture(tmp_path)
    adapter = VisAAdapter(tmp_path)

    first = adapter.records("test")
    second = adapter.records("test")
    ids = [r.sample_id for r in first]

    assert ids == sorted(ids)
    assert ids == [r.sample_id for r in second]
    assert ids == [
        "candle/Data/Images/Anomaly/000.JPG",
        "candle/Data/Images/Normal/0790.JPG",
        "cashew/Data/Images/Normal/0001.JPG",
    ]


def test_visa_normal_mask_none_and_anomaly_mask_resolves(tmp_path: Path) -> None:
    _build_visa_meta_fixture(tmp_path)
    adapter = VisAAdapter(tmp_path)
    records = {r.sample_id: r for r in adapter.records("test")}

    normal = records["candle/Data/Images/Normal/0790.JPG"]
    anomalous = records["candle/Data/Images/Anomaly/000.JPG"]

    assert normal.dataset == "visa"
    assert normal.image_label == 0
    assert normal.mask_path is None
    assert adapter.open_mask(normal) is None

    assert anomalous.image_label == 1
    assert anomalous.mask_path is not None
    mask = adapter.open_mask(anomalous)
    assert mask is not None
    assert mask.mode == "L"


def test_visa_open_image_is_rgb(tmp_path: Path) -> None:
    _build_visa_meta_fixture(tmp_path)
    adapter = VisAAdapter(tmp_path)
    image = adapter.open_image(adapter.records("test")[0])
    assert image.mode == "RGB"


def test_visa_missing_image_fails(tmp_path: Path) -> None:
    _build_visa_meta_fixture(tmp_path)
    (tmp_path / "candle" / "Data" / "Images" / "Normal" / "0790.JPG").unlink()

    with pytest.raises(DatasetIntegrityError, match="missing image|0790"):
        VisAAdapter(tmp_path).records("test")


def test_visa_anomalous_missing_mask_fails(tmp_path: Path) -> None:
    _build_visa_meta_fixture(tmp_path)
    (tmp_path / "candle" / "Data" / "Masks" / "Anomaly" / "000.png").unlink()

    with pytest.raises(DatasetIntegrityError, match="mask"):
        VisAAdapter(tmp_path).records("test")


def test_visa_duplicate_sample_id_fails(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "candle" / "Data" / "Images" / "Normal" / "0790.JPG")
    meta = {
        "test": {
            "candle": [
                {
                    "img_path": "candle/Data/Images/Normal/0790.JPG",
                    "mask_path": "",
                    "cls_name": "candle",
                    "anomaly": 0,
                },
                {
                    "img_path": "candle/Data/Images/Normal/0790.JPG",
                    "mask_path": "",
                    "cls_name": "candle",
                    "anomaly": 0,
                },
            ]
        }
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(DatasetIntegrityError, match="duplicate"):
        VisAAdapter(tmp_path).records("test")


def test_get_adapter_returns_live_visa_adapter(tmp_path: Path) -> None:
    _build_visa_meta_fixture(tmp_path)
    adapter = get_adapter("VisA", tmp_path)
    assert isinstance(adapter, VisAAdapter)
    assert len(adapter.records("test")) == 3
