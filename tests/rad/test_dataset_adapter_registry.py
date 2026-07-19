from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pytest

from rad.data.adapters.protocol import AnomalyDatasetAdapter
from rad.data.adapters.registry import (
    get_adapter,
    normalize_dataset_name,
    planned_unsupported_dataset_names,
    supported_dataset_names,
)
from rad.data.adapters.types import EvaluationRecord
from rad.errors import ConfigurationContractError


def test_evaluation_record_is_frozen_and_complete(tmp_path: Path) -> None:
    record = EvaluationRecord(
        sample_id="bottle/test/good/000.png",
        dataset="mvtec",
        category="bottle",
        image_path=tmp_path / "bottle/test/good/000.png",
        mask_path=None,
        image_label=0,
        split="test",
    )
    assert record.mask_path is None
    with pytest.raises(AttributeError):
        record.sample_id = "mutated"  # type: ignore[misc]


def test_adapter_protocol_declares_required_methods() -> None:
    hints = get_type_hints(AnomalyDatasetAdapter.records)
    assert "split" in hints or "return" in hints
    assert hasattr(AnomalyDatasetAdapter, "records")
    assert hasattr(AnomalyDatasetAdapter, "open_image")
    assert hasattr(AnomalyDatasetAdapter, "open_mask")


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("mvtec", "mvtec"),
        ("MVTec", "mvtec"),
        ("MVTec AD", "mvtec"),
        ("MVTec-AD", "mvtec"),
        ("mvtec_ad", "mvtec"),
        ("visa", "visa"),
        ("VisA", "visa"),
        ("VISA", "visa"),
    ],
)
def test_normalize_dataset_name_aliases(raw: str, canonical: str) -> None:
    assert normalize_dataset_name(raw) == canonical


@pytest.mark.parametrize(
    "raw",
    ["btad", "BTAD", "ksdd2", "KSDD2", "dagm", "DAGM", "dtd-synthetic", "DTD-Synthetic"],
)
def test_normalize_planned_unsupported_names(raw: str) -> None:
    key = normalize_dataset_name(raw)
    assert key in planned_unsupported_dataset_names()


def test_supported_datasets_are_only_mvtec_and_visa() -> None:
    assert supported_dataset_names() == frozenset({"mvtec", "visa"})


def test_get_adapter_unknown_name_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationContractError, match="unknown|Unknown"):
        get_adapter("not-a-dataset", tmp_path)


@pytest.mark.parametrize("name", ["btad", "ksdd2", "dagm", "dtd-synthetic"])
def test_get_adapter_planned_unsupported_raises_not_implemented(
    name: str, tmp_path: Path
) -> None:
    with pytest.raises(NotImplementedError, match=name):
        get_adapter(name, tmp_path)


def test_get_adapter_routes_mvtec_and_visa_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: dict[str, Path] = {}

    class FakeMVTec:
        def __init__(self, root: Path) -> None:
            created["mvtec"] = Path(root)

    class FakeVisA:
        def __init__(self, root: Path) -> None:
            created["visa"] = Path(root)

    monkeypatch.setattr(
        "rad.data.adapters.registry._ADAPTER_FACTORIES",
        {
            "mvtec": FakeMVTec,
            "visa": FakeVisA,
        },
    )

    mvtec = get_adapter("MVTec-AD", tmp_path / "mvtec")
    visa = get_adapter("VisA", tmp_path / "visa")

    assert isinstance(mvtec, FakeMVTec)
    assert isinstance(visa, FakeVisA)
    assert created["mvtec"] == tmp_path / "mvtec"
    assert created["visa"] == tmp_path / "visa"
