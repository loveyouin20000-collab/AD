from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rad.artifacts import atomic_write_json, refuse_existing_run
from rad.errors import (
    ArtifactIntegrityError,
    ConfigurationContractError,
    DatasetIntegrityError,
    MetricComputationError,
    OutputProtectionError,
    RADContractError,
    ScientificGateError,
    UnsupportedDatasetError,
)


def test_contract_error_hierarchy_is_explicit() -> None:
    assert issubclass(RADContractError, RuntimeError)
    for cls in (
        ConfigurationContractError,
        OutputProtectionError,
        DatasetIntegrityError,
        ArtifactIntegrityError,
        UnsupportedDatasetError,
        MetricComputationError,
        ScientificGateError,
    ):
        assert issubclass(cls, RADContractError)


def test_atomic_write_json_roundtrips_payload(tmp_path: Path) -> None:
    target = tmp_path / "run" / "manifest.json"
    payload = {"schema_version": 1, "status": "completed", "seed": 111}

    atomic_write_json(target, payload)

    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    leftovers = list(target.parent.glob("*.tmp*")) + list(target.parent.glob(".*.tmp*"))
    assert leftovers == []


def test_atomic_write_json_uses_temp_flush_fsync_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifacts" / "metrics.json"
    calls: dict[str, object] = {"fsync": 0, "replace": None}

    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd: int) -> None:
        calls["fsync"] = int(calls["fsync"]) + 1
        real_fsync(fd)

    def tracking_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        calls["replace"] = (Path(src), Path(dst))
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", tracking_fsync)
    monkeypatch.setattr(os, "replace", tracking_replace)

    atomic_write_json(target, {"ok": True})

    assert int(calls["fsync"]) >= 1
    assert calls["replace"] is not None
    src, dst = calls["replace"]  # type: ignore[misc]
    assert dst == target
    assert src != target
    assert src.parent == target.parent
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_atomic_write_json_does_not_leave_partial_destination_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out" / "manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"status": "completed"}', encoding="utf-8")

    def boom(_fd: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", boom)

    with pytest.raises(OSError, match="simulated fsync failure"):
        atomic_write_json(target, {"status": "running"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "completed"}


def test_refuse_existing_run_allows_missing_path(tmp_path: Path) -> None:
    refuse_existing_run(tmp_path / "fresh_run")


def test_refuse_existing_run_rejects_existing_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "existing_run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(OutputProtectionError, match="existing_run"):
        refuse_existing_run(run_dir)


def test_refuse_existing_run_rejects_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(OutputProtectionError, match="metrics.json"):
        refuse_existing_run(path)
