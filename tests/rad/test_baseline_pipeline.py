from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from rad.errors import (
    ArtifactIntegrityError,
    MetricComputationError,
    OutputProtectionError,
)
from tests.rad.contracts.baseline import (
    assert_completed_baseline_manifest,
    assert_no_baseline_artifacts,
    assert_required_metrics_finite,
    load_json,
    normalized_metrics_from_log_percentages,
    write_minimal_baseline_config,
    write_sample_log_txt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.reproduce_baseline import (  # noqa: E402
    build_test_command,
    build_train_command,
    run_baseline,
    BaselineConfig,
)


@pytest.fixture
def dataset_roots(tmp_path: Path) -> tuple[Path, Path]:
    train_data = tmp_path / "mvtec"
    test_data = tmp_path / "visa"
    train_data.mkdir()
    test_data.mkdir()
    return train_data, test_data


@pytest.fixture
def baseline_config_path(
    tmp_path: Path, dataset_roots: tuple[Path, Path]
) -> tuple[Path, Path]:
    train_data, test_data = dataset_roots
    output_dir = tmp_path / "baseline_out"
    config_path = tmp_path / "baseline.yaml"
    write_minimal_baseline_config(
        config_path,
        train_data=train_data,
        test_data=test_data,
        output_dir=output_dir,
    )
    return config_path, output_dir


def _cfg_from_path(config_path: Path) -> BaselineConfig:
    return BaselineConfig.from_yaml(config_path)


def test_mock_training_creates_checkpoint_and_runs_evaluation(
    baseline_config_path: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, output_dir = baseline_config_path
    cfg = _cfg_from_path(config_path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"epoch_{cfg.epoch}.pth"
    result_dir = output_dir / "results" / f"epoch_{cfg.epoch}"
    train_cmd = build_train_command(cfg, checkpoint_dir)
    test_cmd = build_test_command(cfg, checkpoint_path, result_dir)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd == train_cmd:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"trained-checkpoint")
        elif cmd == test_cmd:
            write_sample_log_txt(result_dir)
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.reproduce_baseline.subprocess.run", fake_run)
    monkeypatch.setattr("tools.reproduce_baseline.git_sha", lambda: "deadbeef")
    monkeypatch.setattr(
        "tools.reproduce_baseline.package_versions",
        lambda: {"python": "3.11.0", "torch": "2.0.0"},
    )

    assert run_baseline(config_path) == 0
    assert calls[0] == train_cmd
    assert calls[1] == test_cmd
    assert checkpoint_path.is_file()

    manifest = load_json(output_dir / "manifest.json")
    assert_completed_baseline_manifest(manifest)
    metrics = load_json(result_dir / "metrics.json")
    assert_required_metrics_finite(metrics)
    assert metrics == normalized_metrics_from_log_percentages()


def test_checkpoint_validation_occurs_after_training_not_before(
    baseline_config_path: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, output_dir = baseline_config_path
    cfg = _cfg_from_path(config_path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"epoch_{cfg.epoch}.pth"
    result_dir = output_dir / "results" / f"epoch_{cfg.epoch}"
    train_cmd = build_train_command(cfg, checkpoint_dir)
    test_cmd = build_test_command(cfg, checkpoint_path, result_dir)
    observed_missing_before_train = checkpoint_path.exists()

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd == train_cmd:
            assert not checkpoint_path.exists()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"trained-checkpoint")
        elif cmd == test_cmd:
            assert checkpoint_path.is_file()
            write_sample_log_txt(result_dir)
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.reproduce_baseline.subprocess.run", fake_run)
    monkeypatch.setattr("tools.reproduce_baseline.git_sha", lambda: "deadbeef")
    monkeypatch.setattr(
        "tools.reproduce_baseline.package_versions",
        lambda: {"python": "3.11.0", "torch": "2.0.0"},
    )

    assert observed_missing_before_train is False
    assert run_baseline(config_path) == 0


def test_external_checkpoint_skips_training(
    baseline_config_path: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, output_dir = baseline_config_path
    cfg = _cfg_from_path(config_path)
    external_ckpt = config_path.parent / "external.pth"
    external_ckpt.write_bytes(b"external-checkpoint")
    result_dir = output_dir / "results" / f"epoch_{cfg.epoch}"
    test_cmd = build_test_command(cfg, external_ckpt, result_dir)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert cmd == test_cmd
        write_sample_log_txt(result_dir)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.reproduce_baseline.subprocess.run", fake_run)
    monkeypatch.setattr("tools.reproduce_baseline.git_sha", lambda: "deadbeef")
    monkeypatch.setattr(
        "tools.reproduce_baseline.package_versions",
        lambda: {"python": "3.11.0", "torch": "2.0.0"},
    )

    assert run_baseline(config_path, checkpoint=external_ckpt) == 0
    assert len(calls) == 1
    manifest = load_json(output_dir / "manifest.json")
    assert manifest["eval_only"] is True
    assert manifest["commands"]["train"] is None


def test_missing_external_checkpoint_fails(
    baseline_config_path: tuple[Path, Path],
) -> None:
    config_path, output_dir = baseline_config_path
    missing_ckpt = output_dir / "missing.pth"
    with pytest.raises(ArtifactIntegrityError, match="checkpoint"):
        run_baseline(config_path, checkpoint=missing_ckpt)


def test_training_success_without_checkpoint_fails(
    baseline_config_path: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _output_dir = baseline_config_path
    cfg = _cfg_from_path(config_path)
    checkpoint_dir = cfg.output_dir / "checkpoints"
    train_cmd = build_train_command(cfg, checkpoint_dir)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd == train_cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("tools.reproduce_baseline.subprocess.run", fake_run)
    monkeypatch.setattr("tools.reproduce_baseline.git_sha", lambda: "deadbeef")
    monkeypatch.setattr(
        "tools.reproduce_baseline.package_versions",
        lambda: {"python": "3.11.0", "torch": "2.0.0"},
    )

    with pytest.raises(ArtifactIntegrityError, match="checkpoint"):
        run_baseline(config_path)


def test_zero_evaluation_exit_with_missing_metrics_fails(
    baseline_config_path: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, output_dir = baseline_config_path
    cfg = _cfg_from_path(config_path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"epoch_{cfg.epoch}.pth"
    result_dir = output_dir / "results" / f"epoch_{cfg.epoch}"
    train_cmd = build_train_command(cfg, checkpoint_dir)
    test_cmd = build_test_command(cfg, checkpoint_path, result_dir)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd == train_cmd:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"trained-checkpoint")
        elif cmd == test_cmd:
            result_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.reproduce_baseline.subprocess.run", fake_run)
    monkeypatch.setattr("tools.reproduce_baseline.git_sha", lambda: "deadbeef")
    monkeypatch.setattr(
        "tools.reproduce_baseline.package_versions",
        lambda: {"python": "3.11.0", "torch": "2.0.0"},
    )

    with pytest.raises(MetricComputationError, match="metric"):
        run_baseline(config_path)


def test_existing_output_is_refused(
    baseline_config_path: tuple[Path, Path],
) -> None:
    config_path, output_dir = baseline_config_path
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text('{"status":"completed"}', encoding="utf-8")
    with pytest.raises(OutputProtectionError):
        run_baseline(config_path)


def test_dry_run_writes_nothing(
    baseline_config_path: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, output_dir = baseline_config_path
    assert run_baseline(config_path, dry_run=True) == 0
    captured = capsys.readouterr().out
    assert "train.py" in captured
    assert "test.py" in captured
    assert_no_baseline_artifacts(output_dir)


def test_full_pipeline_integration(
    baseline_config_path: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, output_dir = baseline_config_path
    cfg = _cfg_from_path(config_path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"epoch_{cfg.epoch}.pth"
    result_dir = output_dir / "results" / f"epoch_{cfg.epoch}"
    train_cmd = build_train_command(cfg, checkpoint_dir)
    test_cmd = build_test_command(cfg, checkpoint_path, result_dir)
    stages: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd == train_cmd:
            stages.append("train")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"pipeline-checkpoint")
        elif cmd == test_cmd:
            stages.append("test")
            write_sample_log_txt(result_dir)
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.reproduce_baseline.subprocess.run", fake_run)
    monkeypatch.setattr("tools.reproduce_baseline.git_sha", lambda: "cafebabe")
    monkeypatch.setattr(
        "tools.reproduce_baseline.package_versions",
        lambda: {"python": "3.11.0", "torch": "2.0.0"},
    )

    assert run_baseline(config_path) == 0
    assert stages == ["train", "test"]
    manifest = load_json(output_dir / "manifest.json")
    assert_completed_baseline_manifest(manifest)
    assert manifest["checkpoint_sha256"]
    metrics = load_json(result_dir / "metrics.json")
    assert_required_metrics_finite(metrics)
