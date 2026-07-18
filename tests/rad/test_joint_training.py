from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rad.config import ExperimentConfig
from rad.models.descriptors import CheckpointContextExtractor, LayerDescriptorExtractor
from rad.models.dlcm import DLCM
from rad.models.lse import LSE
from rad.trainers.joint_trainer import (
    JointTrainer,
    NoRegressionThresholds,
    compute_cost_weight,
    evaluate_no_regression,
    soft_expected_depth_ratio,
    validate_staged_checkpoint_pair,
)

REPO = Path(__file__).resolve().parents[2]


def test_joint_config_is_explicit_ablation_opt_in() -> None:
    cfg = ExperimentConfig.from_yaml("configs/rad/joint.yaml")
    assert cfg.training is not None
    assert cfg.training.mode == "joint"
    assert cfg.joint is not None
    assert cfg.joint.enabled is True
    assert cfg.joint.primary_pipeline is False
    assert cfg.joint.trainable_modules == ("dlcm", "lse")


def test_cost_weight_ramps_over_first_twenty_percent() -> None:
    assert compute_cost_weight(0.00, final_weight=0.05, ramp_fraction=0.20) == 0.0
    assert compute_cost_weight(0.10, final_weight=0.05, ramp_fraction=0.20) == pytest.approx(0.025)
    assert compute_cost_weight(0.20, final_weight=0.05, ramp_fraction=0.20) == pytest.approx(0.05)
    assert compute_cost_weight(0.90, final_weight=0.05, ramp_fraction=0.20) == pytest.approx(0.05)


def test_soft_expected_depth_is_configuration_driven() -> None:
    exit_at_first = soft_expected_depth_ratio(
        torch.tensor([[40.0, -40.0]]), (4, 6), 8, temperature=1.0
    )
    exit_at_second = soft_expected_depth_ratio(
        torch.tensor([[-40.0, 40.0]]), (4, 6), 8, temperature=1.0
    )
    run_full = soft_expected_depth_ratio(
        torch.tensor([[-40.0, -40.0]]), (4, 6), 8, temperature=1.0
    )
    assert exit_at_first.item() == pytest.approx(0.5, abs=1e-5)
    assert exit_at_second.item() == pytest.approx(0.75, abs=1e-5)
    assert run_full.item() == pytest.approx(1.0, abs=1e-5)


def test_no_regression_gate_requires_every_metric() -> None:
    thresholds = NoRegressionThresholds(0.002, 0.002, 0.01)
    reference = {"pixel_ap": 0.800, "pro": 0.900, "mean_sample_error": 0.200}
    passing = {"pixel_ap": 0.799, "pro": 0.899, "mean_sample_error": 0.201}
    failing = {"pixel_ap": 0.797, "pro": 0.899, "mean_sample_error": 0.201}
    assert evaluate_no_regression(passing, reference, thresholds).passed is True
    result = evaluate_no_regression(failing, reference, thresholds)
    assert result.passed is False
    assert "pixel_ap_drop" in result.reasons


def _write_checkpoint_with_manifest(
    root: Path,
    name: str,
    checkpoint_hash: str,
    upstream_fusion_hash: str | None,
) -> Path:
    checkpoint = root / f"{name}.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    stage = "fusion" if name == "fusion" else "lse"
    manifest = {
        "schema_version": "rad-checkpoint-v1",
        "stage": stage,
        "status": "passed",
        "checkpoint_sha256": checkpoint_hash,
        "candidate_layers": [6, 12, 18, 24],
        "source_dataset": "mvtec",
        "split_manifest_hash": "split-111",
        "preprocessing_hash": "pre-v1",
        "teacher_checkpoint_hash": "teacher-v1",
        "descriptor_stats_hash": "stats-v1",
        "upstream_fusion_checkpoint_hash": upstream_fusion_hash,
        "gates": {"staged_training": True, "source_only_selection": True},
        "reference_full_depth_metrics": {
            "pixel_ap": 0.80,
            "pro": 0.90,
            "mean_sample_error": 0.20,
        },
    }
    if stage == "lse":
        manifest.pop("reference_full_depth_metrics")
    checkpoint.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    return checkpoint


def test_checkpoint_pair_rejects_mismatched_upstream_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fusion = _write_checkpoint_with_manifest(tmp_path, "fusion", "fusion-hash", None)
    lse = _write_checkpoint_with_manifest(
        tmp_path, "lse", "lse-hash", "wrong-fusion-hash"
    )
    monkeypatch.setattr(
        "rad.trainers.joint_trainer.sha256_file",
        lambda path: "fusion-hash" if Path(path) == fusion else "lse-hash",
    )
    with pytest.raises(ValueError, match="upstream_fusion_checkpoint_hash"):
        validate_staged_checkpoint_pair(fusion, lse)


def _synthetic_joint_batch(
    *,
    b: int = 2,
    h: int = 8,
    layers: tuple[int, ...] = (2, 4, 6, 8),
    train_depths: tuple[int, ...] = (4, 6, 8),
) -> dict:
    maps_by_depth = {}
    shapley_by_depth = {}
    layer_ids_by_depth = {}
    for depth in train_depths:
        avail = [x for x in layers if x <= depth]
        n_layers = len(avail)
        maps_by_depth[depth] = torch.randn(b, n_layers, 1, h, h)
        dist = torch.softmax(torch.randn(b, n_layers), dim=-1)
        shapley_by_depth[depth] = {
            "distribution": dist,
            "phi": torch.randn(b, n_layers),
        }
        layer_ids_by_depth[depth] = torch.tensor(
            [avail for _ in range(b)], dtype=torch.long
        )
    return {
        "maps_by_depth": maps_by_depth,
        "layer_ids_by_depth": layer_ids_by_depth,
        "mask": (torch.rand(b, 1, h, h) > 0.8).float(),
        "image_label": torch.ones(b),
        "teacher_logits": torch.randn(b, 1, h, h),
        "shapley_by_depth": shapley_by_depth,
        "source_dataset": "mvtec",
    }


def test_joint_train_step_synthetic_microbatch() -> None:
    torch.manual_seed(0)
    layers = (2, 4, 6, 8)
    early = (4, 6)
    dlcm = DLCM(max_layer_id=8, alpha=0.0)
    with torch.no_grad():
        dlcm.scorer.weight.normal_(0, 0.05)
        dlcm.scorer.bias.normal_(0, 0.05)
    lse = LSE(state_dim=26, early_depths=early)
    trainer = JointTrainer(
        dlcm=dlcm,
        lse=lse,
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=8),
        train_depths=(4, 6, 8),
        candidate_layers=layers,
        early_depths=early,
        full_depth=8,
        fusion_loss_weight=1.0,
        lse_loss_weight=0.5,
        compute_final_weight=0.05,
        compute_ramp_fraction=0.2,
        compute_target_depth_ratio=0.75,
        soft_exit_temperature=1.0,
        epsilon_gain=0.05,
        epsilon_absolute=0.5,
    )
    names = trainer.trainable_parameter_names()
    assert all(n.startswith("dlcm.") or n.startswith("lse.") for n in names)
    assert any(n.startswith("dlcm.") for n in names)
    assert any(n.startswith("lse.") for n in names)

    batch = _synthetic_joint_batch()
    opt = torch.optim.AdamW(trainer.trainable_parameters(), lr=1e-5)
    out = trainer.train_step(batch, progress=0.5, optimizer=opt)
    assert torch.isfinite(out.total_loss)
    assert out.online_targets_detached is True
    assert batch["source_dataset"] != "visa"


def test_train_joint_cli_requires_allow_joint() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "train_joint.py"),
            "--config",
            str(REPO / "configs" / "rad" / "joint.yaml"),
            "--fusion-checkpoint",
            str(REPO / "artifacts/fusion/mvtec_seed111/checkpoints/best_gate_passed.pt"),
            "--lse-checkpoint",
            str(REPO / "artifacts/lse/mvtec_seed111/checkpoints/best_gate_passed.pt"),
            "--dry-run",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "allow-joint" in (proc.stderr + proc.stdout).lower()


def test_train_joint_dry_run_prints_provenance(tmp_path: Path) -> None:
    from rad.checkpoints.manifest_v1 import (
        SCHEMA_VERSION,
        CheckpointManifestV1,
        sha256_file,
        write_checkpoint_with_manifest,
    )

    fusion_ckpt = tmp_path / "fusion_best.pt"
    lse_ckpt = tmp_path / "lse_best.pt"
    fusion_ckpt.write_bytes(b"fusion-ci")
    lse_ckpt.write_bytes(b"lse-ci")
    fusion_hash = sha256_file(fusion_ckpt)
    lse_hash = sha256_file(lse_ckpt)
    layers = (6, 12, 18, 24)
    common = dict(
        schema_version=SCHEMA_VERSION,
        status="passed",
        candidate_layers=layers,
        source_dataset="mvtec",
        split_manifest_hash="split-ci",
        preprocessing_hash="pre-ci",
        teacher_checkpoint_hash="teacher-ci",
        descriptor_stats_hash="stats-ci",
        gates={"staged_training": True, "source_only_selection": True},
    )
    write_checkpoint_with_manifest(
        fusion_ckpt,
        CheckpointManifestV1(
            stage="fusion",
            checkpoint_sha256=fusion_hash,
            upstream_fusion_checkpoint_hash=None,
            reference_full_depth_metrics={
                "pixel_ap": 1.0,
                "pro": 1.0,
                "mean_sample_error": 1.0,
            },
            **common,
        ),
    )
    write_checkpoint_with_manifest(
        lse_ckpt,
        CheckpointManifestV1(
            stage="lse",
            checkpoint_sha256=lse_hash,
            upstream_fusion_checkpoint_hash=fusion_hash,
            reference_full_depth_metrics=None,
            **common,
        ),
    )
    out = tmp_path / "joint_dry_run"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "train_joint.py"),
            "--config",
            str(REPO / "configs" / "rad" / "joint.yaml"),
            "--fusion-checkpoint",
            str(fusion_ckpt),
            "--lse-checkpoint",
            str(lse_ckpt),
            "--allow-joint",
            "--dry-run",
            "--output-dir",
            str(out),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    blob = proc.stdout + proc.stderr
    assert "training_ablation" in blob
    assert "primary_pipeline=false" in blob or "primary_pipeline=False" in blob
    assert "dlcm" in blob and "lse" in blob
    assert "1e-5" in blob or "1.0e-5" in blob or "0.00001" in blob
    assert not (out / "checkpoints" / "best_gate_passed.pt").exists()
