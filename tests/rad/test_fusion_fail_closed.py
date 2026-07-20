"""Increment 9: fusion training must fail closed."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from rad.errors import ArtifactIntegrityError
from tools import train_fusion as tf

REPO = Path(__file__).resolve().parents[2]


def test_missing_shapley_targets_fail_by_default(tmp_path: Path) -> None:
    missing = tmp_path / "no_shapley.pt"
    with pytest.raises(ArtifactIntegrityError, match="[Ss]hapley"):
        tf.require_shapley_targets(missing, allow_missing=False)


def test_missing_descriptor_stats_fail_by_default(tmp_path: Path) -> None:
    missing = tmp_path / "no_stats.json"
    with pytest.raises(ArtifactIntegrityError, match="[Dd]escriptor|[Ss]tat"):
        tf.require_descriptor_stats(missing, require=True)


def test_legacy_shapley_requires_explicit_opt_in(tmp_path: Path) -> None:
    missing = tmp_path / "no_shapley.pt"
    empty = tf.require_shapley_targets(missing, allow_missing=True)
    assert empty == {}


def test_missing_sample_shapley_fails_without_legacy() -> None:
    class _Cache:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict:
            return {
                "sample_id": "s0",
                "maps": {
                    12: {6: torch.zeros(4, 4), 12: torch.zeros(4, 4)},
                    18: {
                        6: torch.zeros(4, 4),
                        12: torch.zeros(4, 4),
                        18: torch.zeros(4, 4),
                    },
                    24: {
                        6: torch.zeros(4, 4),
                        12: torch.zeros(4, 4),
                        18: torch.zeros(4, 4),
                        24: torch.zeros(4, 4),
                    },
                },
                "mask_path": "",
                "label": 1,
                "teacher_logits": torch.zeros(1, 4, 4),
            }

    ds = tf.FusionCacheDataset(
        cache=_Cache(),  # type: ignore[arg-type]
        shapley_by_id={},
        data_root=Path("."),
        image_size=4,
        candidate_layers=(6, 12, 18, 24),
        train_depths=(12, 18, 24),
        allow_missing_shapley=False,
    )
    with pytest.raises(ArtifactIntegrityError, match="[Ss]hapley|sample"):
        _ = ds[0]


def test_legacy_outputs_marked_ineligible(tmp_path: Path) -> None:
    results = [
        {
            "seed": 1,
            "checkpoint": str(tmp_path / "seed_1" / "last.pt"),
            "gate_checkpoint": str(tmp_path / "seed_1" / "gate_passed.pt"),
            "cal_metrics": {"pixel_ap": 0.9, "equal_pixel_ap": 0.8},
            "no_regression_vs_equal": True,
            "legacy_mode": True,
            "eligible_for_evaluation": False,
            "steps": 1,
        }
    ]
    (tmp_path / "seed_1").mkdir()
    torch.save({"seed": 1}, tmp_path / "seed_1" / "last.pt")
    torch.save({"seed": 1}, tmp_path / "seed_1" / "gate_passed.pt")
    summary, code = tf.finalize_fusion_run(
        results,
        tmp_path,
        fail_if_no_gate_passes=True,
        legacy_mode=True,
    )
    assert code == 0
    assert summary["legacy_mode"] is True
    assert summary["eligible_for_evaluation"] is False
    assert summary["status"] == "completed"
    assert summary.get("best") is None
    assert not (tmp_path / "best_gate_passed.pt").exists()


def test_legacy_mode_never_writes_best_gate_passed_even_if_gate_passes(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "seed_7"
    seed_dir.mkdir()
    torch.save({"seed": 7, "legacy": True}, seed_dir / "last.pt")
    torch.save({"seed": 7, "legacy": True}, seed_dir / "gate_passed.pt")
    results = [
        {
            "seed": 7,
            "checkpoint": str(seed_dir / "last.pt"),
            "gate_checkpoint": str(seed_dir / "gate_passed.pt"),
            "cal_metrics": {"pixel_ap": 0.95, "equal_pixel_ap": 0.50},
            "no_regression_vs_equal": True,
            "legacy_mode": True,
            "eligible_for_evaluation": False,
            "steps": 1,
        }
    ]
    summary, code = tf.finalize_fusion_run(
        results,
        tmp_path,
        fail_if_no_gate_passes=True,
        legacy_mode=True,
    )
    assert code == 0
    assert summary["status"] == "completed"
    assert summary["best"] is None
    assert not (tmp_path / "best_gate_passed.pt").exists()


def test_calibration_metrics_use_dataset_level_paper_metrics() -> None:
    # Per-batch mean AP must not be used: flatten all pixels then PaperMetrics.
    fused = [
        torch.tensor([[[[0.1, 0.9], [0.1, 0.1]]]]),
        torch.tensor([[[[0.2, 0.8], [0.1, 0.1]]]]),
    ]
    equal = [
        torch.tensor([[[[0.9, 0.1], [0.1, 0.1]]]]),
        torch.tensor([[[[0.8, 0.2], [0.1, 0.1]]]]),
    ]
    masks = [
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
    ]
    labels = [1.0, 1.0]
    out = tf.dataset_level_calibration_metrics(
        fused_maps=fused,
        equal_maps=equal,
        masks=masks,
        image_labels=labels,
    )
    assert "pixel_ap" in out and "equal_pixel_ap" in out
    assert "paper_metrics" in out
    assert set(out["paper_metrics"]) >= {
        "image_auroc",
        "pixel_ap",
        "pixel_aupro",
    }
    # Dataset-level AP differs from naive mean of singleton APs in general;
    # at minimum the helper must call PaperMetrics (pixel_ap finite).
    assert np.isfinite(out["pixel_ap"])
    assert np.isfinite(out["equal_pixel_ap"])


def test_no_passing_seed_gate_failed_exit_5_no_best(tmp_path: Path) -> None:
    results = [
        {
            "seed": 111,
            "checkpoint": str(tmp_path / "seed_111" / "last.pt"),
            "cal_metrics": {"pixel_ap": 0.1, "equal_pixel_ap": 0.9},
            "no_regression_vs_equal": False,
            "steps": 1,
        },
        {
            "seed": 222,
            "checkpoint": str(tmp_path / "seed_222" / "last.pt"),
            "cal_metrics": {"pixel_ap": 0.2, "equal_pixel_ap": 0.9},
            "no_regression_vs_equal": False,
            "steps": 1,
        },
    ]
    for seed in (111, 222):
        d = tmp_path / f"seed_{seed}"
        d.mkdir()
        torch.save({"seed": seed}, d / "last.pt")
    summary, code = tf.finalize_fusion_run(
        results,
        tmp_path,
        fail_if_no_gate_passes=True,
        legacy_mode=False,
    )
    assert code == 5
    assert summary["status"] == "gate_failed"
    assert summary.get("best") is None
    assert not (tmp_path / "best_gate_passed.pt").exists()
    assert summary["eligible_for_evaluation"] is False
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["status"] == "gate_failed"


def test_only_passing_seeds_create_best_gate_passed(tmp_path: Path) -> None:
    pass_dir = tmp_path / "seed_111"
    fail_dir = tmp_path / "seed_222"
    pass_dir.mkdir()
    fail_dir.mkdir()
    torch.save({"seed": 111, "ok": True}, pass_dir / "last.pt")
    torch.save({"seed": 111, "ok": True}, pass_dir / "gate_passed.pt")
    torch.save({"seed": 222, "ok": False}, fail_dir / "last.pt")
    results = [
        {
            "seed": 111,
            "checkpoint": str(pass_dir / "last.pt"),
            "gate_checkpoint": str(pass_dir / "gate_passed.pt"),
            "cal_metrics": {"pixel_ap": 0.9, "equal_pixel_ap": 0.5},
            "no_regression_vs_equal": True,
            "legacy_mode": False,
            "eligible_for_evaluation": True,
            "steps": 1,
        },
        {
            "seed": 222,
            "checkpoint": str(fail_dir / "last.pt"),
            "cal_metrics": {"pixel_ap": 0.1, "equal_pixel_ap": 0.5},
            "no_regression_vs_equal": False,
            "legacy_mode": False,
            "eligible_for_evaluation": False,
            "steps": 1,
        },
    ]
    summary, code = tf.finalize_fusion_run(
        results,
        tmp_path,
        fail_if_no_gate_passes=True,
        legacy_mode=False,
    )
    assert code == 0
    assert summary["status"] == "completed"
    best = tmp_path / "best_gate_passed.pt"
    assert best.is_file()
    blob = torch.load(best, map_location="cpu")
    assert blob["seed"] == 111
    assert summary["best"]["seed"] == 111
    assert summary["eligible_for_evaluation"] is True


def test_mixed_seed_finalization_selects_only_valid_passing_seed(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "seed_111"
    failed = tmp_path / "seed_222"
    legacy = tmp_path / "seed_333"
    for d, seed in ((valid, 111), (failed, 222), (legacy, 333)):
        d.mkdir()
        torch.save({"seed": seed}, d / "last.pt")
    torch.save({"seed": 111, "tag": "valid"}, valid / "gate_passed.pt")
    torch.save({"seed": 333, "tag": "legacy"}, legacy / "gate_passed.pt")
    results = [
        {
            "seed": 111,
            "checkpoint": str(valid / "last.pt"),
            "gate_checkpoint": str(valid / "gate_passed.pt"),
            "cal_metrics": {"pixel_ap": 0.80, "equal_pixel_ap": 0.70},
            "no_regression_vs_equal": True,
            "legacy_mode": False,
            "eligible_for_evaluation": True,
            "steps": 1,
        },
        {
            "seed": 222,
            "checkpoint": str(failed / "last.pt"),
            "cal_metrics": {"pixel_ap": 0.10, "equal_pixel_ap": 0.70},
            "no_regression_vs_equal": False,
            "legacy_mode": False,
            "eligible_for_evaluation": False,
            "steps": 1,
        },
        {
            "seed": 333,
            "checkpoint": str(legacy / "last.pt"),
            "gate_checkpoint": str(legacy / "gate_passed.pt"),
            # Numerically best, but legacy → must not win.
            "cal_metrics": {"pixel_ap": 0.99, "equal_pixel_ap": 0.70},
            "no_regression_vs_equal": True,
            "legacy_mode": True,
            "eligible_for_evaluation": False,
            "steps": 1,
        },
    ]
    summary, code = tf.finalize_fusion_run(
        results,
        tmp_path,
        fail_if_no_gate_passes=True,
        legacy_mode=False,
    )
    assert code == 0
    assert summary["best"]["seed"] == 111
    best = torch.load(tmp_path / "best_gate_passed.pt", map_location="cpu")
    assert best["seed"] == 111
    assert best.get("tag") == "valid"


def test_cli_propagates_gate_failed_exit_code_5(tmp_path: Path) -> None:
    cfg_path = tmp_path / "fusion_min.yaml"
    out_dir = tmp_path / "out"
    cfg_path.write_text(
        textwrap.dedent(
            f"""\
            seed: 1
            device: cpu
            image_size: 8
            backbone:
              depth: 24
              candidate_layers: [6, 12, 18, 24]
            data:
              dataset: mvtec
              data_path: {tmp_path}
              split_manifest: {tmp_path / "split.jsonl"}
            zero_shot:
              source_dataset: mvtec
              target_datasets: [visa]
              target_tuning: false
            fusion:
              train_cache: missing
              calibration_cache: missing
              shapley_targets: missing.pt
              descriptor_stats: missing.json
              allow_missing_shapley: false
              require_descriptor_stats: true
              fail_if_no_gate_passes: true
              seeds: [1]
              epochs: 1
              output_dir: {out_dir}
            """
        )
    )
    harness = tmp_path / "harness_exit5.py"
    harness.write_text(
        textwrap.dedent(
            f"""\
            import sys
            from pathlib import Path
            import torch
            sys.path.insert(0, {str(REPO)!r})
            from tools import train_fusion as tf

            def fake_train_one_seed(**kwargs):
                seed = int(kwargs["seed"])
                d = Path(kwargs["output_dir"]) / f"seed_{{seed}}"
                d.mkdir(parents=True, exist_ok=True)
                last = d / "last.pt"
                torch.save({{"seed": seed}}, last)
                return {{
                    "seed": seed,
                    "checkpoint": str(last),
                    "cal_metrics": {{"pixel_ap": 0.1, "equal_pixel_ap": 0.9}},
                    "no_regression_vs_equal": False,
                    "legacy_mode": False,
                    "eligible_for_evaluation": False,
                    "steps": 0,
                }}

            tf.train_one_seed = fake_train_one_seed
            sys.argv = [
                "train_fusion.py",
                "--config", {str(cfg_path)!r},
                "--output-dir", {str(out_dir)!r},
                "--seeds", "1",
                "--device", "cpu",
            ]
            raise SystemExit(tf.main())
            """
        )
    )
    proc = subprocess.run(
        [sys.executable, str(harness)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 5, proc.stdout + proc.stderr
    assert not (out_dir / "best_gate_passed.pt").exists()


def test_preflight_checks_before_optimizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def boom_adam(*_a: object, **_k: object) -> None:
        calls.append("adam")
        raise AssertionError("optimizer must not run before preflight fails")

    monkeypatch.setattr(torch.optim, "Adam", boom_adam)

    fusion_cfg = {
        "shapley_targets": str(tmp_path / "missing_shapley.pt"),
        "descriptor_stats": str(tmp_path / "missing_stats.json"),
        "allow_missing_shapley": False,
        "require_descriptor_stats": True,
        "train_cache": "artifacts/cache/mvtec_teacher/train",
        "calibration_cache": "artifacts/cache/mvtec_teacher/calibration",
    }
    with pytest.raises(ArtifactIntegrityError, match="[Ss]hapley|[Dd]escriptor|[Ss]tat"):
        tf.preflight_fusion_artifacts(fusion_cfg, repo_root=tmp_path)
    assert calls == []

    # Absolute missing paths under tmp_path via repo_root-relative keys:
    fusion_cfg2 = {
        "shapley_targets": "no_shapley.pt",
        "descriptor_stats": "no_stats.json",
        "allow_missing_shapley": False,
        "require_descriptor_stats": True,
    }
    with pytest.raises(ArtifactIntegrityError, match="[Ss]hapley"):
        tf.preflight_fusion_artifacts(fusion_cfg2, repo_root=tmp_path)
    assert calls == []


def test_calibration_accumulation_detaches_to_cpu_numpy() -> None:
    fused = torch.randn(1, 2, 2, requires_grad=True)
    equal = torch.randn(1, 2, 2, requires_grad=True)
    mask = torch.zeros(1, 2, 2)
    mask[0, 0, 1] = 1.0
    # Retain references that would keep a GPU/autograd graph if not detached.
    out = tf.dataset_level_calibration_metrics(
        fused_maps=[fused],
        equal_maps=[equal],
        masks=[mask],
        image_labels=[1.0],
        aupro_steps=5,
    )
    assert np.isfinite(out["pixel_ap"])
    # Helper must consume detached CPU arrays internally (no retained graph).
    cpu_maps = tf.detach_maps_to_cpu_numpy([fused, equal])
    assert all(isinstance(a, np.ndarray) for a in cpu_maps)
    assert all(a.flags.writeable for a in cpu_maps)
    # Original tensors may still require grad; converted arrays must be plain ndarray.
    assert fused.requires_grad
    assert not isinstance(cpu_maps[0], torch.Tensor)


def test_gate_thresholds_use_unit_interval_not_percentage_points() -> None:
    """Metric drops are fractional on [0,1]; 0.2 pp == 0.002."""
    joint = yaml.safe_load((REPO / "configs/rad/joint.yaml").read_text())
    nr = joint["joint"]["no_regression"]
    assert nr["max_pixel_ap_drop"] == pytest.approx(0.002)
    assert nr["max_pro_drop"] == pytest.approx(0.002)
    assert 0.0 <= float(nr["max_mean_error_relative_increase"]) <= 1.0

    policy = yaml.safe_load((REPO / "configs/rad/policy.yaml").read_text())["policy"]
    # 0.03 == 3 percentage points in unit-interval metrics.
    assert policy["max_pixel_ap_drop"] == pytest.approx(0.03)
    assert 0.0 <= float(policy["max_pixel_ap_drop"]) <= 1.0

    fusion = yaml.safe_load((REPO / "configs/rad/fusion.yaml").read_text())["fusion"]
    # Fusion no-regression uses unit-interval pixel_ap comparison; optional drop
    # threshold (if present) must also be fractional, not percentage points.
    drop = float(fusion.get("max_pixel_ap_drop", 0.0))
    assert 0.0 <= drop <= 1.0
    assert drop != pytest.approx(0.2), "0.2 would be 20pp; encode 0.2pp as 0.002"


def test_fusion_yaml_defaults_are_fail_closed() -> None:
    raw = yaml.safe_load((REPO / "configs/rad/fusion.yaml").read_text())
    fusion = raw["fusion"]
    assert fusion.get("allow_missing_shapley") is False
    assert fusion.get("require_descriptor_stats") is True
    assert fusion.get("fail_if_no_gate_passes") is True
