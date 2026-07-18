from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

JOINT_MAX_LEARNING_RATE = 1e-5


@dataclass(frozen=True)
class BackboneConfig:
    depth: int
    candidate_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.candidate_layers))) != self.candidate_layers:
            raise ValueError("candidate_layers must be strictly increasing and unique")
        if not self.candidate_layers or self.candidate_layers[-1] != self.depth:
            raise ValueError("the final candidate layer must equal backbone depth")


@dataclass(frozen=True)
class ZeroShotConfig:
    source_dataset: str
    target_datasets: tuple[str, ...]
    target_tuning: bool = False


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    data_path: Path
    split_manifest: Path
    train_cache: Path | None = None
    calibration_cache: Path | None = None
    descriptor_stats: Path | None = None
    shapley_targets: Path | None = None


@dataclass(frozen=True)
class TeacherConfig:
    checkpoint_path: Path
    backbone: str = "ViT-L/14@336px"


@dataclass(frozen=True)
class CacheConfig:
    schema_version: int = 1
    shard_size: int = 16


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    amp: bool
    validation_interval_epochs: int
    early_stopping_patience: int

    def __post_init__(self) -> None:
        if self.mode not in {"staged", "joint"}:
            raise ValueError(f"unsupported training.mode: {self.mode}")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.mode == "joint" and self.learning_rate > JOINT_MAX_LEARNING_RATE:
            raise ValueError(
                f"joint training learning_rate must not exceed {JOINT_MAX_LEARNING_RATE}"
            )


@dataclass(frozen=True)
class JointNoRegressionConfig:
    max_pixel_ap_drop: float
    max_pro_drop: float
    max_mean_error_relative_increase: float

    def __post_init__(self) -> None:
        if self.max_pixel_ap_drop < 0.0:
            raise ValueError("max_pixel_ap_drop must be nonnegative")
        if self.max_pro_drop < 0.0:
            raise ValueError("max_pro_drop must be nonnegative")
        if self.max_mean_error_relative_increase < 0.0:
            raise ValueError("max_mean_error_relative_increase must be nonnegative")


@dataclass(frozen=True)
class JointConfig:
    enabled: bool
    primary_pipeline: bool
    data_mode: str
    trainable_modules: tuple[str, ...]
    fusion_loss_weight: float
    lse_loss_weight: float
    compute_final_weight: float
    compute_ramp_fraction: float
    compute_target_depth_ratio: float
    soft_exit_temperature: float
    no_regression: JointNoRegressionConfig
    early_depths: tuple[int, ...] = (12, 18)
    full_depth: int = 24
    epsilon_gain: float = 0.05
    epsilon_absolute: float = 0.5
    sufficiency_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.data_mode != "cached":
            raise ValueError("joint.data_mode must be cached")
        if self.trainable_modules != ("dlcm", "lse"):
            raise ValueError("joint.trainable_modules must be exactly ('dlcm', 'lse')")
        if not 0.0 < self.compute_ramp_fraction <= 1.0:
            raise ValueError("compute_ramp_fraction must be in (0, 1]")
        if self.compute_final_weight < 0.0:
            raise ValueError("compute_final_weight must be nonnegative")
        if self.soft_exit_temperature <= 0.0:
            raise ValueError("soft_exit_temperature must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    backbone: BackboneConfig
    zero_shot: ZeroShotConfig
    device: str = "cuda:0"
    image_size: int = 518
    data: DataConfig | None = None
    teacher: TeacherConfig | None = None
    cache: CacheConfig = CacheConfig()
    training: TrainingConfig | None = None
    joint: JointConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        backbone = BackboneConfig(
            depth=int(raw["backbone"]["depth"]),
            candidate_layers=tuple(raw["backbone"]["candidate_layers"]),
        )
        zero_shot = ZeroShotConfig(
            source_dataset=str(raw["zero_shot"]["source_dataset"]),
            target_datasets=tuple(raw["zero_shot"]["target_datasets"]),
            target_tuning=bool(raw["zero_shot"].get("target_tuning", False)),
        )
        if zero_shot.target_tuning:
            raise ValueError("target-domain tuning is forbidden")

        data = None
        if "data" in raw:
            data_raw = raw["data"]
            data = DataConfig(
                dataset=str(data_raw["dataset"]),
                data_path=Path(data_raw["data_path"]),
                split_manifest=Path(data_raw["split_manifest"]),
                train_cache=(
                    Path(data_raw["train_cache"]) if "train_cache" in data_raw else None
                ),
                calibration_cache=(
                    Path(data_raw["calibration_cache"])
                    if "calibration_cache" in data_raw
                    else None
                ),
                descriptor_stats=(
                    Path(data_raw["descriptor_stats"])
                    if "descriptor_stats" in data_raw
                    else None
                ),
                shapley_targets=(
                    Path(data_raw["shapley_targets"])
                    if "shapley_targets" in data_raw
                    else None
                ),
            )

        teacher = None
        if "teacher" in raw:
            teacher_raw = raw["teacher"]
            teacher = TeacherConfig(
                checkpoint_path=Path(teacher_raw["checkpoint_path"]),
                backbone=str(teacher_raw.get("backbone", "ViT-L/14@336px")),
            )

        cache_raw = raw.get("cache", {}) or {}
        cache = CacheConfig(
            schema_version=int(cache_raw.get("schema_version", 1)),
            shard_size=int(cache_raw.get("shard_size", 16)),
        )

        training = None
        if "training" in raw:
            t = raw["training"]
            training = TrainingConfig(
                mode=str(t["mode"]),
                epochs=int(t["epochs"]),
                batch_size=int(t["batch_size"]),
                learning_rate=float(t["learning_rate"]),
                weight_decay=float(t.get("weight_decay", 0.0)),
                num_workers=int(t.get("num_workers", 0)),
                amp=bool(t.get("amp", False)),
                validation_interval_epochs=int(t.get("validation_interval_epochs", 1)),
                early_stopping_patience=int(t.get("early_stopping_patience", 5)),
            )

        joint = None
        if "joint" in raw:
            j = raw["joint"]
            nr = j.get("no_regression", {}) or {}
            joint = JointConfig(
                enabled=bool(j["enabled"]),
                primary_pipeline=bool(j["primary_pipeline"]),
                data_mode=str(j["data_mode"]),
                trainable_modules=tuple(str(x) for x in j["trainable_modules"]),
                fusion_loss_weight=float(j["fusion_loss_weight"]),
                lse_loss_weight=float(j["lse_loss_weight"]),
                compute_final_weight=float(j["compute_final_weight"]),
                compute_ramp_fraction=float(j["compute_ramp_fraction"]),
                compute_target_depth_ratio=float(j["compute_target_depth_ratio"]),
                soft_exit_temperature=float(j["soft_exit_temperature"]),
                no_regression=JointNoRegressionConfig(
                    max_pixel_ap_drop=float(nr["max_pixel_ap_drop"]),
                    max_pro_drop=float(nr["max_pro_drop"]),
                    max_mean_error_relative_increase=float(
                        nr["max_mean_error_relative_increase"]
                    ),
                ),
                early_depths=tuple(int(d) for d in j.get("early_depths", (12, 18))),
                full_depth=int(j.get("full_depth", backbone.depth)),
                epsilon_gain=float(j.get("epsilon_gain", 0.05)),
                epsilon_absolute=float(j.get("epsilon_absolute", 0.5)),
                sufficiency_weight=float(j.get("sufficiency_weight", 0.5)),
            )
            if training is not None and training.mode == "joint":
                if not joint.enabled:
                    raise ValueError("joint.enabled must be true when training.mode is joint")
                if joint.primary_pipeline:
                    raise ValueError("joint.primary_pipeline must be false for joint ablation")

        return cls(
            seed=int(raw["seed"]),
            backbone=backbone,
            zero_shot=zero_shot,
            device=str(raw.get("device", "cuda:0")),
            image_size=int(raw.get("image_size", 518)),
            data=data,
            teacher=teacher,
            cache=cache,
            training=training,
            joint=joint,
        )
