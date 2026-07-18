from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


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


@dataclass(frozen=True)
class TeacherConfig:
    checkpoint_path: Path
    backbone: str = "ViT-L/14@336px"


@dataclass(frozen=True)
class CacheConfig:
    schema_version: int = 1
    shard_size: int = 16


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

        return cls(
            seed=int(raw["seed"]),
            backbone=backbone,
            zero_shot=zero_shot,
            device=str(raw.get("device", "cuda:0")),
            image_size=int(raw.get("image_size", 518)),
            data=data,
            teacher=teacher,
            cache=cache,
        )
