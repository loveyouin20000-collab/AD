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
class ExperimentConfig:
    seed: int
    backbone: BackboneConfig
    zero_shot: ZeroShotConfig

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
        return cls(seed=int(raw["seed"]), backbone=backbone, zero_shot=zero_shot)
