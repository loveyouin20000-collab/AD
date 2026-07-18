from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "rad-checkpoint-v1"


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointManifestV1:
    schema_version: str
    stage: str
    status: str
    checkpoint_sha256: str
    candidate_layers: tuple[int, ...]
    source_dataset: str
    split_manifest_hash: str
    preprocessing_hash: str
    teacher_checkpoint_hash: str
    descriptor_stats_hash: str
    upstream_fusion_checkpoint_hash: str | None
    gates: Mapping[str, bool]
    reference_full_depth_metrics: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if self.stage not in {"fusion", "lse", "joint"}:
            raise ValueError(f"unsupported stage: {self.stage}")
        if self.status not in {"passed", "failed", "diagnostic"}:
            raise ValueError(f"unsupported status: {self.status}")
        if not self.gates.get("source_only_selection", False):
            raise ValueError("gates.source_only_selection must be true")
        if self.stage == "fusion":
            if self.reference_full_depth_metrics is None:
                raise ValueError("fusion manifest requires reference_full_depth_metrics")
            required = {"pixel_ap", "pro", "mean_sample_error"}
            missing = required - set(self.reference_full_depth_metrics)
            if missing:
                raise ValueError(f"missing reference metrics: {sorted(missing)}")
        if self.stage == "lse" and not self.upstream_fusion_checkpoint_hash:
            raise ValueError("lse manifest requires upstream_fusion_checkpoint_hash")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_layers"] = list(self.candidate_layers)
        payload["gates"] = dict(self.gates)
        if self.reference_full_depth_metrics is not None:
            payload["reference_full_depth_metrics"] = {
                k: float(v) for k, v in self.reference_full_depth_metrics.items()
            }
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CheckpointManifestV1:
        ref = raw.get("reference_full_depth_metrics")
        return cls(
            schema_version=str(raw["schema_version"]),
            stage=str(raw["stage"]),
            status=str(raw["status"]),
            checkpoint_sha256=str(raw["checkpoint_sha256"]),
            candidate_layers=tuple(int(x) for x in raw["candidate_layers"]),
            source_dataset=str(raw["source_dataset"]),
            split_manifest_hash=str(raw["split_manifest_hash"]),
            preprocessing_hash=str(raw["preprocessing_hash"]),
            teacher_checkpoint_hash=str(raw["teacher_checkpoint_hash"]),
            descriptor_stats_hash=str(raw["descriptor_stats_hash"]),
            upstream_fusion_checkpoint_hash=(
                None
                if raw.get("upstream_fusion_checkpoint_hash") in (None, "")
                else str(raw["upstream_fusion_checkpoint_hash"])
            ),
            gates={str(k): bool(v) for k, v in dict(raw["gates"]).items()},
            reference_full_depth_metrics=(
                None
                if ref is None
                else {str(k): float(v) for k, v in dict(ref).items()}
            ),
        )


def write_checkpoint_with_manifest(
    checkpoint_path: Path | str,
    manifest: CheckpointManifestV1,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual = sha256_file(checkpoint_path)
    if actual != manifest.checkpoint_sha256:
        raise ValueError(
            f"checkpoint_sha256 mismatch: manifest={manifest.checkpoint_sha256} file={actual}"
        )
    side = checkpoint_path.with_suffix(".manifest.json")
    side.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return side


def load_manifest(path: Path | str) -> CheckpointManifestV1:
    raw = json.loads(Path(path).read_text())
    return CheckpointManifestV1.from_dict(raw)


def validate_manifest_against_file(
    checkpoint_path: Path | str,
    manifest: CheckpointManifestV1,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual = sha256_file(checkpoint_path)
    if actual != manifest.checkpoint_sha256:
        raise ValueError(
            f"checkpoint_sha256 mismatch: expected {manifest.checkpoint_sha256}, got {actual}"
        )
