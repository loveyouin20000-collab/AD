from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def deep_merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_config(out[key], value)
        else:
            out[key] = value
    return out


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_effective_config(
    base_path: Path,
    overlay_path: Path | None = None,
) -> dict[str, Any]:
    base = load_yaml_mapping(base_path)
    if overlay_path is None:
        return base
    overlay = load_yaml_mapping(overlay_path)
    return deep_merge_config(base, overlay)


@dataclass(frozen=True)
class AdaptiveConfigIdentity:
    base_config_sha256: str
    overlay_sha256: str | None
    effective_config_sha256: str

    def as_manifest_fields(self) -> dict[str, str]:
        fields = {
            "base_config_sha256": self.base_config_sha256,
            "effective_config_sha256": self.effective_config_sha256,
            # Backward-compatible alias: experiment identity follows the merged config.
            "config_sha256": self.effective_config_sha256,
        }
        if self.overlay_sha256 is not None:
            fields["overlay_sha256"] = self.overlay_sha256
        return fields


def adaptive_config_identity(
    base_path: Path,
    *,
    overlay_path: Path | None = None,
) -> AdaptiveConfigIdentity:
    base_config_sha256 = sha256_file(base_path)
    overlay_sha256 = sha256_file(overlay_path) if overlay_path is not None else None
    effective = merge_effective_config(base_path, overlay_path)
    return AdaptiveConfigIdentity(
        base_config_sha256=base_config_sha256,
        overlay_sha256=overlay_sha256,
        effective_config_sha256=canonical_config_sha256(effective),
    )
