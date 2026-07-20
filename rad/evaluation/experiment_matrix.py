from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REQUIRED_METHODS: tuple[str, ...] = (
    "original_visualad",
    "fixed_exit_12_equal",
    "fixed_exit_18_equal",
    "fixed_exit_12_dynamic",
    "fixed_exit_18_dynamic",
    "static_learned_fusion",
    "dynamic_fusion_only",
    "confidence_only_exit",
    "stability_only_exit",
    "confidence_stability_exit",
    "residual_gain_equal_fusion",
    "full_conservative",
    "full_balanced",
    "full_aggressive",
    "random_exit_matched",
    "oracle_earliest_exit",
)

REQUIRED_ABLATIONS: tuple[str, ...] = (
    "ablation_dlcm_with_shapley",
    "ablation_dlcm_without_shapley",
    "ablation_shapley_exact",
    "ablation_shapley_loo",
    "selector_full",
    "selector_without_response",
    "selector_without_uncertainty",
    "selector_without_stability",
    "selector_without_complementarity",
    "selector_without_token_separation",
    "ablation_kd_none",
    "ablation_kd_map",
    "ablation_kd_map_boundary",
    "ablation_training_staged",
    "ablation_training_joint",
)

REQUIRED_ROW_KEYS: tuple[str, ...] = (
    "id",
    "group",
    "description",
    "immutable",
    "seed",
    "device",
    "command",
    "config",
    "estimated_gpu_hours",
    "output_dir",
)


@dataclass(frozen=True)
class MatrixRow:
    id: str
    group: str
    description: str
    immutable: bool
    seed: int
    device: str
    command: str
    config: dict[str, Any]
    estimated_gpu_hours: float
    output_dir: str
    tags: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def config_hash(self) -> str:
        payload = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ExperimentMatrix:
    defaults: dict[str, Any]
    rows: tuple[MatrixRow, ...]
    schema_version: int = 1

    def row_by_id(self, row_id: str) -> MatrixRow:
        for row in self.rows:
            if row.id == row_id:
                return row
        raise KeyError(row_id)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def validate_row_immutable(row: MatrixRow) -> None:
    if not row.immutable:
        raise ValueError(f"row {row.id} must set immutable: true")
    if not row.id:
        raise ValueError("row id must be non-empty")
    if row.group not in {"method", "ablation"}:
        raise ValueError(f"row {row.id} group must be method|ablation")
    if not row.command.strip():
        raise ValueError(f"row {row.id} missing command")
    if float(row.estimated_gpu_hours) < 0:
        raise ValueError(f"row {row.id} estimated_gpu_hours must be >= 0")
    cfg = row.config
    for key in ("seed", "backbone", "zero_shot", "method"):
        if key not in cfg:
            raise ValueError(f"row {row.id} config missing required key: {key}")
    zs = cfg["zero_shot"]
    if bool(zs.get("target_tuning", False)):
        raise ValueError(f"row {row.id} forbids target_tuning")
    bb = cfg["backbone"]
    if "candidate_layers" not in bb or "depth" not in bb:
        raise ValueError(f"row {row.id} backbone incomplete")
    method = cfg["method"]
    if "name" not in method:
        raise ValueError(f"row {row.id} method.name required")
    if "output_dir" not in cfg and not row.output_dir:
        raise ValueError(f"row {row.id} missing output_dir")


def load_experiment_matrix(path: str | Path) -> ExperimentMatrix:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("experiments.yaml must be a mapping")
    defaults = dict(raw.get("defaults", {}) or {})
    rows_raw = raw.get("matrix") or raw.get("rows") or []
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("experiments.yaml must define a non-empty matrix list")

    rows: list[MatrixRow] = []
    for item in rows_raw:
        if not isinstance(item, dict):
            raise ValueError("each matrix row must be a mapping")
        for key in REQUIRED_ROW_KEYS:
            if key not in item:
                raise ValueError(f"matrix row missing key: {key}")
        cfg = _deep_merge(defaults, dict(item["config"]))
        # Freeze identity fields into the immutable config snapshot
        cfg = _deep_merge(
            cfg,
            {
                "seed": int(item["seed"]),
                "device": str(item["device"]),
                "output_dir": str(item["output_dir"]),
                "matrix_row_id": str(item["id"]),
            },
        )
        row = MatrixRow(
            id=str(item["id"]),
            group=str(item["group"]),
            description=str(item["description"]),
            immutable=bool(item["immutable"]),
            seed=int(item["seed"]),
            device=str(item["device"]),
            command=str(item["command"]),
            config=cfg,
            estimated_gpu_hours=float(item["estimated_gpu_hours"]),
            output_dir=str(item["output_dir"]),
            tags=tuple(str(t) for t in item.get("tags", []) or []),
            raw=dict(item),
        )
        rows.append(row)

    return ExperimentMatrix(
        defaults=defaults,
        rows=tuple(rows),
        schema_version=int(raw.get("schema_version", 1)),
    )


def estimate_gpu_hours(
    matrix: ExperimentMatrix,
    *,
    num_gpus: int = 1,
    row_ids: list[str] | None = None,
) -> dict[str, Any]:
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    selected = matrix.rows
    if row_ids is not None:
        id_set = set(row_ids)
        selected = tuple(r for r in matrix.rows if r.id in id_set)
    total = float(sum(r.estimated_gpu_hours for r in selected))
    return {
        "num_rows": len(selected),
        "num_gpus": int(num_gpus),
        "total_gpu_hours": total,
        "wall_clock_hours_est": total / float(num_gpus),
        "per_row_gpu_hours": {r.id: float(r.estimated_gpu_hours) for r in selected},
    }


def assign_devices(
    matrix: ExperimentMatrix,
    *,
    num_gpus: int,
    row_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Round-robin CUDA device assignment for independent matrix rows."""
    selected = list(matrix.rows)
    if row_ids is not None:
        id_set = set(row_ids)
        selected = [r for r in selected if r.id in id_set]
    plans: list[dict[str, Any]] = []
    for i, row in enumerate(selected):
        gpu = i % max(1, int(num_gpus))
        device = f"cuda:{gpu}"
        cmd = row.command.replace("{device}", device).replace("{output_dir}", row.output_dir)
        cmd = cmd.replace("{seed}", str(row.seed)).replace("{row_id}", row.id)
        plans.append(
            {
                "id": row.id,
                "group": row.group,
                "device": device,
                "estimated_gpu_hours": row.estimated_gpu_hours,
                "command": cmd,
                "config_hash": row.config_hash(),
                "output_dir": row.output_dir,
            }
        )
    return plans
