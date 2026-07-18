from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PAPER_TABLE_IDS: tuple[str, ...] = (
    "main_comparison",
    "fusion_ablation",
    "exit_strategy_comparison",
    "selector_ablation",
    "risk_coverage",
    "oracle_gap",
    "zero_shot_transfer",
)

TABLE_EXPERIMENT_IDS: dict[str, tuple[str, ...]] = {
    "main_comparison": (
        "original_visualad",
        "fixed_exit_12",
        "fixed_exit_18",
        "static_learned_fusion",
        "dynamic_fusion_only",
        "full_conservative",
        "full_balanced",
        "full_aggressive",
    ),
    "fusion_ablation": (
        "ablation_dlcm_with_shapley",
        "ablation_dlcm_without_shapley",
        "ablation_shapley_exact",
        "ablation_shapley_loo",
        "ablation_kd_none",
        "ablation_kd_map",
        "ablation_kd_map_boundary",
        "ablation_training_staged",
        "ablation_training_joint",
    ),
    "exit_strategy_comparison": (
        "confidence_only_exit",
        "stability_only_exit",
        "confidence_stability_exit",
        "residual_gain_equal_fusion",
        "random_exit_matched",
        "fixed_exit_12",
        "fixed_exit_18",
        "full_balanced",
    ),
    "selector_ablation": (
        "ablation_selector_cumulative",
        "ablation_selector_loo",
        "residual_gain_equal_fusion",
        "confidence_only_exit",
        "stability_only_exit",
    ),
    "risk_coverage": (
        "full_conservative",
        "full_balanced",
        "full_aggressive",
        "random_exit_matched",
        "oracle_earliest_exit",
    ),
    "oracle_gap": (
        "oracle_earliest_exit",
        "full_balanced",
        "full_conservative",
        "full_aggressive",
        "random_exit_matched",
    ),
    "zero_shot_transfer": (
        "zero_shot_transfer",
        "full_balanced",
        "original_visualad",
    ),
}

METRIC_COLUMNS: tuple[str, ...] = (
    "method_id",
    "pixel_ap",
    "pro",
    "pixel_ap_adaptive",
    "pixel_ap_full",
    "pro_adaptive",
    "pro_full",
    "expected_depth",
    "false_safe_exit_rate",
    "status",
)

REQUIRED_RELEASE_MANIFEST_KEYS: tuple[str, ...] = (
    "schema_version",
    "git_sha",
    "seed",
    "configs",
    "hashes",
    "environment",
    "checkpoints",
    "result_files",
    "paper_tables",
    "tag_recommendation",
)


def load_result_summaries(results_dir: Path | str) -> dict[str, dict[str, Any]]:
    root = Path(results_dir)
    summaries: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return summaries
    for summary_path in sorted(root.rglob("summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        method_id = str(payload.get("method_id") or summary_path.parent.name)
        row = dict(payload)
        row.setdefault("method_id", method_id)
        row.setdefault("status", payload.get("status", "present"))
        summaries[method_id] = row
    return summaries


def _metric_value(row: Mapping[str, Any] | None, key: str) -> str:
    if row is None:
        return ""
    if key == "method_id":
        return str(row.get("method_id", ""))
    if key == "status":
        return str(row.get("status", "present"))
    value = row.get(key)
    if value is None:
        # Fallbacks for common aliases
        aliases = {
            "pixel_ap": ("pixel_ap_adaptive", "pixel_ap_full"),
            "pro": ("pro_adaptive", "pro_full"),
        }
        for alias in aliases.get(key, ()):
            if alias in row and row[alias] is not None:
                value = row[alias]
                break
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _rows_for_table(
    table_id: str, summaries: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, str]]:
    method_ids = TABLE_EXPERIMENT_IDS[table_id]
    rows: list[dict[str, str]] = []
    for method_id in method_ids:
        raw = summaries.get(method_id)
        if raw is None:
            rows.append(
                {
                    "method_id": method_id,
                    "pixel_ap": "",
                    "pro": "",
                    "pixel_ap_adaptive": "",
                    "pixel_ap_full": "",
                    "pro_adaptive": "",
                    "pro_full": "",
                    "expected_depth": "",
                    "false_safe_exit_rate": "",
                    "status": "missing",
                }
            )
            continue
        rows.append({col: _metric_value(raw, col) for col in METRIC_COLUMNS})
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    lines = [",".join(METRIC_COLUMNS)]
    for row in rows:
        lines.append(",".join(row.get(col, "") for col in METRIC_COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _write_latex(path: Path, table_id: str, rows: Sequence[Mapping[str, str]]) -> None:
    header = " & ".join(_latex_escape(c) for c in METRIC_COLUMNS) + " \\\\"
    body_lines = []
    for row in rows:
        body_lines.append(
            " & ".join(_latex_escape(row.get(col, "")) for col in METRIC_COLUMNS) + " \\\\"
        )
    content = "\n".join(
        [
            "% Auto-generated by tools/export_paper_tables.py — do not edit by hand",
            f"% table_id={table_id}",
            "\\begin{tabular}{" + ("l" * len(METRIC_COLUMNS)) + "}",
            "\\toprule",
            header,
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_manifest(
    *,
    results_dir: Path | str,
    output_dir: Path | str,
    git_sha: str,
    config_paths: Sequence[Path | str],
    checkpoint_paths: Sequence[Path | str],
    environment: Mapping[str, Any],
    seed: int = 111,
) -> dict[str, Any]:
    out = Path(output_dir)
    results = Path(results_dir)
    configs = [str(Path(p)) for p in config_paths]
    checkpoints = [str(Path(p)) for p in checkpoint_paths if Path(p).exists()]
    result_files = sorted(str(p) for p in out.glob("*") if p.is_file())
    if results.is_dir():
        result_files.extend(sorted(str(p) for p in results.rglob("summary.json")))
    hashes: dict[str, str] = {}
    for path_str in [*configs, *checkpoints, *result_files]:
        path = Path(path_str)
        if path.is_file():
            hashes[path_str] = _sha256_file(path)
    paper_tables = {
        table_id: {
            "csv": str(out / f"{table_id}.csv"),
            "tex": str(out / f"{table_id}.tex"),
        }
        for table_id in PAPER_TABLE_IDS
        if (out / f"{table_id}.csv").exists() or (out / f"{table_id}.tex").exists()
    }
    return {
        "schema_version": "rad-release-manifest-v1",
        "git_sha": git_sha,
        "seed": int(seed),
        "configs": configs,
        "hashes": hashes,
        "environment": dict(environment),
        "checkpoints": checkpoints,
        "result_files": result_files,
        "paper_tables": paper_tables,
        "tag_recommendation": "cvpr-rad-visualad-v2",
    }


def export_paper_tables(
    *,
    results_dir: Path | str,
    output_dir: Path | str,
    git_sha: str = "unknown",
    seed: int = 111,
    config_paths: Sequence[Path | str] | None = None,
    checkpoint_paths: Sequence[Path | str] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    results = Path(results_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = load_result_summaries(results)
    artifacts: dict[str, dict[str, str]] = {}
    for table_id in PAPER_TABLE_IDS:
        rows = _rows_for_table(table_id, summaries)
        csv_path = out / f"{table_id}.csv"
        tex_path = out / f"{table_id}.tex"
        _write_csv(csv_path, rows)
        _write_latex(tex_path, table_id, rows)
        artifacts[table_id] = {"csv": str(csv_path), "tex": str(tex_path)}

    configs = list(config_paths or [])
    checkpoints = list(checkpoint_paths or [])
    env = dict(environment or {})
    manifest = build_release_manifest(
        results_dir=results,
        output_dir=out,
        git_sha=git_sha,
        config_paths=configs,
        checkpoint_paths=checkpoints,
        environment=env,
        seed=seed,
    )
    (out / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifacts
