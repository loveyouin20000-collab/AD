from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.calibration.policy_search import search_policy_profiles  # noqa: E402
from rad.calibration.temperature import fit_temperature  # noqa: E402
from rad.config import ExperimentConfig  # noqa: E402
from rad.models.policy import PolicyProfile  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate map temperature and exit policies (source only)")
    p.add_argument("--config", type=str, default="configs/rad/policy.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--calibration-cache", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def assert_no_target_dataset_path(
    path: Path | str,
    *,
    source_dataset: str,
    target_datasets: tuple[str, ...],
) -> None:
    """Refuse calibration paths that mention a target dataset name."""
    text = str(path).replace("\\", "/").lower()
    source = source_dataset.lower()
    for tgt in target_datasets:
        name = str(tgt).lower()
        if not name:
            continue
        # Path segment contains target name and is not the source dataset
        if name in text.split("/") and name != source:
            raise SystemExit(
                f"refusing target-dataset path for calibration: {path} (target={tgt})"
            )
        # Also catch patterns like visa_target_cache
        if f"{name}_" in text or f"_{name}" in text or f"/{name}" in text:
            if name != source:
                raise SystemExit(
                    f"refusing target-dataset path for calibration: {path} (target={tgt})"
                )


def _profile_to_dict(p: PolicyProfile) -> dict[str, Any]:
    return {
        "name": p.name,
        "gain_threshold": p.gain_threshold,
        "kappa": p.kappa,
        "map_uncertainty_threshold": p.map_uncertainty_threshold,
        "image_confidence_margin": p.image_confidence_margin,
        "stability_threshold": p.stability_threshold,
        "require_map_uncertainty": p.require_map_uncertainty,
        "require_image_confidence": p.require_image_confidence,
        "require_stability": p.require_stability,
    }


def _synthetic_candidate_grid(policy_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a searchable candidate table with placeholder metrics for dry/offline use.

    Full adaptive-engine metrics are filled in Task 18/19; here we emit a constrained
    grid so profiles can be selected under explicit AP-drop / false-safe caps.
    """
    gains = [float(x) for x in policy_cfg.get("gain_thresholds", [0.05, 0.1])]
    kappas = [float(x) for x in policy_cfg.get("kappas", [1.0])]
    map_ts = [float(x) for x in policy_cfg.get("map_uncertainty_thresholds", [0.5])]
    margins = [float(x) for x in policy_cfg.get("image_confidence_margins", [0.3])]
    stabs = [float(x) for x in policy_cfg.get("stability_thresholds", [0.1])]
    rows: list[dict[str, Any]] = []
    for i, g in enumerate(gains):
        for j, k in enumerate(kappas):
            for m in map_ts:
                for margin in margins:
                    for stab in stabs:
                        # Monotone heuristic: stricter thresholds -> smaller AP drop / FSE, deeper expected depth
                        strict = (0.2 - g) + 0.1 * k + (0.8 - m) + margin + (0.3 - stab)
                        rows.append(
                            {
                                "gain_threshold": g,
                                "kappa": k,
                                "map_uncertainty_threshold": m,
                                "image_confidence_margin": margin,
                                "stability_threshold": stab,
                                "pixel_ap_drop": max(0.0, 0.04 - 0.01 * strict),
                                "false_safe_exit_rate": max(0.0, 0.12 - 0.02 * strict),
                                "expected_depth": 12.0 + 2.0 * strict,
                            }
                        )
    return rows


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = ExperimentConfig.from_yaml(args.config)
    policy_cfg = dict(raw.get("policy", {}))

    seed = args.seed if args.seed is not None else cfg.seed
    torch.manual_seed(seed)
    device = torch.device(args.device or raw.get("device", cfg.device))

    cal_cache = args.calibration_cache or Path(policy_cfg.get("calibration_cache", ""))
    if not Path(cal_cache).is_absolute():
        cal_cache = REPO_ROOT / cal_cache

    assert_no_target_dataset_path(
        cal_cache,
        source_dataset=cfg.zero_shot.source_dataset,
        target_datasets=cfg.zero_shot.target_datasets,
    )

    output_dir = args.output_dir or Path(policy_cfg.get("output_dir", "artifacts/calibration/policy"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    config_hash = sha256_file(Path(args.config))
    sha = git_sha()
    print(f"config: {args.config}")
    print(f"config_hash: {config_hash}")
    print(f"git_sha: {sha}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"calibration_cache: {cal_cache}")
    print(f"output_dir: {output_dir}")
    print(f"source_dataset: {cfg.zero_shot.source_dataset}")
    print(f"target_datasets: {cfg.zero_shot.target_datasets}")

    if args.dry_run:
        return 0

    max_ap_drop = float(policy_cfg.get("max_pixel_ap_drop", 0.03))
    max_fse = float(policy_cfg.get("max_false_safe_exit_rate", 0.1))
    candidates = _synthetic_candidate_grid(policy_cfg)
    result = search_policy_profiles(
        candidates,
        max_pixel_ap_drop=max_ap_drop,
        max_false_safe_exit_rate=max_fse,
    )

    # Fit a smoke temperature on synthetic calibration logits (shape log)
    logits = torch.randn(32, 1, 8, 8)
    labels = (torch.sigmoid(logits) > 0.5).float()
    temperature = fit_temperature(logits, labels)
    print(f"temperature: {temperature}")
    print(f"tensor_shapes logits={tuple(logits.shape)} labels={tuple(labels.shape)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = {k: _profile_to_dict(v) for k, v in result["profiles"].items()}
    payload = {
        "schema_version": 1,
        "seed": seed,
        "config_hash": config_hash,
        "git_sha": sha,
        "temperature": temperature,
        "constraints": {
            "max_pixel_ap_drop": max_ap_drop,
            "max_false_safe_exit_rate": max_fse,
        },
        "n_feasible": len(result["feasible"]),
        "n_pareto": len(result["pareto"]),
        "pareto": result["pareto"],
        "profiles": profiles,
        "note": "Candidate metrics are grid heuristics until adaptive-engine eval tables exist.",
    }
    (output_dir / "policy_profiles.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "pareto.json").write_text(json.dumps(result["pareto"], indent=2) + "\n")
    print(json.dumps({"profiles": profiles, "n_pareto": len(result["pareto"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
