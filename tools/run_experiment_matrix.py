from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.evaluation.experiment_matrix import (  # noqa: E402
    assign_devices,
    estimate_gpu_hours,
    load_experiment_matrix,
    validate_row_immutable,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run or dry-run the paper experiment matrix")
    p.add_argument("--config", type=str, default="configs/rad/experiments.yaml")
    p.add_argument("--seed", type=int, default=None, help="Override seed for all selected rows")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-gpus", type=int, default=None, help="GPUs for wall-clock estimate / assignment")
    p.add_argument("--ids", type=str, default=None, help="Comma-separated row ids")
    p.add_argument("--group", type=str, default=None, choices=["method", "ablation"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", type=str, default=None)
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


def detect_num_gpus() -> int:
    try:
        import torch

        return max(1, int(torch.cuda.device_count()))
    except Exception:
        return 1


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path

    matrix = load_experiment_matrix(cfg_path)
    for row in matrix.rows:
        validate_row_immutable(row)

    row_ids = None
    if args.ids:
        row_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    selected = list(matrix.rows)
    if row_ids is not None:
        id_set = set(row_ids)
        selected = [r for r in matrix.rows if r.id in id_set]
        missing = id_set - {r.id for r in selected}
        if missing:
            raise SystemExit(f"unknown row ids: {sorted(missing)}")
    if args.group:
        selected = [r for r in selected if r.group == args.group]
    selected_ids = [r.id for r in selected]

    num_gpus = args.num_gpus if args.num_gpus is not None else detect_num_gpus()
    if num_gpus < 1:
        raise SystemExit("--num-gpus must be >= 1")

    estimates = estimate_gpu_hours(matrix, num_gpus=num_gpus, row_ids=selected_ids)
    plans = assign_devices(matrix, num_gpus=num_gpus, row_ids=selected_ids)
    if args.device:
        for p in plans:
            p["device"] = args.device
            p["command"] = p["command"].replace("cuda:0", args.device).replace("cuda:1", args.device)

    if args.seed is not None:
        for p in plans:
            p["command"] = p["command"].replace(f"--seed {matrix.row_by_id(p['id']).seed}", f"--seed {args.seed}")
            if "--seed " not in p["command"]:
                p["command"] += f" --seed {args.seed}"

    output_dir = args.output_dir or Path("artifacts/experiments/matrix_runs")
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    config_hash = sha256_file(cfg_path)
    sha = git_sha()

    print(f"config: {cfg_path}")
    print(f"config_hash: {config_hash}")
    print(f"git_sha: {sha}")
    print(f"schema_version: {matrix.schema_version}")
    print(f"num_rows: {estimates['num_rows']}")
    print(f"num_gpus: {estimates['num_gpus']}")
    print(f"total_gpu_hours: {estimates['total_gpu_hours']:.3f}")
    print(f"wall_clock_hours_est: {estimates['wall_clock_hours_est']:.3f}")
    print("batched_dynamic_regrouping: not part of this matrix (separate experiment)")

    for plan in plans:
        print("---")
        print(f"id: {plan['id']}")
        print(f"group: {plan['group']}")
        print(f"device: {plan['device']}")
        print(f"estimated_gpu_hours: {plan['estimated_gpu_hours']}")
        print(f"config_hash: {plan['config_hash']}")
        print(f"command: {plan['command']}")

    summary: dict[str, Any] = {
        "config": str(cfg_path),
        "config_hash": config_hash,
        "git_sha": sha,
        "dry_run": bool(args.dry_run),
        "estimates": estimates,
        "plans": plans,
    }

    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dry_run_plan.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {output_dir / 'dry_run_plan.json'}")
        print("dry-run ok")
        return 0

    # Execution mode: launch sequentially with assigned devices (future: parallel pool).
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for plan in plans:
        print(f"launching {plan['id']} on {plan['device']} ...")
        proc = subprocess.run(plan["command"], shell=True, cwd=str(REPO_ROOT))
        results.append({"id": plan["id"], "returncode": proc.returncode, "device": plan["device"]})
        if proc.returncode != 0:
            print(f"FAILED {plan['id']} rc={proc.returncode}")
            break
    summary["results"] = results
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
