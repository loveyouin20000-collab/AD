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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.config import ExperimentConfig  # noqa: E402
from rad.data.teacher_inference import load_teacher_bundle  # noqa: E402
from rad.inference.adaptive_engine import AdaptiveEngine  # noqa: E402
from rad.models.checkpoint_maps import CheckpointMapGenerator  # noqa: E402
from rad.models.descriptors import (  # noqa: E402
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM  # noqa: E402
from rad.models.lse import LSE  # noqa: E402
from rad.models.policy import PolicyProfile  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate batch-size-1 adaptive inference")
    p.add_argument("--config", type=str, default="configs/rad/adaptive.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--profile", type=str, default=None, help="aggressive|balanced|conservative")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--force-full-depth", action="store_true")
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


def _resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_profile(path: Path, name: str) -> PolicyProfile:
    payload = json.loads(path.read_text())
    raw = payload["profiles"][name]
    return PolicyProfile(
        name=str(raw["name"]),
        gain_threshold=float(raw["gain_threshold"]),
        kappa=float(raw["kappa"]),
        map_uncertainty_threshold=float(raw["map_uncertainty_threshold"]),
        image_confidence_margin=float(raw["image_confidence_margin"]),
        stability_threshold=float(raw["stability_threshold"]),
        require_map_uncertainty=bool(raw.get("require_map_uncertainty", False)),
        require_image_confidence=bool(raw.get("require_image_confidence", False)),
        require_stability=bool(raw.get("require_stability", False)),
    )


def build_engine(
    *,
    raw: dict[str, Any],
    cfg: ExperimentConfig,
    device: torch.device,
    profile: PolicyProfile,
) -> AdaptiveEngine:
    adaptive = dict(raw.get("adaptive", {}))
    candidate_layers = tuple(int(x) for x in cfg.backbone.candidate_layers)
    early_depths = tuple(int(x) for x in adaptive.get("early_depths", [12, 18]))
    full_depth = int(adaptive.get("full_depth", max(candidate_layers)))
    image_size = int(raw.get("image_size", 518))
    temperature = float(adaptive.get("temperature", 1.0))

    teacher = load_teacher_bundle(
        _resolve(raw["teacher"]["checkpoint_path"]),
        device=device,
        backbone=raw["teacher"].get("backbone"),
    )
    visual = teacher.model.visual

    dlcm_ckpt = torch.load(_resolve(adaptive["dlcm_checkpoint"]), map_location="cpu")
    dlcm = DLCM(max_layer_id=max(candidate_layers), alpha=0.0)
    dlcm.load_state_dict(dlcm_ckpt["dlcm"])
    dlcm.eval().to(device)

    lse_ckpt = torch.load(_resolve(adaptive["lse_checkpoint"]), map_location="cpu")
    state_dim = int(lse_ckpt.get("state_dim", adaptive.get("state_dim", 26)))
    lse_depths = tuple(int(x) for x in lse_ckpt.get("early_depths", early_depths))
    lse = LSE(state_dim=state_dim, early_depths=lse_depths)
    lse.load_state_dict(lse_ckpt["lse"])
    lse.eval().to(device)

    stats_path = _resolve(adaptive["descriptor_stats"])
    normalizer = DescriptorNormalizer.load(stats_path) if stats_path.is_file() else None

    engine = AdaptiveEngine(
        visual=visual,
        map_generator=CheckpointMapGenerator(
            image_size=image_size, candidate_layers=candidate_layers
        ),
        dlcm=dlcm,
        lse=lse,
        layer_extractor=LayerDescriptorExtractor(),
        context_extractor=CheckpointContextExtractor(backbone_depth=cfg.backbone.depth),
        profile=profile,
        candidate_layers=candidate_layers,
        early_depths=early_depths,
        full_depth=full_depth,
        image_size=image_size,
        normalizer=normalizer,
        temperature=temperature,
    )
    return engine.to(device)


def _synthetic_batch(image_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(1, 3, image_size, image_size, device=device)


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = ExperimentConfig.from_yaml(args.config)
    adaptive = dict(raw.get("adaptive", {}))

    seed = args.seed if args.seed is not None else cfg.seed
    torch.manual_seed(seed)
    device = torch.device(args.device or raw.get("device", cfg.device))

    output_dir = args.output_dir or Path(adaptive.get("output_dir", "artifacts/eval/adaptive"))
    output_dir = _resolve(output_dir)
    profile_name = args.profile or str(adaptive.get("profile_name", "balanced"))
    warmup = args.warmup if args.warmup is not None else int(adaptive.get("warmup", 50))
    limit = args.limit if args.limit is not None else adaptive.get("limit")
    limit_n = int(limit) if limit is not None else 1

    config_hash = sha256_file(Path(args.config))
    sha = git_sha()
    lse_path = _resolve(adaptive["lse_checkpoint"])
    dlcm_path = _resolve(adaptive["dlcm_checkpoint"])
    profiles_path = _resolve(adaptive["policy_profiles"])

    print(f"config: {args.config}")
    print(f"config_hash: {config_hash}")
    print(f"git_sha: {sha}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"lse_checkpoint: {lse_path}")
    print(f"dlcm_checkpoint: {dlcm_path}")
    print(f"profile: {profile_name}")
    print(f"output_dir: {output_dir}")
    print(f"warmup: {warmup}")

    if args.dry_run:
        print("dry-run ok")
        return 0

    if not lse_path.is_file():
        raise SystemExit(f"missing LSE checkpoint: {lse_path}")
    if not dlcm_path.is_file():
        raise SystemExit(f"missing DLCM checkpoint: {dlcm_path}")
    if not profiles_path.is_file():
        raise SystemExit(f"missing policy profiles: {profiles_path}")

    profile = load_profile(profiles_path, profile_name)
    engine = build_engine(raw=raw, cfg=cfg, device=device, profile=profile)

    image_size = int(raw.get("image_size", 518))
    rows: list[dict[str, Any]] = []
    depths: list[int] = []

    for i in tqdm(range(limit_n), desc="evaluate_adaptive"):
        image = _synthetic_batch(image_size, device)
        if i == 0 and warmup > 0:
            result = engine.timed_infer(
                image,
                force_full_depth=args.force_full_depth,
                warmup=warmup,
                repetitions=1,
            )
        else:
            result = engine.infer(
                image,
                force_full_depth=args.force_full_depth,
                measure_timing=False,
            )
        depths.append(result.selected_depth)
        row: dict[str, Any] = {
            "sample_index": i,
            "selected_depth": result.selected_depth,
            "checkpoint_trace": result.checkpoint_trace,
            "image_score": float(result.image_score.mean().item()),
            "final_map_shape": list(result.final_map.shape),
            "exit_decisions": result.exit_decisions,
            "timing_breakdown_ms": result.timing_breakdown,
            "gain_means": {
                str(d): float(g.mean.mean().item()) for d, g in result.gain_predictions.items()
            },
        }
        rows.append(row)
        print(
            f"sample={i} depth={result.selected_depth} "
            f"trace={result.checkpoint_trace} "
            f"map_shape={tuple(result.final_map.shape)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    summary = {
        "config_hash": config_hash,
        "git_sha": sha,
        "seed": seed,
        "profile": profile_name,
        "lse_checkpoint": str(lse_path),
        "lse_checkpoint_hash": sha256_file(lse_path),
        "dlcm_checkpoint_hash": sha256_file(dlcm_path),
        "n_samples": len(rows),
        "mean_selected_depth": float(sum(depths) / max(len(depths), 1)),
        "depth_histogram": {str(d): depths.count(d) for d in sorted(set(depths))},
        "force_full_depth": bool(args.force_full_depth),
        "warmup": warmup,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
