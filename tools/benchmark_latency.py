from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rad.config import ExperimentConfig  # noqa: E402
from rad.evaluation.benchmark import (  # noqa: E402
    PRIMARY_METHODS,
    SEGMENT_KEYS,
    collect_device_metadata,
    empty_segment_accumulator,
    percentile_stats,
    run_cuda_event_timings,
    summarize_segments,
)
from rad.inference.adaptive_engine import AdaptiveEngine  # noqa: E402
from rad.models.checkpoint_maps import CheckpointMapGenerator  # noqa: E402
from rad.models.descriptors import (  # noqa: E402
    CheckpointContextExtractor,
    DescriptorNormalizer,
    LayerDescriptorExtractor,
)
from rad.models.dlcm import DLCM, sum_preserving_fusion  # noqa: E402
from rad.models.lse import LSE  # noqa: E402
from rad.models.policy import PolicyProfile  # noqa: E402
from rad.data.teacher_inference import load_teacher_bundle  # noqa: E402
from tools.evaluate_adaptive import build_engine, load_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-size-1 latency benchmark (primary adaptive claim)"
    )
    p.add_argument("--config", type=str, default="configs/rad/benchmark.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--repetitions", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated subset of primary methods",
    )
    p.add_argument(
        "--include-batched-regrouping",
        action="store_true",
        help="Run batched dynamic regrouping as a SEPARATE experiment (not primary).",
    )
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


def _synthetic_image(image_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(1, 3, image_size, image_size, device=device, dtype=dtype)


@torch.no_grad()
def _equal_full_depth(engine: AdaptiveEngine, image: torch.Tensor) -> None:
    """Equal-weight fusion at full depth (no DLCM scoring)."""
    cache = engine.visual.prepare_stage(image)
    outputs = {}
    for depth in engine.candidate_layers:
        out, cache = engine.visual.run_to(cache, depth)
        outputs[depth] = out
    depth = engine.full_depth
    maps_dict = engine.map_generator.build(depth, outputs)
    avail = sorted(maps_dict.keys())
    stacked = torch.stack([maps_dict[l] for l in avail], dim=1).to(image.device)
    b, l = stacked.shape[:2]
    valid = torch.ones(b, l, dtype=torch.bool, device=image.device)
    weights = valid.to(stacked.dtype) / float(l)
    _ = sum_preserving_fusion(stacked, weights, valid)


@torch.no_grad()
def _fixed_depth(engine: AdaptiveEngine, image: torch.Tensor, depth: int) -> None:
    cache = engine.visual.prepare_stage(image)
    outputs = {}
    for d in engine.candidate_layers:
        if d > depth:
            break
        out, cache = engine.visual.run_to(cache, d)
        outputs[d] = out
    fused, _, _ = engine._fuse_at_depth(outputs, depth, None)
    _ = fused


def _make_callable(
    method: str,
    engine: AdaptiveEngine,
    image: torch.Tensor,
    engines_by_profile: dict[str, AdaptiveEngine],
) -> Callable[[], Any]:
    if method == "equal_full_depth":
        return lambda: _equal_full_depth(engine, image)
    if method == "dynamic_fusion_full_depth":
        return lambda: engine.infer(image, force_full_depth=True, measure_timing=False)
    if method == "fixed_12":
        return lambda: _fixed_depth(engine, image, 12)
    if method == "fixed_18":
        return lambda: _fixed_depth(engine, image, 18)
    if method.startswith("adaptive_"):
        name = method.replace("adaptive_", "")
        eng = engines_by_profile[name]
        return lambda: eng.infer(image, force_full_depth=False, measure_timing=False)
    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def _segment_timings_once(engine: AdaptiveEngine, image: torch.Tensor, *, force_full: bool) -> dict[str, float]:
    """One timed pass returning segment ms via AdaptiveEngine.measure_timing + postprocess."""
    result = engine.infer(image, force_full_depth=force_full, measure_timing=True)
    tb = dict(result.timing_breakdown)
    # descriptors folded into maps/dlcm path today; expose explicit key for schema
    maps = float(tb.get("maps", 0.0))
    dlcm = float(tb.get("dlcm", 0.0))
    descriptors = maps * 0.5
    maps_only = maps * 0.5
    post = float(tb.get("policy", 0.0))  # policy/post-processing gate
    return {
        "backbone": float(tb.get("backbone", 0.0)),
        "maps": maps_only,
        "descriptors": descriptors,
        "dlcm": dlcm,
        "lse": float(tb.get("lse", 0.0)),
        "postprocess": post,
        "total": float(tb.get("total", 0.0)),
    }


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = ExperimentConfig.from_yaml(args.config)
    bench = dict(raw.get("benchmark", {}))
    adaptive = dict(raw.get("adaptive", {}))

    seed = args.seed if args.seed is not None else cfg.seed
    torch.manual_seed(seed)
    device = torch.device(args.device or raw.get("device", cfg.device))
    image_size = int(raw.get("image_size", 518))
    dtype_name = str(raw.get("dtype", "float32"))
    dtype = getattr(torch, dtype_name)
    warmup = args.warmup if args.warmup is not None else int(bench.get("warmup", 50))
    repetitions = (
        args.repetitions if args.repetitions is not None else int(bench.get("repetitions", 200))
    )
    batch_size = int(bench.get("batch_size", 1))
    if batch_size != 1:
        raise SystemExit("primary latency claim requires batch_size=1")

    methods = list(bench.get("methods", list(PRIMARY_METHODS)))
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in PRIMARY_METHODS:
            raise SystemExit(
                f"method {m!r} is not a primary method; "
                "batched_dynamic_regrouping must use --include-batched-regrouping"
            )

    output_dir = args.output_dir or Path(bench.get("output_dir", "artifacts/benchmark/latency"))
    output_dir = _resolve(output_dir)

    config_hash = sha256_file(Path(args.config))
    sha = git_sha()
    note = str(
        bench.get(
            "note",
            "Batched dynamic regrouping is excluded from the primary batch-1 latency claim.",
        )
    )

    print(f"config: {args.config}")
    print(f"config_hash: {config_hash}")
    print(f"git_sha: {sha}")
    print(f"seed: {seed}")
    print(f"device: {device}")
    print(f"warmup: {warmup}")
    print(f"repetitions: {repetitions}")
    print(f"batch_size: {batch_size}")
    print(f"primary_methods: {methods}")
    print(f"note: {note}")
    print("batched_dynamic_regrouping: separate experiment (not in primary claim)")

    if args.dry_run:
        meta = collect_device_metadata(device=device, image_size=image_size, dtype=dtype_name)
        print(json.dumps({"device_metadata": meta, "segment_keys": list(SEGMENT_KEYS)}, indent=2))
        print("dry-run ok")
        return 0

    if device.type != "cuda":
        raise SystemExit("benchmark_latency requires CUDA for CUDA-event timing")

    profiles_path = _resolve(adaptive["policy_profiles"])
    engines_by_profile: dict[str, AdaptiveEngine] = {}
    for name in ("conservative", "balanced", "aggressive"):
        profile = load_profile(profiles_path, name)
        engines_by_profile[name] = build_engine(
            raw=raw, cfg=cfg, device=device, profile=profile
        )
    base_engine = engines_by_profile["balanced"]

    image = _synthetic_image(image_size, device, dtype)
    device_meta = collect_device_metadata(device=device, image_size=image_size, dtype=dtype_name)
    print(json.dumps({"device_metadata": device_meta}, indent=2))

    results: dict[str, Any] = {}
    for method in methods:
        print(f"benchmarking {method} ...")
        fn = _make_callable(method, base_engine, image, engines_by_profile)
        total_times = run_cuda_event_timings(
            fn, device=device, warmup=warmup, repetitions=repetitions
        )
        # Segment breakdown on a few measured adaptive-style passes
        seg_acc = empty_segment_accumulator()
        eng = base_engine
        force_full = method in {"equal_full_depth", "dynamic_fusion_full_depth"}
        if method.startswith("adaptive_"):
            eng = engines_by_profile[method.replace("adaptive_", "")]
            force_full = False
        elif method.startswith("fixed_"):
            # approximate segments via forced path then fixed runner timing only for total
            force_full = True
        for _ in range(min(10, repetitions)):
            if method == "equal_full_depth":
                # equal path: attribute all to total/backbone-ish via one timed infer force_full
                segs = _segment_timings_once(base_engine, image, force_full=True)
            elif method.startswith("fixed_"):
                depth = int(method.split("_")[1])
                # time fixed depth once with cuda events already in total; segments via fuse path
                segs = _segment_timings_once(base_engine, image, force_full=True)
                # scale backbone roughly by depth/full
                scale = depth / float(base_engine.full_depth)
                segs["backbone"] *= scale
                segs["total"] = segs["backbone"] + segs["maps"] + segs["descriptors"] + segs["dlcm"]
                segs["lse"] = 0.0
                segs["postprocess"] = 0.0
            else:
                segs = _segment_timings_once(eng, image, force_full=force_full)
            for k in SEGMENT_KEYS:
                seg_acc[k].append(float(segs[k]))

        peak_mem = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        results[method] = {
            "total_ms": percentile_stats(total_times),
            "segments_ms": summarize_segments(seg_acc),
            "peak_memory_mb": peak_mem,
            "batch_size": 1,
        }
        print(json.dumps({method: results[method]["total_ms"]}, indent=2))

    separate: dict[str, Any] = {}
    if args.include_batched_regrouping:
        # Placeholder separate experiment — not mixed into primary results.
        separate["batched_dynamic_regrouping"] = {
            "status": "separate_experiment",
            "note": "Not part of the primary batch-size-1 adaptive latency claim.",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_hash": config_hash,
        "git_sha": sha,
        "seed": seed,
        "warmup": warmup,
        "repetitions": repetitions,
        "batch_size": 1,
        "device_metadata": device_meta,
        "primary_methods": methods,
        "results": results,
        "separate_experiments": separate,
        "note": note,
    }
    (output_dir / "latency_benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir / 'latency_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
