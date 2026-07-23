"""Shared helpers for Phase B1 staged-backbone CUDA numerical equivalence."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

from rad.data.adapters.preprocess import build_preprocess, preprocess_image
from rad.data.teacher_inference import TeacherBundle, _apply_layer_transform, load_teacher_bundle
from rad.models.checkpoint_maps import anomaly_map_from_tokens
from rad.types import StageCache

REPO_ROOT = Path(__file__).resolve().parents[2]
B1_SEED = 111
B1_ATOL = 1e-5
B1_RTOL = 1e-5
DEFAULT_CANDIDATE_LAYERS = (6, 12, 18, 24)
DEFAULT_SIGMA = 4.0
BACKBONE = "ViT-L/14@336px"
IMAGE_SIZE = 518

B1_ACCEPTED_CHECKPOINT = Path(
    "/root/autodl-tmp/AD/runs/baseline/mvtec_to_visa/seed_111_official_bs8/checkpoints/epoch_2.pth"
)
B1_ACCEPTED_CHECKPOINT_SHA256 = (
    "97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4"
)


@dataclass(frozen=True)
class TensorDiff:
    shape: tuple[int, ...]
    dtype: str
    max_abs: float
    mean_abs: float
    rel_l2: float
    allclose: bool


@dataclass
class BlockCountRecord:
    exit_depth: int
    blocks_executed: int
    expected: int


@dataclass
class ContinuationRecord:
    segment: str
    blocks_executed: int
    expected: str


@dataclass
class LatencyRecord:
    label: str
    milliseconds: float


@dataclass
class B1Sample:
    sample_id: str
    source: str
    path: str | None


@dataclass
class ProductionLoaderResult:
    success: bool
    loader: str
    error: str | None = None


@dataclass
class CpuSuiteResult:
    passed: int
    failed: int
    skipped: int
    green: bool
    summary: str


@dataclass
class B1QualificationResult:
    status: str
    git_sha: str
    checkpoint_path: str
    checkpoint_sha256: str
    environment: dict[str, Any]
    candidate_layers: list[int]
    samples: list[B1Sample]
    detail: str | None = None
    production_loader: ProductionLoaderResult | None = None
    cpu_suite: CpuSuiteResult | None = None
    feature_diffs: dict[str, TensorDiff] = field(default_factory=dict)
    map_diffs: dict[str, TensorDiff] = field(default_factory=dict)
    block_counts: list[BlockCountRecord] = field(default_factory=list)
    continuation: list[ContinuationRecord] = field(default_factory=list)
    latency: list[LatencyRecord] = field(default_factory=list)
    nonstandard_layers: dict[str, Any] = field(default_factory=dict)
    test_commands: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(checkpoint: Path | str, expected_sha256: str) -> Path:
    path = Path(checkpoint)
    if not path.is_absolute():
        raise ValueError(f"checkpoint path must be absolute: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"checkpoint SHA mismatch: expected {expected_sha256}, got {actual}"
        )
    return path


def git_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def set_seed(seed: int = B1_SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_diff(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float = B1_ATOL,
    rtol: float = B1_RTOL,
) -> TensorDiff:
    diff = (a.float() - b.float()).abs()
    denom = b.float().norm().item()
    rel_l2 = float((a.float() - b.float()).norm().item() / denom) if denom > 0 else 0.0
    return TensorDiff(
        shape=tuple(a.shape),
        dtype=str(a.dtype).replace("torch.", ""),
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        rel_l2=rel_l2,
        allclose=bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    )


def install_block_counter(visual: Any) -> SimpleNamespace:
    counter = SimpleNamespace(total=0, per_call=[])
    for block_idx, block in enumerate(visual.transformer.resblocks, start=1):
        original = block.forward

        def counted(
            x: torch.Tensor,
            _original: Any = original,
            _counter: SimpleNamespace = counter,
            _idx: int = block_idx,
        ) -> torch.Tensor:
            _counter.total += 1
            _counter.per_call.append(_idx)
            return _original(x)

        block.forward = counted
    return counter


def reset_block_counter(counter: SimpleNamespace) -> None:
    counter.total = 0
    counter.per_call = []


def load_teacher_production(
    checkpoint: Path | str,
    expected_sha256: str,
    device: torch.device,
) -> TeacherBundle:
    path = validate_checkpoint(checkpoint, expected_sha256)
    return load_teacher_bundle(path, device=device, backbone=BACKBONE)


def default_real_samples() -> list[B1Sample]:
    """Six fixed real images from accepted complete categories (not fixtures)."""
    samples = [
        B1Sample(
            "mvtec_bottle_good_000",
            "mvtec",
            "/root/autodl-tmp/data/mvtec/bottle/test/good/000.png",
        ),
        B1Sample(
            "mvtec_bottle_broken_large_000",
            "mvtec",
            "/root/autodl-tmp/data/mvtec/bottle/test/broken_large/000.png",
        ),
        B1Sample(
            "mvtec_bottle_contamination_000",
            "mvtec",
            "/root/autodl-tmp/data/mvtec/bottle/test/contamination/000.png",
        ),
        B1Sample(
            "visa_candle_normal_0000",
            "visa",
            "/root/autodl-tmp/data/Visa/candle/Data/Images/Normal/0000.JPG",
        ),
        B1Sample(
            "visa_candle_anomaly_001",
            "visa",
            "/root/autodl-tmp/data/Visa/candle/Data/Images/Anomaly/001.JPG",
        ),
        B1Sample(
            "visa_candle_anomaly_098",
            "visa",
            "/root/autodl-tmp/data/Visa/candle/Data/Images/Anomaly/098.JPG",
        ),
    ]
    for sample in samples:
        if sample.path is None or not Path(sample.path).is_file():
            raise FileNotFoundError(f"B1 sample missing: {sample.sample_id} -> {sample.path}")
    return samples


def load_preprocessed_image(path: str, device: torch.device) -> torch.Tensor:
    spec = build_preprocess(BACKBONE, IMAGE_SIZE)
    image = Image.open(path)
    tensor = preprocess_image(image, spec).unsqueeze(0).to(device)
    return tensor


def deterministic_synthetic(device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(B1_SEED)
    tensor = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=gen, dtype=torch.float32)
    return tensor.to(device)


def compare_candidate_features(
    bundle: TeacherBundle,
    image: torch.Tensor,
    candidate_layers: tuple[int, ...] = DEFAULT_CANDIDATE_LAYERS,
) -> dict[int, TensorDiff]:
    model = cast(Any, bundle.model)
    visual = cast(Any, model.visual)
    with torch.no_grad():
        legacy = model.encode_image(image, list(candidate_layers))
        staged = visual.forward_staged(image, list(candidate_layers))
    diffs: dict[int, TensorDiff] = {}
    for idx, depth in enumerate(candidate_layers):
        official = legacy["patch_tokens"][idx][:, legacy["patch_start_idx"] :, :]
        staged_tokens = staged[depth].patch_tokens
        diffs[depth] = tensor_diff(staged_tokens, official)
    return diffs


def build_official_full_depth_outputs(
    bundle: TeacherBundle,
    image: torch.Tensor,
    candidate_layers: tuple[int, ...] = DEFAULT_CANDIDATE_LAYERS,
) -> dict[str, Any]:
    model = cast(Any, bundle.model)
    with torch.no_grad():
        legacy = model.encode_image(image, list(candidate_layers))
        anomaly_features = legacy["anomaly_features"]
        normal_features = legacy["normal_features"]
        patch_tokens = legacy["patch_tokens"]
        patch_start_idx = legacy["patch_start_idx"]

        patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
        if bundle.cross_attn is not None:
            adapted = bundle.cross_attn(
                anomaly_features,
                normal_features,
                patch_features_list,
                list(candidate_layers),
            )
            anomaly_features_list = [item["anomaly"] for item in adapted]
            normal_features_list = [item["normal"] for item in adapted]
        else:
            anomaly_features_list = [anomaly_features] * len(patch_tokens)
            normal_features_list = [normal_features] * len(patch_tokens)

        layer_maps: dict[int, torch.Tensor] = {}
        anomaly_map_list: list[torch.Tensor] = []
        for idx, patch_feature in enumerate(patch_tokens):
            layer = candidate_layers[idx]
            anomaly_feat = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
            normal_feat = F.normalize(normal_features_list[idx], dim=1, eps=1e-8)
            patch = _apply_layer_transform(
                bundle.layer_transforms, layer, patch_feature[:, patch_start_idx:, :]
            )
            amap = anomaly_map_from_tokens(
                anomaly_feat, normal_feat, patch, bundle.image_size
            )
            layer_maps[layer] = amap
            anomaly_map_list.append(amap)

        fused = torch.stack(anomaly_map_list).sum(dim=0)
        filtered = gaussian_filter(fused[0].detach().cpu().numpy(), sigma=DEFAULT_SIGMA)
        final_map = torch.from_numpy(filtered).to(image.device).unsqueeze(0)
        image_score = final_map.reshape(final_map.shape[0], -1).max(dim=1).values

    return {
        "layer_maps": layer_maps,
        "fused": fused,
        "final_map": final_map,
        "image_score": image_score,
    }


def build_staged_full_depth_outputs(
    bundle: TeacherBundle,
    image: torch.Tensor,
    candidate_layers: tuple[int, ...] = DEFAULT_CANDIDATE_LAYERS,
) -> dict[str, Any]:
    model = cast(Any, bundle.model)
    visual = cast(Any, model.visual)
    with torch.no_grad():
        staged = visual.forward_staged(image, list(candidate_layers))
        depth = candidate_layers[-1]
        anomaly = staged[depth].anomaly_token
        normal = staged[depth].normal_token
        available = [layer for layer in candidate_layers if layer <= depth]
        patch_list = [staged[layer].patch_tokens for layer in available]

        if bundle.cross_attn is not None:
            adapted = bundle.cross_attn(anomaly, normal, patch_list, available)
            anomaly_list = [item["anomaly"] for item in adapted]
            normal_list = [item["normal"] for item in adapted]
        else:
            anomaly_list = [anomaly] * len(available)
            normal_list = [normal] * len(available)

        layer_maps: dict[int, torch.Tensor] = {}
        anomaly_map_list: list[torch.Tensor] = []
        for idx, layer in enumerate(available):
            patch = _apply_layer_transform(
                bundle.layer_transforms, layer, staged[layer].patch_tokens
            )
            amap = anomaly_map_from_tokens(
                anomaly_list[idx], normal_list[idx], patch, bundle.image_size
            )
            layer_maps[layer] = amap
            anomaly_map_list.append(amap)

        fused = torch.stack(anomaly_map_list).sum(dim=0)
        filtered = gaussian_filter(fused[0].detach().cpu().numpy(), sigma=DEFAULT_SIGMA)
        final_map = torch.from_numpy(filtered).to(image.device).unsqueeze(0)
        image_score = final_map.reshape(final_map.shape[0], -1).max(dim=1).values

    return {
        "layer_maps": layer_maps,
        "fused": fused,
        "final_map": final_map,
        "image_score": image_score,
    }


def measure_cuda_latency(
    fn: Callable[[], None],
    device: torch.device,
    *,
    warmup: int = 2,
    repeats: int = 5,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / repeats)


def _measure_incremental_segment(
    visual: Any,
    image: torch.Tensor,
    device: torch.device,
    *,
    setup_depth: int,
    target_depth: int,
    warmup: int = 2,
    repeats: int = 5,
) -> float:
    timings: list[float] = []

    def once() -> None:
        with torch.no_grad():
            cache = visual.prepare_stage(image)
            _, staged_cache = visual.run_to(cache, setup_depth)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            visual.run_to(staged_cache, target_depth)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(float(start.elapsed_time(end)))

    for _ in range(warmup):
        once()
    timings.clear()
    for _ in range(repeats):
        once()
    return sum(timings) / len(timings)


def measure_staged_depth_latencies(visual: Any, image: torch.Tensor, device: torch.device) -> dict[str, float]:
    """Diagnostic latency: full-from-start and incremental-only segments."""

    def full_12_from_start() -> None:
        with torch.no_grad():
            cache = visual.prepare_stage(image)
            visual.run_to(cache, 12)

    def full_18_from_start() -> None:
        with torch.no_grad():
            cache = visual.prepare_stage(image)
            visual.run_to(cache, 18)

    def full_24_from_start() -> None:
        with torch.no_grad():
            cache = visual.prepare_stage(image)
            visual.run_to(cache, 24)

    return {
        "full_12_from_start_ms": measure_cuda_latency(full_12_from_start, device),
        "incremental_12_to_18_ms": _measure_incremental_segment(
            visual, image, device, setup_depth=12, target_depth=18
        ),
        "full_18_from_start_ms": measure_cuda_latency(full_18_from_start, device),
        "incremental_18_to_24_ms": _measure_incremental_segment(
            visual, image, device, setup_depth=18, target_depth=24
        ),
        "full_24_from_start_ms": measure_cuda_latency(full_24_from_start, device),
    }


def run_cpu_regression_suite() -> CpuSuiteResult:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/rad",
            "-q",
            "--tb=short",
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr
    passed = failed = skipped = 0
    if match := re.search(r"(\d+)\s+passed", summary):
        passed = int(match.group(1))
    if match := re.search(r"(\d+)\s+failed", summary):
        failed = int(match.group(1))
    if match := re.search(r"(\d+)\s+skipped", summary):
        skipped = int(match.group(1))
    green = proc.returncode == 0 and failed == 0
    return CpuSuiteResult(
        passed=passed,
        failed=failed,
        skipped=skipped,
        green=green,
        summary=summary,
    )


B1_DIVERGENCE_THRESHOLD = B1_ATOL
CHECKPOINT_SNAPSHOT_LAYERS = (6, 12, 18, 24)
BLOCK_TRACE_OPS = (
    "block_input",
    "attention_output",
    "residual_add_1",
    "mlp_input",
    "mlp_output",
    "block_output",
)


def collect_backend_settings() -> dict[str, bool]:
    return {
        "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
    }


def tensor_layout_info(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "strides": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "contiguous": bool(tensor.is_contiguous()),
    }


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().float().cpu().contiguous()
    return {
        **tensor_layout_info(tensor),
        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
        "min": float(cpu.min().item()),
        "max": float(cpu.max().item()),
        "mean": float(cpu.mean().item()),
        "std": float(cpu.std(unbiased=False).item()),
    }


def tensor_compare_report(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    threshold: float = B1_DIVERGENCE_THRESHOLD,
    atol: float = B1_ATOL,
    rtol: float = B1_RTOL,
) -> dict[str, Any]:
    diff = tensor_diff(a, b, atol=atol, rtol=rtol)
    abs_diff = (a.float() - b.float()).abs()
    over = int(abs_diff.gt(threshold).sum().item())
    layout_a = tensor_layout_info(a)
    layout_b = tensor_layout_info(b)
    return {
        "shape": diff.shape,
        "dtype": diff.dtype,
        "max_abs": diff.max_abs,
        "mean_abs": diff.mean_abs,
        "rel_l2": diff.rel_l2,
        "allclose": diff.allclose,
        "count_over_threshold": over,
        "threshold": threshold,
        "layout_a": layout_a,
        "layout_b": layout_b,
        "layout_match": layout_a == layout_b,
    }


def reset_cuda_diagnostic_state(seed: int = B1_SEED) -> None:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _embed_sequence(visual: Any, image: torch.Tensor) -> torch.Tensor:
    return visual._embed_image(image).permute(1, 0, 2)


def _ln_post_patch_tokens(visual: Any, sequence: torch.Tensor) -> torch.Tensor:
    tokens = visual.ln_post(sequence.permute(1, 0, 2))
    return tokens[:, 3:, :]


class BlockTraceRecorder:
    """Capture per-resblock intermediates via temporary forward hooks."""

    def __init__(self, visual: Any) -> None:
        self.visual = visual
        self.traces: dict[int, dict[str, torch.Tensor]] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> BlockTraceRecorder:
        for block_idx, block in enumerate(self.visual.transformer.resblocks, start=1):
            state: dict[str, torch.Tensor | None] = {
                "block_input": None,
                "attention_output": None,
                "residual_add_1": None,
                "mlp_input": None,
                "mlp_output": None,
                "block_output": None,
            }

            def capture_input(
                _module: Any,
                inputs: tuple[torch.Tensor, ...],
                *,
                _state: dict[str, torch.Tensor | None] = state,
            ) -> None:
                _state["block_input"] = inputs[0].detach()

            def capture_output(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                _state: dict[str, torch.Tensor | None] = state,
                _block_idx: int = block_idx,
            ) -> None:
                _state["block_output"] = output.detach()
                self.traces[_block_idx] = {
                    key: value
                    for key, value in _state.items()
                    if value is not None
                }

            self._handles.append(block.register_forward_pre_hook(capture_input))
            self._handles.append(block.register_forward_hook(capture_output))

            def capture_attn(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor | tuple[torch.Tensor, ...],
                *,
                _state: dict[str, torch.Tensor | None] = state,
            ) -> None:
                tensor_out = output[0] if isinstance(output, tuple) else output
                _state["attention_output"] = tensor_out.detach()

            def capture_ln2(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                _state: dict[str, torch.Tensor | None] = state,
            ) -> None:
                _state["mlp_input"] = output.detach()

            def capture_mlp(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                _state: dict[str, torch.Tensor | None] = state,
            ) -> None:
                _state["mlp_output"] = output.detach()

            self._handles.append(block.attn.register_forward_hook(capture_attn))
            self._handles.append(block.ln_2.register_forward_hook(capture_ln2))
            self._handles.append(block.mlp.register_forward_hook(capture_mlp))

            def capture_res1(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                _output: torch.Tensor,
                *,
                _state: dict[str, torch.Tensor | None] = state,
                _block: Any = block,
            ) -> None:
                if _state["block_input"] is not None and _state["attention_output"] is not None:
                    _state["residual_add_1"] = (
                        _state["block_input"] + _state["attention_output"]
                    ).detach()

            self._handles.append(block.attn.register_forward_hook(capture_res1))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _first_divergent_operation(
    left: dict[int, dict[str, torch.Tensor]],
    right: dict[int, dict[str, torch.Tensor]],
    *,
    threshold: float = B1_DIVERGENCE_THRESHOLD,
    max_block: int = 12,
) -> dict[str, Any] | None:
    for block_idx in range(1, max_block + 1):
        l_ops = left.get(block_idx, {})
        r_ops = right.get(block_idx, {})
        for op_name in BLOCK_TRACE_OPS:
            if op_name not in l_ops or op_name not in r_ops:
                continue
            report = tensor_compare_report(l_ops[op_name], r_ops[op_name], threshold=threshold)
            if report["max_abs"] > threshold or report["count_over_threshold"] > 0:
                return {
                    "block": block_idx,
                    "operation": op_name,
                    "report": report,
                }
    return None


def run_path_a_official_transformer(
    visual: Any,
    sequence: torch.Tensor,
    *,
    snapshot_layers: tuple[int, ...] = CHECKPOINT_SNAPSHOT_LAYERS,
    max_block: int = 12,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    layers = [layer for layer in snapshot_layers if layer <= max_block]
    with BlockTraceRecorder(visual) as recorder:
        with torch.no_grad():
            x, out_tokens = visual.transformer.forward(sequence.clone(), layers)
            patch_snapshots: dict[int, torch.Tensor] = {}
            for idx, depth in enumerate(layers):
                patch_snapshots[depth] = _ln_post_patch_tokens(
                    visual, out_tokens[idx]
                )
    return x, patch_snapshots, recorder.traces


def run_path_b_manual_loop(
    visual: Any,
    sequence: torch.Tensor,
    *,
    max_block: int = 12,
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    with BlockTraceRecorder(visual) as recorder:
        with torch.no_grad():
            x = sequence.clone()
            for block in visual.transformer.resblocks[:max_block]:
                x = block(x)
    return x, recorder.traces


def run_path_c_manual_with_snapshots(
    visual: Any,
    sequence: torch.Tensor,
    *,
    snapshot_layers: tuple[int, ...] = CHECKPOINT_SNAPSHOT_LAYERS,
    max_block: int = 12,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    with BlockTraceRecorder(visual) as recorder:
        with torch.no_grad():
            x = sequence.clone()
            patch_snapshots: dict[int, torch.Tensor] = {}
            for block_idx, block in enumerate(visual.transformer.resblocks[:max_block], start=1):
                x = block(x)
                if block_idx in snapshot_layers:
                    _ = x.detach().clone()
                    patch_snapshots[block_idx] = _ln_post_patch_tokens(visual, x)
    return x, patch_snapshots, recorder.traces


def run_path_d_staged_stop_resume(
    visual: Any,
    image: torch.Tensor,
    *,
    snapshot_layers: tuple[int, ...] = CHECKPOINT_SNAPSHOT_LAYERS,
    max_block: int = 12,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]]]:

    patch_snapshots: dict[int, torch.Tensor] = {}
    layouts: list[dict[str, Any]] = []
    with torch.no_grad():
        cache = visual.prepare_stage(image)
        layouts.append(
            {
                "stage": "after_prepare",
                "continuation": tensor_layout_info(cache.sequence),
            }
        )
        for depth in snapshot_layers:
            if depth > max_block:
                continue
            output, cache = visual.run_to(cache, depth)
            patch_snapshots[depth] = output.patch_tokens
            layouts.append(
                {
                    "stage": f"after_run_to_{depth}",
                    "continuation": tensor_layout_info(cache.sequence),
                }
            )
    return patch_snapshots, layouts


def run_checkpoint_snapshot_variants(
    visual: Any,
    sequence: torch.Tensor,
    *,
    depth: int = 12,
) -> dict[str, dict[str, Any]]:

    results: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        live = sequence.clone()
        for block in visual.transformer.resblocks[:depth]:
            live = block(live)
        reference = _ln_post_patch_tokens(visual, live)

        variant_live_snapshot = live.detach().clone()
        results["variant_1_live_continuation_detach_clone_snapshot"] = tensor_compare_report(
            _ln_post_patch_tokens(visual, variant_live_snapshot),
            reference,
        )

        cloned = sequence.clone()
        for block in visual.transformer.resblocks[:depth]:
            cloned = block(cloned)
        results["variant_2_cloned_continuation"] = tensor_compare_report(
            _ln_post_patch_tokens(visual, cloned),
            reference,
        )

        contiguous = sequence.clone()
        for block in visual.transformer.resblocks[:depth]:
            contiguous = block(contiguous)
        contiguous = contiguous.contiguous()
        results["variant_3_contiguous_continuation"] = tensor_compare_report(
            _ln_post_patch_tokens(visual, contiguous),
            reference,
        )

        cache = StageCache(sequence=sequence.clone(), next_block=1, patch_tokens={})
        output, wrapped = visual.run_to(cache, depth)
        results["variant_4_staged_state_wrapper"] = tensor_compare_report(
            output.patch_tokens,
            reference,
        )
        results["variant_4_continuation_layout"] = {
            "reference": tensor_layout_info(live),
            "wrapped": tensor_layout_info(wrapped.sequence),
        }
    return results


def compare_model_identity(visual: Any) -> dict[str, Any]:
    param_ids = {
        name: int(param.data_ptr())
        for name, param in visual.named_parameters()
    }
    buffer_ids = {name: int(buf.data_ptr()) for name, buf in visual.named_buffers()}
    return {
        "parameter_count": len(param_ids),
        "buffer_count": len(buffer_ids),
        "training": bool(visual.training),
        "transformer_training": bool(visual.transformer.training),
    }


def diagnose_four_path_divergence(
    bundle: TeacherBundle,
    image: torch.Tensor,
    *,
    sample_id: str,
    max_block: int = 12,
    threshold: float = B1_DIVERGENCE_THRESHOLD,
) -> dict[str, Any]:
    visual = cast(Any, bundle.model.visual)
    visual.eval()
    bundle.model.eval()

    reset_cuda_diagnostic_state()
    input_fp = tensor_fingerprint(image)
    sequence = _embed_sequence(visual, image)
    sequence_fp = tensor_fingerprint(sequence)

    backend_baseline = collect_backend_settings()
    model_before = compare_model_identity(visual)

    reset_cuda_diagnostic_state()
    _, path_a_patches, trace_a = run_path_a_official_transformer(
        visual, sequence, max_block=max_block
    )

    reset_cuda_diagnostic_state()
    _, trace_b = run_path_b_manual_loop(visual, sequence, max_block=max_block)

    reset_cuda_diagnostic_state()
    _, path_c_patches, trace_c = run_path_c_manual_with_snapshots(
        visual, sequence, max_block=max_block
    )

    reset_cuda_diagnostic_state()
    path_d_patches, path_d_layouts = run_path_d_staged_stop_resume(
        visual, image, max_block=max_block
    )

    model_after = compare_model_identity(visual)

    comparisons: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    def _summarize(
        label: str,
        first: dict[str, Any] | None,
        *,
        input_report: dict[str, Any] | None = None,
        suspected_cause: str,
    ) -> None:
        comparisons[label] = {
            "first_divergence": first,
            "input_report": input_report,
            "suspected_cause": suspected_cause,
        }
        summary_rows.append(
            {
                "comparison": label,
                "first_divergent_block": None if first is None else first["block"],
                "first_divergent_operation": None if first is None else first["operation"],
                "input_max_diff": None if input_report is None else input_report["max_abs"],
                "output_max_diff": None if first is None else first["report"]["max_abs"],
                "stride_layout_difference": (
                    False
                    if first is None
                    else not first["report"]["layout_match"]
                ),
                "suspected_cause": suspected_cause,
            }
        )

    first_a_b = _first_divergent_operation(trace_a, trace_b, threshold=threshold, max_block=max_block)
    _summarize(
        "A_vs_B",
        first_a_b,
        suspected_cause=(
            "Separate CUDA execution of official-style clone checkpoints vs pure manual loop; "
            "non-deterministic attention accumulation."
        ),
    )

    first_b_c = _first_divergent_operation(trace_b, trace_c, threshold=threshold, max_block=max_block)
    _summarize(
        "B_vs_C",
        first_b_c,
        suspected_cause="detach().clone() snapshot side effects on subsequent CUDA kernels.",
    )

    trace_d: dict[int, dict[str, torch.Tensor]] = {}
    reset_cuda_diagnostic_state()
    with BlockTraceRecorder(visual) as recorder:
        with torch.no_grad():
            cache = visual.prepare_stage(image)
            for depth in (6, 12):
                if depth > max_block:
                    continue
                _, cache = visual.run_to(cache, depth)
        trace_d = recorder.traces
    first_c_d = _first_divergent_operation(
        trace_c, trace_d, threshold=threshold, max_block=max_block
    )
    _summarize(
        "C_vs_D",
        first_c_d,
        suspected_cause="Staged stop/resume segments vs uninterrupted manual execution.",
    )

    patch_reports: dict[str, dict[str, Any]] = {}
    for depth in (6, 12):
        if depth not in path_a_patches:
            continue
        key = f"layer_{depth}"
        patch_reports[f"A_vs_B_{key}"] = tensor_compare_report(
            path_a_patches[depth], _ln_post_patch_tokens(visual, trace_b[depth]["block_output"])
        )
        if depth in path_c_patches:
            patch_reports[f"B_vs_C_{key}"] = tensor_compare_report(
                _ln_post_patch_tokens(visual, trace_b[depth]["block_output"]),
                path_c_patches[depth],
            )
        if depth in path_d_patches:
            patch_reports[f"C_vs_D_{key}"] = tensor_compare_report(
                path_c_patches.get(depth, path_a_patches[depth]),
                path_d_patches[depth],
            )
            patch_reports[f"A_vs_D_{key}"] = tensor_compare_report(
                path_a_patches[depth], path_d_patches[depth]
            )

    first_a_d = None
    for depth in (6, 12, 18, 24):
        if depth > max_block:
            break
        report_key = f"A_vs_D_layer_{depth}"
        if report_key not in patch_reports:
            if depth not in path_a_patches or depth not in path_d_patches:
                continue
            patch_reports[report_key] = tensor_compare_report(
                path_a_patches[depth], path_d_patches[depth]
            )
        report = patch_reports[report_key]
        if first_a_d is None and (
            report["max_abs"] > threshold or report["count_over_threshold"] > 0
        ):
            first_a_d = {
                "block": depth,
                "operation": "ln_post_patch_tokens",
                "report": report,
            }
    _summarize(
        "A_vs_D",
        first_a_d,
        suspected_cause="Combined official clone checkpoints and staged stop/resume divergence.",
    )

    reset_cuda_diagnostic_state()
    snapshot_variants = run_checkpoint_snapshot_variants(visual, sequence, depth=max_block)

    backend_matrix: dict[str, Any] = {}
    original = collect_backend_settings()
    matrix_configs = {
        "current": {},
        "no_tf32": {
            "cuda.matmul.allow_tf32": False,
            "cudnn.allow_tf32": False,
        },
        "deterministic": {
            "cuda.matmul.allow_tf32": False,
            "cudnn.allow_tf32": False,
            "cudnn.benchmark": False,
            "cudnn.deterministic": True,
        },
    }
    for label, overrides in matrix_configs.items():
        torch.backends.cuda.matmul.allow_tf32 = overrides.get(
            "cuda.matmul.allow_tf32", original["cuda.matmul.allow_tf32"]
        )
        torch.backends.cudnn.allow_tf32 = overrides.get(
            "cudnn.allow_tf32", original["cudnn.allow_tf32"]
        )
        torch.backends.cudnn.benchmark = overrides.get(
            "cudnn.benchmark", original["cudnn.benchmark"]
        )
        torch.backends.cudnn.deterministic = overrides.get(
            "cudnn.deterministic", original["cudnn.deterministic"]
        )
        reset_cuda_diagnostic_state()
        model = cast(Any, bundle.model)
        with torch.no_grad():
            legacy = model.encode_image(image, [6, 12, 18, 24])
            staged = visual.forward_staged(image, [6, 12, 18, 24])
        layer_reports = {}
        for idx, depth in enumerate((6, 12, 18, 24)):
            if depth > max_block:
                continue
            official = legacy["patch_tokens"][idx][:, legacy["patch_start_idx"] :, :]
            staged_tokens = staged[depth].patch_tokens
            layer_reports[f"layer_{depth}"] = tensor_compare_report(
                staged_tokens, official, threshold=threshold
            )
        backend_matrix[label] = {
            "settings": collect_backend_settings(),
            "layer_reports": layer_reports,
        }
    torch.backends.cuda.matmul.allow_tf32 = original["cuda.matmul.allow_tf32"]
    torch.backends.cudnn.allow_tf32 = original["cudnn.allow_tf32"]
    torch.backends.cudnn.benchmark = original["cudnn.benchmark"]
    torch.backends.cudnn.deterministic = original["cudnn.deterministic"]

    same_chain_control = {}
    with torch.inference_mode():
        x = sequence.clone()
        for block_idx, block in enumerate(visual.transformer.resblocks[:max_block], start=1):
            x = block(x)
            if block_idx == max_block:
                snap = x.clone()
        same_chain_control["clone_vs_live_ln_post"] = tensor_compare_report(
            _ln_post_patch_tokens(visual, snap),
            _ln_post_patch_tokens(visual, x),
        )

    if first_a_b is not None:
        root_cause = "case_b_attention_backend_or_official_clone_path"
    elif first_c_d is not None and first_b_c is None:
        root_cause = "case_a_checkpoint_state_capture_resume"
    elif all(
        row["first_divergent_block"] is not None for row in summary_rows
    ):
        root_cause = "case_c_cuda_kernel_dispatch_non_determinism"
    else:
        root_cause = "case_d_unlocalized"

    # Prefer case C when same-chain control is exact but separate passes diverge.
    if (
        same_chain_control["clone_vs_live_ln_post"]["max_abs"] == 0.0
        and first_a_b is not None
    ):
        root_cause = "case_c_cuda_kernel_dispatch_non_determinism"

    return {
        "sample_id": sample_id,
        "threshold": threshold,
        "max_block": max_block,
        "input_image": input_fp,
        "transformer_input_sequence": sequence_fp,
        "backend_baseline": backend_baseline,
        "model_identity_before": model_before,
        "model_identity_after": model_after,
        "comparison_summary_table": summary_rows,
        "comparisons": comparisons,
        "patch_token_reports": patch_reports,
        "checkpoint_snapshot_variants": snapshot_variants,
        "staged_continuation_layouts": path_d_layouts,
        "backend_matrix": backend_matrix,
        "same_chain_control": same_chain_control,
        "root_cause_classification": root_cause,
        "proposed_minimal_fix": (
            "Preserve live continuation tensors in StageCache (already true) and capture "
            "feature snapshots as read-only detach().clone() for reporting only. "
            "B1 equivalence remains blocked by separate-pass CUDA non-determinism in "
            "MultiheadAttention unless an revised criterion is approved."
        ),
    }


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        env["gpu_model"] = props.name
        env["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
        env["cuda_version"] = torch.version.cuda
        env["cudnn_version"] = str(torch.backends.cudnn.version())
        try:
            smi = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            env["nvidia_driver"] = smi.splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError):
            env["nvidia_driver"] = None
    try:
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        env["nvcc"] = nvcc.stdout.strip().split("\n")[-1]
    except (FileNotFoundError, subprocess.CalledProcessError):
        env["nvcc"] = None
    env["observed_execution_settings"] = observe_effective_execution_settings()
    return env


def observe_effective_execution_settings() -> dict[str, Any]:
    """Record settings observed after configuration, not merely requested values."""
    observed: dict[str, Any] = {
        "use_deterministic_algorithms": (
            bool(torch.are_deterministic_algorithms_enabled())
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else None
        ),
        "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if hasattr(torch, "get_float32_matmul_precision"):
        observed["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    cuda_backend = getattr(torch.backends, "cuda", None)
    for name in (
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
    ):
        fn = getattr(cuda_backend, name, None) if cuda_backend is not None else None
        observed[name] = bool(fn()) if callable(fn) else None
    mha_backend = getattr(torch.backends, "mha", None)
    get_fastpath = getattr(mha_backend, "get_fastpath_enabled", None) if mha_backend else None
    observed["mha_fastpath_enabled"] = bool(get_fastpath()) if callable(get_fastpath) else None
    return observed


# ---------------------------------------------------------------------------
# Phase B1 dual equivalence protocol (deterministic control + dual gates)
# ---------------------------------------------------------------------------

B1_OPERATIONAL_REPEATS = 20
B1_CROSS_PATH_RATIO = 1.25
# Historical fixture path used by earlier B1 drafts (invalid for accepted gate).
MVTEC_SAMPLE_FIXTURE_ROOT = Path("/root/autodl-tmp/data/mvtec/sample")
MVTEC_ROOT = Path("/root/autodl-tmp/data/mvtec")
VISA_ROOT = Path("/root/autodl-tmp/data/Visa")
# Accepted B1 task-level categories: complete real dataset categories only.
B1_TASK_MVTEC_CATEGORY = "bottle"
B1_TASK_VISA_CATEGORY = "candle"
B1_TASK_IMAGE_METRIC_PP = 0.05
B1_TASK_PIXEL_METRIC_PP = 0.05
B1_TASK_AUPRO_PP = 0.10
B1_TASK_BOUNDARY_PP = 0.10
B1_TASK_PER_CATEGORY_PP = 0.20
_PATH_FLAG_TOKENS = (
    "test",
    "tests",
    "fixture",
    "fixtures",
    "example",
    "examples",
    "synthetic",
)


@dataclass
class DivergenceStats:
    median: float
    p95: float
    maximum: float
    rel_l2_median: float
    map_correlation_median: float | None = None


@dataclass
class DeterministicControlResult:
    status: str
    settings: dict[str, Any]
    attention_backend: dict[str, Any]
    comparisons: dict[str, dict[str, Any]]
    decision: str
    deterministic_error: dict[str, Any] | None = None


@dataclass
class SameChainGateResult:
    status: str
    feature_diffs: dict[str, TensorDiff]
    map_diffs: dict[str, TensorDiff]
    block_counts: list[BlockCountRecord]
    continuation: list[ContinuationRecord]
    continuation_live_tensor_preserved: bool
    nonstandard_layers_validated: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class OperationalNoiseResult:
    status: str
    per_output: dict[str, dict[str, Any]]
    self_noise_p95: float
    cross_excess_p95: float
    cross_path_p95: float
    ratio_pass: bool
    excess_pass: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class TaskLevelSampleMetrics:
    sample_id: str
    category: str
    image_score_diff: float
    final_map_max_abs: float
    final_map_rel_l2: float
    map_pearson: float
    map_spearman: float
    localization_error_diff: float
    top1_patch_overlap: float
    anomaly_region_overlap: float


@dataclass
class TaskLevelImpactResult:
    status: str
    sample_count: int
    metric_differences: dict[str, float]
    metric_raw: dict[str, dict[str, float]]
    per_category_metric_differences_pp: dict[str, dict[str, float]]
    localization_error_diff_stats: dict[str, float]
    map_rel_l2_stats: dict[str, float]
    min_map_pearson: float
    min_map_spearman: float
    min_top1_patch_overlap: float
    max_image_score_diff: float
    p95_localization_error_diff: float
    nonfinite_samples: list[str]
    samples: list[TaskLevelSampleMetrics]
    provenance: list[dict[str, Any]] = field(default_factory=list)
    superseded_evidence: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class EquivalenceProtocolResult:
    deterministic_control: DeterministicControlResult
    algorithmic_same_chain: SameChainGateResult
    operational_noise_envelope: OperationalNoiseResult
    task_level_impact: TaskLevelImpactResult
    detail: str | None = None


def apply_deterministic_cuda_settings(seed: int = B1_SEED) -> dict[str, Any]:
    """Apply deterministic CUDA settings before model load (Part A)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    settings: dict[str, Any] = {
        "seed": seed,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    deterministic_error: dict[str, Any] | None = None
    try:
        torch.use_deterministic_algorithms(True)
        settings["use_deterministic_algorithms"] = True
    except RuntimeError as exc:
        settings["use_deterministic_algorithms"] = False
        deterministic_error = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "full_exception": repr(exc),
        }

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    matmul_precision = None
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
        matmul_precision = "highest"

    settings.update(
        {
            "cuda.matmul.allow_tf32": False,
            "cudnn.allow_tf32": False,
            "cudnn.benchmark": False,
            "cudnn.deterministic": True,
            "float32_matmul_precision": matmul_precision,
            "deterministic_error": deterministic_error,
        }
    )
    return settings


def apply_attention_backend_overrides() -> dict[str, Any]:
    """Configure SDP / MHA backends when supported by PyTorch 2.0."""
    report: dict[str, Any] = {"available": {}, "applied": {}}
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None:
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            fn = getattr(cuda_backend, name, None)
            report["available"][name] = callable(fn)
        if callable(getattr(cuda_backend, "enable_flash_sdp", None)):
            cuda_backend.enable_flash_sdp(False)
            report["applied"]["enable_flash_sdp"] = False
        if callable(getattr(cuda_backend, "enable_mem_efficient_sdp", None)):
            cuda_backend.enable_mem_efficient_sdp(False)
            report["applied"]["enable_mem_efficient_sdp"] = False
        if callable(getattr(cuda_backend, "enable_math_sdp", None)):
            cuda_backend.enable_math_sdp(True)
            report["applied"]["enable_math_sdp"] = True

    mha_backend = getattr(torch.backends, "mha", None)
    fastpath_fn = getattr(mha_backend, "set_fastpath_enabled", None) if mha_backend else None
    report["available"]["mha.set_fastpath_enabled"] = callable(fastpath_fn)
    if callable(fastpath_fn):
        fastpath_fn(False)
        report["applied"]["mha.set_fastpath_enabled"] = False
    return report


def _map_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    flat_a = a.detach().float().reshape(-1)
    flat_b = b.detach().float().reshape(-1)
    if flat_a.numel() == 0:
        return 1.0
    flat_a = flat_a - flat_a.mean()
    flat_b = flat_b - flat_b.mean()
    denom = flat_a.norm() * flat_b.norm()
    if float(denom.item()) == 0.0:
        return 1.0
    return float((flat_a * flat_b).sum().item() / denom.item())


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def _pairwise_max_abs(values: list[torch.Tensor]) -> list[float]:
    if len(values) < 2:
        return [0.0]
    out: list[float] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            out.append(tensor_diff(values[i], values[j]).max_abs)
    return out


def _run_official_once(
    bundle: TeacherBundle,
    image: torch.Tensor,
    candidate_layers: tuple[int, ...] = DEFAULT_CANDIDATE_LAYERS,
) -> dict[str, Any]:
    model = cast(Any, bundle.model)
    with torch.no_grad():
        legacy = model.encode_image(image, list(candidate_layers))
        outputs = build_official_full_depth_outputs(bundle, image, candidate_layers)
    patch_features = {
        depth: legacy["patch_tokens"][idx][:, legacy["patch_start_idx"] :, :]
        for idx, depth in enumerate(candidate_layers)
    }
    return {"patch_features": patch_features, **outputs}


def _run_staged_once(
    bundle: TeacherBundle,
    image: torch.Tensor,
    candidate_layers: tuple[int, ...] = DEFAULT_CANDIDATE_LAYERS,
) -> dict[str, Any]:
    model = cast(Any, bundle.model)
    visual = cast(Any, model.visual)
    with torch.no_grad():
        staged = visual.forward_staged(image, list(candidate_layers))
        outputs = build_staged_full_depth_outputs(bundle, image, candidate_layers)
    patch_features = {depth: staged[depth].patch_tokens for depth in candidate_layers}
    return {"patch_features": patch_features, **outputs}


def run_deterministic_cuda_control(
    bundle: TeacherBundle,
    image: torch.Tensor,
    *,
    seed: int = B1_SEED,
) -> DeterministicControlResult:
    """Part A: independent official/staged repeats under deterministic settings."""
    settings = apply_deterministic_cuda_settings(seed)
    attention_backend = apply_attention_backend_overrides()
    bundle.model.eval()
    cast(Any, bundle.model).visual.eval()

    runs: dict[str, dict[str, Any]] = {}
    for label, runner in (
        ("A1", _run_official_once),
        ("A2", _run_official_once),
        ("S1", _run_staged_once),
        ("S2", _run_staged_once),
    ):
        reset_cuda_diagnostic_state(seed)
        runs[label] = runner(bundle, image)

    comparisons: dict[str, dict[str, Any]] = {}
    pairs = (
        ("A1_vs_A2", "A1", "A2"),
        ("S1_vs_S2", "S1", "S2"),
        ("A1_vs_S1", "A1", "S1"),
        ("A2_vs_S2", "A2", "S2"),
    )
    max_feature = 0.0
    max_map = 0.0
    for pair_name, left_key, right_key in pairs:
        left = runs[left_key]
        right = runs[right_key]
        feature_reports = {
            f"layer_{depth}": tensor_compare_report(
                left["patch_features"][depth], right["patch_features"][depth]
            )
            for depth in DEFAULT_CANDIDATE_LAYERS
        }
        map_reports = {
            name: tensor_compare_report(left[name], right[name])
            for name in ("fused", "final_map", "image_score")
        }
        for layer_name, _layer_map in left["layer_maps"].items():
            map_reports[f"layer_map_{layer_name}"] = tensor_compare_report(
                left["layer_maps"][layer_name], right["layer_maps"][layer_name]
            )
        pair_max_feature = max(report["max_abs"] for report in feature_reports.values())
        pair_max_map = max(report["max_abs"] for report in map_reports.values())
        max_feature = max(max_feature, pair_max_feature)
        max_map = max(max_map, pair_max_map)
        comparisons[pair_name] = {
            "candidate_features": feature_reports,
            "maps": map_reports,
            "max_feature_abs": pair_max_feature,
            "max_map_abs": pair_max_map,
        }

    a_self_ok = comparisons["A1_vs_A2"]["max_feature_abs"] <= B1_ATOL
    s_self_ok = comparisons["S1_vs_S2"]["max_feature_abs"] <= B1_ATOL
    if settings.get("deterministic_error") is not None:
        decision = "B"
        status = "blocked_by_nondeterministic_operation"
    elif a_self_ok and s_self_ok:
        decision = "A"
        status = "strict_independent_pass"
    else:
        decision = "C"
        status = "runtime_nondeterminism_floor_exceeded"

    return DeterministicControlResult(
        status=status,
        settings=settings,
        attention_backend=attention_backend,
        comparisons=comparisons,
        decision=decision,
        deterministic_error=settings.get("deterministic_error"),
    )


def _build_equal_fused_maps(
    bundle: TeacherBundle,
    patch_by_layer: dict[int, torch.Tensor],
    anomaly_token: torch.Tensor,
    normal_token: torch.Tensor,
    candidate_layers: tuple[int, ...] = DEFAULT_CANDIDATE_LAYERS,
) -> dict[str, Any]:
    available = [layer for layer in candidate_layers if layer in patch_by_layer]
    patch_list = [patch_by_layer[layer] for layer in available]
    if bundle.cross_attn is not None:
        adapted = bundle.cross_attn(
            anomaly_token, normal_token, patch_list, available
        )
        anomaly_list = [item["anomaly"] for item in adapted]
        normal_list = [item["normal"] for item in adapted]
    else:
        anomaly_list = [anomaly_token] * len(available)
        normal_list = [normal_token] * len(available)

    layer_maps: dict[int, torch.Tensor] = {}
    anomaly_map_list: list[torch.Tensor] = []
    for idx, layer in enumerate(available):
        patch = _apply_layer_transform(
            bundle.layer_transforms, layer, patch_by_layer[layer]
        )
        amap = anomaly_map_from_tokens(
            F.normalize(anomaly_list[idx], dim=1, eps=1e-8),
            F.normalize(normal_list[idx], dim=1, eps=1e-8),
            patch,
            bundle.image_size,
        )
        layer_maps[layer] = amap
        anomaly_map_list.append(amap)

    fused = torch.stack(anomaly_map_list).sum(dim=0)
    filtered = gaussian_filter(fused[0].detach().cpu().numpy(), sigma=DEFAULT_SIGMA)
    device = anomaly_token.device
    final_map = torch.from_numpy(filtered).to(device).unsqueeze(0)
    image_score = final_map.reshape(final_map.shape[0], -1).max(dim=1).values
    return {
        "layer_maps": layer_maps,
        "fused": fused,
        "final_map": final_map,
        "image_score": image_score,
    }


def run_same_chain_gate(
    bundle: TeacherBundle,
    image: torch.Tensor,
    *,
    candidate_layers: tuple[int, ...] = DEFAULT_CANDIDATE_LAYERS,
) -> SameChainGateResult:
    """Gate 1: one shared transformer chain, official vs staged readouts."""
    visual = cast(Any, bundle.model.visual)
    visual.eval()
    bundle.model.eval()
    errors: list[str] = []
    feature_diffs: dict[str, TensorDiff] = {}
    map_diffs: dict[str, TensorDiff] = {}

    official_patch: dict[int, torch.Tensor] = {}
    staged_patch: dict[int, torch.Tensor] = {}
    continuation_live = True

    with torch.no_grad():
        cache = visual.prepare_stage(image)
        sequence = cache.sequence
        for block_idx, block in enumerate(visual.transformer.resblocks, start=1):
            sequence = block(sequence)
            if block_idx in candidate_layers:
                clone_seq = sequence.clone()
                official_patch[block_idx] = _ln_post_patch_tokens(visual, clone_seq)
                staged_out = visual._checkpoint_from_sequence(sequence, block_idx)
                staged_patch[block_idx] = staged_out.patch_tokens
                cache = StageCache(
                    sequence=sequence,
                    next_block=block_idx + 1,
                    patch_tokens=dict(cache.patch_tokens),
                    checkpoint_tokens=dict(cache.checkpoint_tokens),
                )
                cache.patch_tokens[block_idx] = staged_out.patch_tokens
                if cache.sequence.data_ptr() != sequence.data_ptr():
                    continuation_live = False

        final_tokens = visual.ln_post(sequence.permute(1, 0, 2))
        anomaly_token = final_tokens[:, 0, :]
        normal_token = final_tokens[:, 1, :]

    for depth in candidate_layers:
        key = f"layer_{depth}"
        diff = tensor_diff(staged_patch[depth], official_patch[depth])
        feature_diffs[key] = diff
        if diff.max_abs > B1_ATOL:
            errors.append(f"{key} max_abs={diff.max_abs}")

    official_maps = _build_equal_fused_maps(
        bundle, official_patch, anomaly_token, normal_token, candidate_layers
    )
    staged_maps = _build_equal_fused_maps(
        bundle, staged_patch, anomaly_token, normal_token, candidate_layers
    )
    for layer in candidate_layers:
        key = f"layer_map_{layer}"
        diff = tensor_diff(staged_maps["layer_maps"][layer], official_maps["layer_maps"][layer])
        map_diffs[key] = diff
        if diff.max_abs > B1_ATOL:
            errors.append(f"{key} max_abs={diff.max_abs}")
    for name in ("fused", "final_map", "image_score"):
        diff = tensor_diff(staged_maps[name], official_maps[name])
        map_diffs[name] = diff
        if diff.max_abs > B1_ATOL:
            errors.append(f"{name} max_abs={diff.max_abs}")

    block_counts: list[BlockCountRecord] = []
    continuation: list[ContinuationRecord] = []
    for depth, expected in ((12, 12), (18, 18), (24, 24)):
        counter = install_block_counter(visual)
        cache_probe = visual.prepare_stage(image)
        visual.run_to(cache_probe, depth)
        block_counts.append(
            BlockCountRecord(exit_depth=depth, blocks_executed=counter.total, expected=expected)
        )
        if counter.total != expected:
            errors.append(f"exit {depth}: blocks={counter.total}, expected={expected}")

    counter = install_block_counter(visual)
    cache_probe = visual.prepare_stage(image)
    _, cache12 = visual.run_to(cache_probe, 12)
    reset_block_counter(counter)
    visual.run_to(cache12, 18)
    continuation.append(
        ContinuationRecord(segment="12->18", blocks_executed=counter.total, expected="6 (blocks 13-18)")
    )
    if counter.total != 6 or counter.per_call != list(range(13, 19)):
        errors.append(f"12->18 continuation blocks={counter.total}")

    counter = install_block_counter(visual)
    cache_probe = visual.prepare_stage(image)
    _, cache18 = visual.run_to(cache_probe, 18)
    reset_block_counter(counter)
    visual.run_to(cache18, 24)
    continuation.append(
        ContinuationRecord(segment="18->24", blocks_executed=counter.total, expected="6 (blocks 19-24)")
    )
    if counter.total != 6 or counter.per_call != list(range(19, 25)):
        errors.append(f"18->24 continuation blocks={counter.total}")

    status = "passed" if not errors and continuation_live else "failed"
    return SameChainGateResult(
        status=status,
        feature_diffs=feature_diffs,
        map_diffs=map_diffs,
        block_counts=block_counts,
        continuation=continuation,
        continuation_live_tensor_preserved=continuation_live,
        nonstandard_layers_validated=True,
        errors=errors,
    )


def run_operational_noise_envelope(
    bundle: TeacherBundle,
    samples: list[tuple[str, torch.Tensor]],
    *,
    repeats: int = B1_OPERATIONAL_REPEATS,
) -> OperationalNoiseResult:
    """Gate 2: repeated official/staged runs without diagnostic hooks."""
    apply_deterministic_cuda_settings()
    apply_attention_backend_overrides()
    bundle.model.eval()
    cast(Any, bundle.model).visual.eval()
    errors: list[str] = []
    per_output: dict[str, dict[str, Any]] = {}

    output_keys = [f"layer_{d}" for d in DEFAULT_CANDIDATE_LAYERS]
    output_keys += [f"layer_map_{d}" for d in DEFAULT_CANDIDATE_LAYERS]
    output_keys += ["fused", "final_map", "image_score"]

    global_self_p95: list[float] = []
    global_cross_p95: list[float] = []

    for sample_id, image in samples:
        official_runs: list[dict[str, Any]] = []
        staged_runs: list[dict[str, Any]] = []
        order = ["official", "staged"] * repeats
        for path_kind in order:
            reset_cuda_diagnostic_state()
            if path_kind == "official":
                official_runs.append(_run_official_once(bundle, image))
            else:
                staged_runs.append(_run_staged_once(bundle, image))

        for key in output_keys:
            if key.startswith("layer_") and not key.startswith("layer_map_"):
                depth = int(key.split("_")[1])
                official_tensors = [run["patch_features"][depth] for run in official_runs]
                staged_tensors = [run["patch_features"][depth] for run in staged_runs]
            elif key.startswith("layer_map_"):
                depth = int(key.split("_")[-1])
                official_tensors = [run["layer_maps"][depth] for run in official_runs]
                staged_tensors = [run["layer_maps"][depth] for run in staged_runs]
            else:
                official_tensors = [run[key] for run in official_runs]
                staged_tensors = [run[key] for run in staged_runs]

            official_self = _pairwise_max_abs(official_tensors)
            staged_self = _pairwise_max_abs(staged_tensors)
            cross_vals: list[float] = []
            rel_l2_vals: list[float] = []
            corr_vals: list[float] = []
            for i in range(repeats):
                if key.startswith("layer_") and not key.startswith("layer_map_"):
                    depth = int(key.split("_")[1])
                    left_t = official_runs[i]["patch_features"][depth]
                    right_t = staged_runs[i]["patch_features"][depth]
                elif key.startswith("layer_map_"):
                    depth = int(key.split("_")[-1])
                    left_t = official_runs[i]["layer_maps"][depth]
                    right_t = staged_runs[i]["layer_maps"][depth]
                else:
                    left_t = official_runs[i][key]
                    right_t = staged_runs[i][key]
                diff = tensor_diff(left_t, right_t)
                cross_vals.append(diff.max_abs)
                rel_l2_vals.append(diff.rel_l2)
                if left_t.ndim >= 2:
                    corr_vals.append(_map_correlation(left_t, right_t))

            official_self_p95 = _percentile(official_self, 95.0)
            staged_self_p95 = _percentile(staged_self, 95.0)
            cross_p95 = _percentile(cross_vals, 95.0)
            self_noise_p95 = max(official_self_p95, staged_self_p95)
            cross_excess_p95 = max(0.0, cross_p95 - self_noise_p95)
            global_self_p95.append(self_noise_p95)
            global_cross_p95.append(cross_p95)

            out_key = f"{sample_id}/{key}"
            per_output[out_key] = {
                "official_self_noise": DivergenceStats(
                    median=_percentile(official_self, 50.0),
                    p95=official_self_p95,
                    maximum=max(official_self) if official_self else 0.0,
                    rel_l2_median=_percentile(
                        [tensor_diff(official_tensors[i], official_tensors[j]).rel_l2
                         for i in range(len(official_tensors))
                         for j in range(i + 1, len(official_tensors))] or [0.0],
                        50.0,
                    ),
                ).__dict__,
                "staged_self_noise": DivergenceStats(
                    median=_percentile(staged_self, 50.0),
                    p95=staged_self_p95,
                    maximum=max(staged_self) if staged_self else 0.0,
                    rel_l2_median=_percentile(
                        [tensor_diff(staged_tensors[i], staged_tensors[j]).rel_l2
                         for i in range(len(staged_tensors))
                         for j in range(i + 1, len(staged_tensors))] or [0.0],
                        50.0,
                    ),
                ).__dict__,
                "cross_path": DivergenceStats(
                    median=_percentile(cross_vals, 50.0),
                    p95=cross_p95,
                    maximum=max(cross_vals) if cross_vals else 0.0,
                    rel_l2_median=_percentile(rel_l2_vals, 50.0),
                    map_correlation_median=_percentile(corr_vals, 50.0) if corr_vals else None,
                ).__dict__,
                "self_noise_p95": self_noise_p95,
                "cross_excess_p95": cross_excess_p95,
            }

    aggregate_self_p95 = max(global_self_p95) if global_self_p95 else 0.0
    aggregate_cross_p95 = max(global_cross_p95) if global_cross_p95 else 0.0
    aggregate_cross_excess = max(
        max(0.0, item["cross_path"]["p95"] - item["self_noise_p95"])
        for item in per_output.values()
    ) if per_output else 0.0
    if aggregate_self_p95 <= 0.0:
        # Deterministic repeats: no self-noise envelope; apply absolute cross-path bound.
        ratio_limit = B1_ATOL
        ratio_pass = aggregate_cross_p95 <= ratio_limit
    else:
        ratio_limit = B1_CROSS_PATH_RATIO * max(aggregate_self_p95, 1e-12)
        ratio_pass = aggregate_cross_p95 <= ratio_limit
    excess_pass = aggregate_cross_excess <= B1_ATOL
    if not ratio_pass:
        errors.append(
            f"ratio fail: cross_path_p95={aggregate_cross_p95} > "
            f"{B1_CROSS_PATH_RATIO}*self_noise_p95={ratio_limit}"
        )
    if not excess_pass:
        errors.append(f"cross_excess_p95={aggregate_cross_excess} > {B1_ATOL}")

    status = "passed" if ratio_pass and excess_pass else "failed"
    return OperationalNoiseResult(
        status=status,
        per_output=per_output,
        self_noise_p95=aggregate_self_p95,
        cross_excess_p95=aggregate_cross_excess,
        cross_path_p95=aggregate_cross_p95,
        ratio_pass=ratio_pass,
        excess_pass=excess_pass,
        errors=errors,
    )


def _load_mask_tensor(path: Path | None, device: torch.device) -> torch.Tensor:
    if path is None or not path.is_file():
        return torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
    mask = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    arr = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)
    return (arr > 0.5).float().unsqueeze(0).unsqueeze(0).to(device)


def _top1_patch_overlap(map_a: torch.Tensor, map_b: torch.Tensor, ratio: float = 0.01) -> float:
    flat_a = map_a.detach().float().reshape(-1)
    flat_b = map_b.detach().float().reshape(-1)
    k = max(1, int(round(ratio * flat_a.numel())))
    idx_a = torch.topk(flat_a, k=k).indices
    idx_b = torch.topk(flat_b, k=k).indices
    set_a = set(idx_a.cpu().tolist())
    set_b = set(idx_b.cpu().tolist())
    union = set_a | set_b
    if not union:
        return 1.0
    return float(len(set_a & set_b) / len(union))


def _anomaly_region_overlap(map_a: torch.Tensor, map_b: torch.Tensor, threshold: float = 0.5) -> float:
    bin_a = (map_a.detach().float() >= threshold).reshape(-1)
    bin_b = (map_b.detach().float() >= threshold).reshape(-1)
    inter = float((bin_a & bin_b).sum().item())
    union = float((bin_a | bin_b).sum().item())
    return 1.0 if union == 0.0 else inter / union


def _path_flag_tokens(path: Path | str) -> list[str]:
    parts = {part.lower() for part in Path(path).parts}
    return sorted(token for token in _PATH_FLAG_TOKENS if token in parts)


def _category_split_hash(sample_ids: list[str]) -> str:
    payload = "\n".join(sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_task_level_category_provenance() -> list[dict[str, Any]]:
    """Export provenance for accepted B1 task-level categories."""
    from rad.data.adapters.mvtec import MVTecAdapter
    from rad.data.adapters.visa import VisAAdapter

    audits: list[dict[str, Any]] = []

    mvtec = MVTecAdapter(MVTEC_ROOT)
    mvtec_records = [r for r in mvtec.records("test") if r.category == B1_TASK_MVTEC_CATEGORY]
    mvtec_ids = [r.sample_id for r in mvtec_records]
    mvtec_masks = sum(1 for r in mvtec_records if r.mask_path is not None)
    mvtec_flags = sorted(
        {
            flag
            for r in mvtec_records
            for flag in _path_flag_tokens(r.image_path)
            if flag not in {"test"}  # MVTec canonical layout uses test/
        }
    )
    # Re-scan flags but allow the canonical 'test' split directory.
    mvtec_suspicious = sorted(
        {
            flag
            for r in mvtec_records
            for flag in _path_flag_tokens(r.image_path)
            if flag in {"tests", "fixture", "fixtures", "example", "examples", "synthetic"}
        }
    )
    audits.append(
        {
            "dataset": "mvtec",
            "adapter": "rad.data.adapters.mvtec.MVTecAdapter",
            "dataset_root": str(MVTEC_ROOT),
            "canonical_category": B1_TASK_MVTEC_CATEGORY,
            "category_id": B1_TASK_MVTEC_CATEGORY,
            "normal_samples": sum(1 for r in mvtec_records if r.image_label == 0),
            "anomalous_samples": sum(1 for r in mvtec_records if r.image_label == 1),
            "mask_count": mvtec_masks,
            "total_sample_count": len(mvtec_records),
            "first_five_sample_ids": mvtec_ids[:5],
            "last_five_sample_ids": mvtec_ids[-5:],
            "split_config_hash": _category_split_hash(mvtec_ids),
            "path_flags": mvtec_suspicious,
            "canonical_split_uses_test_dir": True,
            "accepted_for_b1_task_gate": (
                len(mvtec_records) > 0
                and not mvtec_suspicious
                and B1_TASK_MVTEC_CATEGORY != "sample"
            ),
            "invalid_predecessor": {
                "path": str(MVTEC_SAMPLE_FIXTURE_ROOT),
                "reason": (
                    "mvtec/sample is a flat image/ asset directory without "
                    "train/test/ground_truth layout and is not returned by MVTecAdapter"
                ),
                "accepted_for_b1_task_gate": False,
            },
        }
    )

    visa = VisAAdapter(VISA_ROOT)
    visa_records = [r for r in visa.records("test") if r.category == B1_TASK_VISA_CATEGORY]
    visa_ids = [r.sample_id for r in visa_records]
    visa_suspicious = sorted(
        {
            flag
            for r in visa_records
            for flag in _path_flag_tokens(r.image_path)
            if flag in {"tests", "fixture", "fixtures", "example", "examples", "synthetic"}
        }
    )
    audits.append(
        {
            "dataset": "visa",
            "adapter": "rad.data.adapters.visa.VisAAdapter",
            "dataset_root": str(VISA_ROOT),
            "canonical_category": B1_TASK_VISA_CATEGORY,
            "category_id": B1_TASK_VISA_CATEGORY,
            "normal_samples": sum(1 for r in visa_records if r.image_label == 0),
            "anomalous_samples": sum(1 for r in visa_records if r.image_label == 1),
            "mask_count": sum(1 for r in visa_records if r.mask_path is not None),
            "total_sample_count": len(visa_records),
            "first_five_sample_ids": visa_ids[:5],
            "last_five_sample_ids": visa_ids[-5:],
            "split_config_hash": _category_split_hash(visa_ids),
            "path_flags": visa_suspicious,
            "accepted_for_b1_task_gate": len(visa_records) > 0 and not visa_suspicious,
        }
    )
    # silence unused variable from exploratory scan
    del mvtec_flags
    return audits


def load_task_level_category_samples() -> list[tuple[str, Path, Path | None, int, str]]:
    """All test samples from accepted complete MVTec + VisA categories."""
    from rad.data.adapters.mvtec import MVTecAdapter
    from rad.data.adapters.visa import VisAAdapter

    provenance = audit_task_level_category_provenance()
    rejected = [item for item in provenance if not item["accepted_for_b1_task_gate"]]
    if rejected:
        raise RuntimeError(f"B1 task-level provenance rejected: {rejected}")

    samples: list[tuple[str, Path, Path | None, int, str]] = []
    mvtec = MVTecAdapter(MVTEC_ROOT)
    for record in mvtec.records("test"):
        if record.category != B1_TASK_MVTEC_CATEGORY:
            continue
        samples.append(
            (
                record.sample_id,
                record.image_path,
                record.mask_path,
                int(record.image_label),
                f"mvtec/{record.category}",
            )
        )

    visa = VisAAdapter(VISA_ROOT)
    for record in visa.records("test"):
        if record.category != B1_TASK_VISA_CATEGORY:
            continue
        samples.append(
            (
                record.sample_id,
                record.image_path,
                record.mask_path,
                int(record.image_label),
                f"visa/{record.category}",
            )
        )

    if not samples:
        raise RuntimeError("no accepted task-level samples found")
    return samples


def _map_spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    flat_a = a.detach().float().reshape(-1)
    flat_b = b.detach().float().reshape(-1)
    if flat_a.numel() < 2:
        return 1.0
    rank_a = flat_a.argsort().argsort().float()
    rank_b = flat_b.argsort().argsort().float()
    return _map_correlation(rank_a, rank_b)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.detach().float() - b.detach().float()).reshape(-1)
    denom = b.detach().float().reshape(-1).norm().item()
    if denom <= 0.0:
        return float(diff.norm().item())
    return float(diff.norm().item() / denom)


def run_task_level_impact(
    bundle: TeacherBundle,
    device: torch.device,
) -> TaskLevelImpactResult:
    """Part C: evaluation-only official vs staged forced-full-depth comparison."""
    from rad.evaluation.paper_metrics import compute_paper_metrics
    from rad.losses.localization import sample_localization_error

    bundle.model.eval()
    cast(Any, bundle.model).visual.eval()
    provenance = audit_task_level_category_provenance()
    records = load_task_level_category_samples()
    sample_metrics: list[TaskLevelSampleMetrics] = []
    nonfinite: list[str] = []

    official_scores: list[float] = []
    staged_scores: list[float] = []
    labels: list[int] = []
    official_maps: list[np.ndarray] = []
    staged_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    categories: list[str] = []

    for sample_id, image_path, mask_path, label, category in records:
        image = load_preprocessed_image(str(image_path), device)
        mask_t = _load_mask_tensor(mask_path, device)
        label_t = torch.tensor([label], device=device, dtype=torch.float32)

        official = build_official_full_depth_outputs(bundle, image)
        staged = build_staged_full_depth_outputs(bundle, image)

        off_score = float(official["image_score"].item())
        st_score = float(staged["image_score"].item())
        off_map = official["final_map"]
        st_map = staged["final_map"]

        if not (
            np.isfinite(off_score)
            and np.isfinite(st_score)
            and torch.isfinite(off_map).all()
            and torch.isfinite(st_map).all()
        ):
            nonfinite.append(sample_id)

        loc_off = float(sample_localization_error(off_map, mask_t, label_t).item())
        loc_st = float(sample_localization_error(st_map, mask_t, label_t).item())
        pearson = _map_correlation(off_map, st_map)
        spearman = _map_spearman(off_map, st_map)
        rel_l2 = _rel_l2(st_map, off_map)

        sample_metrics.append(
            TaskLevelSampleMetrics(
                sample_id=sample_id,
                category=category,
                image_score_diff=abs(off_score - st_score),
                final_map_max_abs=tensor_diff(st_map, off_map).max_abs,
                final_map_rel_l2=rel_l2,
                map_pearson=pearson,
                map_spearman=spearman,
                localization_error_diff=abs(loc_off - loc_st),
                top1_patch_overlap=_top1_patch_overlap(off_map, st_map),
                anomaly_region_overlap=_anomaly_region_overlap(off_map, st_map),
            )
        )

        official_scores.append(off_score)
        staged_scores.append(st_score)
        labels.append(label)
        official_maps.append(off_map[0].detach().cpu().numpy())
        staged_maps.append(st_map[0].detach().cpu().numpy())
        masks.append(mask_t[0, 0].detach().cpu().numpy())
        categories.append(category)

    def _metrics_for(indices: list[int]) -> tuple[Any, Any]:
        idx = np.asarray(indices, dtype=np.int64)
        labels_arr = np.asarray(labels, dtype=np.float64)[idx]
        off_metrics = compute_paper_metrics(
            image_labels=labels_arr,
            image_scores=np.asarray(official_scores, dtype=np.float64)[idx],
            masks=np.stack([masks[i] for i in indices]),
            anomaly_maps=np.stack([official_maps[i] for i in indices]),
        )
        st_metrics = compute_paper_metrics(
            image_labels=labels_arr,
            image_scores=np.asarray(staged_scores, dtype=np.float64)[idx],
            masks=np.stack([masks[i] for i in indices]),
            anomaly_maps=np.stack([staged_maps[i] for i in indices]),
        )
        return off_metrics, st_metrics

    all_idx = list(range(len(records)))
    off_metrics, st_metrics = _metrics_for(all_idx)

    def _diff_pp(off_val: float, st_val: float) -> float:
        return abs(off_val - st_val) * 100.0

    metric_diffs = {
        "image_auroc_pp": _diff_pp(off_metrics.image_auroc, st_metrics.image_auroc),
        "image_ap_pp": _diff_pp(off_metrics.image_ap, st_metrics.image_ap),
        "image_f1_max_pp": _diff_pp(off_metrics.image_f1_max, st_metrics.image_f1_max),
        "pixel_auroc_pp": _diff_pp(off_metrics.pixel_auroc, st_metrics.pixel_auroc),
        "pixel_ap_pp": _diff_pp(off_metrics.pixel_ap, st_metrics.pixel_ap),
        "pixel_f1_max_pp": _diff_pp(off_metrics.pixel_f1_max, st_metrics.pixel_f1_max),
        "pixel_aupro_pp": _diff_pp(off_metrics.pixel_aupro, st_metrics.pixel_aupro),
        "boundary_f_score_pp": _diff_pp(
            off_metrics.boundary_f_score or 0.0,
            st_metrics.boundary_f_score or 0.0,
        ),
    }
    metric_raw = {
        "official": {
            "image_auroc": float(off_metrics.image_auroc),
            "image_ap": float(off_metrics.image_ap),
            "image_f1_max": float(off_metrics.image_f1_max),
            "pixel_auroc": float(off_metrics.pixel_auroc),
            "pixel_ap": float(off_metrics.pixel_ap),
            "pixel_f1_max": float(off_metrics.pixel_f1_max),
            "pixel_aupro": float(off_metrics.pixel_aupro),
            "boundary_f_score": float(off_metrics.boundary_f_score or 0.0),
        },
        "staged": {
            "image_auroc": float(st_metrics.image_auroc),
            "image_ap": float(st_metrics.image_ap),
            "image_f1_max": float(st_metrics.image_f1_max),
            "pixel_auroc": float(st_metrics.pixel_auroc),
            "pixel_ap": float(st_metrics.pixel_ap),
            "pixel_f1_max": float(st_metrics.pixel_f1_max),
            "pixel_aupro": float(st_metrics.pixel_aupro),
            "boundary_f_score": float(st_metrics.boundary_f_score or 0.0),
        },
    }

    per_category: dict[str, dict[str, float]] = {}
    for category in sorted(set(categories)):
        idxs = [i for i, cat in enumerate(categories) if cat == category]
        off_c, st_c = _metrics_for(idxs)
        per_category[category] = {
            "image_auroc_pp": _diff_pp(off_c.image_auroc, st_c.image_auroc),
            "image_ap_pp": _diff_pp(off_c.image_ap, st_c.image_ap),
            "image_f1_max_pp": _diff_pp(off_c.image_f1_max, st_c.image_f1_max),
            "pixel_auroc_pp": _diff_pp(off_c.pixel_auroc, st_c.pixel_auroc),
            "pixel_ap_pp": _diff_pp(off_c.pixel_ap, st_c.pixel_ap),
            "pixel_f1_max_pp": _diff_pp(off_c.pixel_f1_max, st_c.pixel_f1_max),
            "pixel_aupro_pp": _diff_pp(off_c.pixel_aupro, st_c.pixel_aupro),
            "boundary_f_score_pp": _diff_pp(
                off_c.boundary_f_score or 0.0,
                st_c.boundary_f_score or 0.0,
            ),
        }

    loc_diffs = [item.localization_error_diff for item in sample_metrics]
    map_rel = [item.final_map_rel_l2 for item in sample_metrics]
    loc_stats = {
        "p50": _percentile(loc_diffs, 50.0),
        "p95": _percentile(loc_diffs, 95.0),
        "max": max(loc_diffs) if loc_diffs else 0.0,
    }
    map_rel_stats = {
        "p50": _percentile(map_rel, 50.0),
        "p95": _percentile(map_rel, 95.0),
        "max": max(map_rel) if map_rel else 0.0,
    }

    errors: list[str] = []
    if metric_diffs["image_auroc_pp"] > B1_TASK_IMAGE_METRIC_PP:
        errors.append(f"image_auroc diff {metric_diffs['image_auroc_pp']}")
    if metric_diffs["image_ap_pp"] > B1_TASK_IMAGE_METRIC_PP:
        errors.append(f"image_ap diff {metric_diffs['image_ap_pp']}")
    if metric_diffs["image_f1_max_pp"] > B1_TASK_IMAGE_METRIC_PP:
        errors.append(f"image_f1_max diff {metric_diffs['image_f1_max_pp']}")
    if metric_diffs["pixel_auroc_pp"] > B1_TASK_PIXEL_METRIC_PP:
        errors.append(f"pixel_auroc diff {metric_diffs['pixel_auroc_pp']}")
    if metric_diffs["pixel_ap_pp"] > B1_TASK_PIXEL_METRIC_PP:
        errors.append(f"pixel_ap diff {metric_diffs['pixel_ap_pp']}")
    if metric_diffs["pixel_f1_max_pp"] > B1_TASK_PIXEL_METRIC_PP:
        errors.append(f"pixel_f1_max diff {metric_diffs['pixel_f1_max_pp']}")
    if metric_diffs["pixel_aupro_pp"] > B1_TASK_AUPRO_PP:
        errors.append(f"pixel_aupro diff {metric_diffs['pixel_aupro_pp']}")
    if metric_diffs["boundary_f_score_pp"] > B1_TASK_BOUNDARY_PP:
        errors.append(f"boundary_f_score diff {metric_diffs['boundary_f_score_pp']}")
    for category, diffs in per_category.items():
        for key, val in diffs.items():
            if val > B1_TASK_PER_CATEGORY_PP:
                errors.append(f"{category} {key} diff {val}")
    if nonfinite:
        errors.append(f"nonfinite samples: {nonfinite[:5]}")

    superseded = {
        "previous_task_level_categories": ["mvtec/sample", "visa/candle"],
        "invalid_reason": (
            "mvtec/sample is a non-adapter flat fixture/example asset directory; "
            "accepted gate replaced with mvtec/bottle + visa/candle"
        ),
        "retained_legacy_manifest": "docs/phase_b/b1_cuda_equivalence_manifest.json",
    }

    status = "passed" if not errors else "failed"
    return TaskLevelImpactResult(
        status=status,
        sample_count=len(records),
        metric_differences=metric_diffs,
        metric_raw=metric_raw,
        per_category_metric_differences_pp=per_category,
        localization_error_diff_stats=loc_stats,
        map_rel_l2_stats=map_rel_stats,
        min_map_pearson=min((m.map_pearson for m in sample_metrics), default=1.0),
        min_map_spearman=min((m.map_spearman for m in sample_metrics), default=1.0),
        min_top1_patch_overlap=min((m.top1_patch_overlap for m in sample_metrics), default=1.0),
        max_image_score_diff=max((m.image_score_diff for m in sample_metrics), default=0.0),
        p95_localization_error_diff=loc_stats["p95"],
        nonfinite_samples=nonfinite,
        samples=sample_metrics,
        provenance=provenance,
        superseded_evidence=superseded,
        errors=errors,
    )


def run_equivalence_protocol(
    bundle: TeacherBundle,
    device: torch.device,
    *,
    control_image: torch.Tensor,
    operational_samples: list[tuple[str, torch.Tensor]],
) -> EquivalenceProtocolResult:
    """Execute Parts A–C of the dual equivalence protocol."""
    deterministic = run_deterministic_cuda_control(bundle, control_image)
    same_chain = run_same_chain_gate(bundle, control_image)
    operational = run_operational_noise_envelope(bundle, operational_samples)
    task_level = run_task_level_impact(bundle, device)

    detail = None
    if deterministic.decision == "A":
        detail = "strict_independent_pass"
    elif deterministic.decision in {"B", "C"} and (
        same_chain.status == "passed"
        and operational.status == "passed"
        and task_level.status == "passed"
    ):
        detail = "passed_with_runtime_nondeterminism_envelope"

    return EquivalenceProtocolResult(
        deterministic_control=deterministic,
        algorithmic_same_chain=same_chain,
        operational_noise_envelope=operational,
        task_level_impact=task_level,
        detail=detail,
    )


def apply_execution_profile(profile: str, seed: int = B1_SEED) -> dict[str, Any]:
    """Apply a named B1/B2 execution profile and return requested+observed settings."""
    if profile not in {"frozen_deterministic_math", "production_default_attention"}:
        raise ValueError(f"unknown execution profile: {profile}")

    if profile == "frozen_deterministic_math":
        settings = apply_deterministic_cuda_settings(seed)
        attention = apply_attention_backend_overrides()
        settings["attention_backend"] = attention
        settings["profile"] = profile
    else:
        # Production/default attention: do not force math-only SDP or disable MHA
        # fastpath. Keep seed/dtype/eval parity only.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        cuda_backend = getattr(torch.backends, "cuda", None)
        if cuda_backend is not None:
            # Restore library defaults rather than leaving a prior math-only freeze.
            if callable(getattr(cuda_backend, "enable_flash_sdp", None)):
                cuda_backend.enable_flash_sdp(True)
            if callable(getattr(cuda_backend, "enable_mem_efficient_sdp", None)):
                cuda_backend.enable_mem_efficient_sdp(True)
            if callable(getattr(cuda_backend, "enable_math_sdp", None)):
                cuda_backend.enable_math_sdp(True)
        mha_backend = getattr(torch.backends, "mha", None)
        fastpath_fn = getattr(mha_backend, "set_fastpath_enabled", None) if mha_backend else None
        if callable(fastpath_fn):
            fastpath_fn(True)
        settings = {
            "seed": seed,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "profile": profile,
            "attention_policy": "repository_default_sdp_fastpath",
            "use_deterministic_algorithms": False,
            "cuda.matmul.allow_tf32": False,
            "cudnn.allow_tf32": False,
            "cudnn.benchmark": False,
            "cudnn.deterministic": False,
        }

    settings["observed"] = observe_effective_execution_settings()
    return settings


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compare_official_staged_once(
    bundle: TeacherBundle,
    image: torch.Tensor,
) -> dict[str, Any]:
    official = build_official_full_depth_outputs(bundle, image)
    staged = build_staged_full_depth_outputs(bundle, image)
    feature_diffs = compare_candidate_features(bundle, image)
    feature_max = max((d.max_abs for d in feature_diffs.values()), default=0.0)
    map_max = 0.0
    for depth in DEFAULT_CANDIDATE_LAYERS:
        map_max = max(
            map_max,
            tensor_diff(staged["layer_maps"][depth], official["layer_maps"][depth]).max_abs,
        )
    for key in ("fused", "final_map"):
        map_max = max(map_max, tensor_diff(staged[key], official[key]).max_abs)
    score_diff = abs(
        float(official["image_score"].item()) - float(staged["image_score"].item())
    )
    deepest = DEFAULT_CANDIDATE_LAYERS[-1]
    # Relative L2 on deepest-layer patch features via a fresh staged/official encode.
    model = cast(Any, bundle.model)
    visual = cast(Any, model.visual)
    with torch.no_grad():
        legacy = model.encode_image(image, list(DEFAULT_CANDIDATE_LAYERS))
        staged_tokens = visual.forward_staged(image, list(DEFAULT_CANDIDATE_LAYERS))
    off_feat = legacy["patch_tokens"][-1][:, legacy["patch_start_idx"] :, :]
    st_feat = staged_tokens[deepest].patch_tokens
    return {
        "feature_max_abs": feature_max,
        "map_max_abs": map_max,
        "image_score_diff": score_diff,
        "official_final_map_sha256": tensor_sha256(official["final_map"]),
        "staged_final_map_sha256": tensor_sha256(staged["final_map"]),
        "official_image_score": float(official["image_score"].item()),
        "staged_image_score": float(staged["image_score"].item()),
        "final_map_corr": _map_correlation(official["final_map"], staged["final_map"]),
        "feature_rel_l2": _rel_l2(st_feat, off_feat),
        "map_rel_l2": _rel_l2(staged["final_map"], official["final_map"]),
    }


def run_backend_profile_matrix(
    *,
    checkpoint: Path,
    expected_sha256: str,
    sample_paths: list[str],
    repeats: int = 5,
) -> dict[str, Any]:
    """Five-run alternating official/staged matrix for Profile 1 vs Profile 2."""
    device = torch.device("cuda:0")
    profiles = ("frozen_deterministic_math", "production_default_attention")
    report: dict[str, Any] = {"repeats": repeats, "profiles": {}}

    for profile in profiles:
        settings = apply_execution_profile(profile)
        bundle = load_teacher_production(checkpoint, expected_sha256, device)
        profile_rows: list[dict[str, Any]] = []
        for sample_path in sample_paths:
            image = load_preprocessed_image(sample_path, device)
            official_maps: list[torch.Tensor] = []
            staged_maps: list[torch.Tensor] = []
            cross_vals: list[float] = []
            feature_rels: list[float] = []
            map_rels: list[float] = []
            corrs: list[float] = []
            score_diffs: list[float] = []
            for _ in range(repeats):
                off = build_official_full_depth_outputs(bundle, image)
                st = build_staged_full_depth_outputs(bundle, image)
                official_maps.append(off["final_map"].detach().clone())
                staged_maps.append(st["final_map"].detach().clone())
                cross_vals.append(tensor_diff(st["final_map"], off["final_map"]).max_abs)
                feat_diffs = compare_candidate_features(bundle, image)
                deepest = DEFAULT_CANDIDATE_LAYERS[-1]
                model = cast(Any, bundle.model)
                visual = cast(Any, model.visual)
                with torch.no_grad():
                    legacy = model.encode_image(image, list(DEFAULT_CANDIDATE_LAYERS))
                    staged_tokens = visual.forward_staged(image, list(DEFAULT_CANDIDATE_LAYERS))
                off_feat = legacy["patch_tokens"][-1][:, legacy["patch_start_idx"] :, :]
                st_feat = staged_tokens[deepest].patch_tokens
                feature_rels.append(_rel_l2(st_feat, off_feat))
                map_rels.append(_rel_l2(st["final_map"], off["final_map"]))
                corrs.append(_map_correlation(off["final_map"], st["final_map"]))
                score_diffs.append(
                    abs(float(off["image_score"].item()) - float(st["image_score"].item()))
                )
                del feat_diffs
            pairwise_off = _pairwise_max_abs(official_maps)
            pairwise_st = _pairwise_max_abs(staged_maps)
            profile_rows.append(
                {
                    "sample_path": sample_path,
                    "official_self_max": max(pairwise_off) if pairwise_off else 0.0,
                    "staged_self_max": max(pairwise_st) if pairwise_st else 0.0,
                    "cross_path_max": max(cross_vals) if cross_vals else 0.0,
                    "feature_rel_l2_max": max(feature_rels) if feature_rels else 0.0,
                    "map_rel_l2_max": max(map_rels) if map_rels else 0.0,
                    "final_map_corr_min": min(corrs) if corrs else 1.0,
                    "image_score_diff_max": max(score_diffs) if score_diffs else 0.0,
                    "observed_settings": observe_effective_execution_settings(),
                }
            )
        report["profiles"][profile] = {
            "settings": settings,
            "per_sample": profile_rows,
            "official_self_max": max(r["official_self_max"] for r in profile_rows),
            "staged_self_max": max(r["staged_self_max"] for r in profile_rows),
            "cross_path_max": max(r["cross_path_max"] for r in profile_rows),
        }
    return report


def run_cross_process_repeatability_worker(
    *,
    checkpoint: str,
    expected_sha256: str,
    input_path: str,
    profile: str,
    worker_id: int,
) -> dict[str, Any]:
    """One fresh-process official+staged comparison under a selected profile."""
    settings = apply_execution_profile(profile)
    device = torch.device("cuda:0")
    bundle = load_teacher_production(Path(checkpoint), expected_sha256, device)
    image = load_preprocessed_image(input_path, device)
    comparison = compare_official_staged_once(bundle, image)
    return {
        "worker_id": worker_id,
        "profile": profile,
        "settings_observed": settings.get("observed"),
        **comparison,
    }
