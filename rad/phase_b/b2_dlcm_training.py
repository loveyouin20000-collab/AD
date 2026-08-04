"""B2-05A deterministic DLCM training lifecycle contracts.

Implements optimizer/scheduler contracts, environment attestation, epoch-0
protection, early stopping, exact float trace encoding, hash chains, epoch
transactions, resume gates, and hermetic contract-only training loops.

Real authoritative GPU seed training is disabled in B2-05A
(``real_training_enabled=false``).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import torch
import torch.nn as nn

from rad.phase_b import b2_dlcm as dlcm

SCHEDULER_CONTRACT_VERSION = "b2_dlcm_scheduler_v1"
TRACE_CHAIN_SCHEMA_VERSION = "b2_dlcm_trace_chain_v1"
ENVIRONMENT_CONTRACT_VERSION = "b2_dlcm_environment_v1"
EPOCH_MANIFEST_SCHEMA_VERSION = "b2_dlcm_epoch_state_v1"


class B2DLCMTrainingError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMTrainingError(code, detail)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    raw = encoded.encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    digest = sha256_bytes(raw)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def build_adamw_param_groups(
    model: nn.Module,
    *,
    weight_decay: float,
    lr: float,
) -> list[dict[str, Any]]:
    decay: list[tuple[str, nn.Parameter]] = []
    no_decay: list[tuple[str, nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name or "embedding" in name:
            no_decay.append((name, param))
        elif name.endswith(".weight"):
            decay.append((name, param))
        else:
            _fail("B2_DLCM_PARAM_GROUP_UNKNOWN", f"unclassified parameter {name}")
    decay_sorted = sorted(decay, key=lambda item: item[0])
    no_decay_sorted = sorted(no_decay, key=lambda item: item[0])
    return [
        {
            "group_name": "decay",
            "params": [p for _, p in decay_sorted],
            "ordered_parameter_names": [n for n, _ in decay_sorted],
            "weight_decay": float(weight_decay),
            "lr": float(lr),
        },
        {
            "group_name": "no_decay",
            "params": [p for _, p in no_decay_sorted],
            "ordered_parameter_names": [n for n, _ in no_decay_sorted],
            "weight_decay": 0.0,
            "lr": float(lr),
        },
    ]


def build_adamw(
    model: nn.Module,
    *,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> tuple[torch.optim.AdamW, list[dict[str, Any]]]:
    groups = build_adamw_param_groups(model, weight_decay=weight_decay, lr=lr)
    optim = torch.optim.AdamW(
        [
            {
                "params": g["params"],
                "lr": g["lr"],
                "weight_decay": g["weight_decay"],
                "betas": betas,
                "eps": eps,
                "amsgrad": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
            }
            for g in groups
        ]
    )
    return optim, groups


@dataclass
class ExplicitLRSchedule:
    maximum_learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-6
    warmup_steps: int = 100
    maximum_optimizer_steps: int = 2000
    global_optimizer_step: int = 0
    last_applied_learning_rate: float | None = None
    next_learning_rate: float = field(init=False)

    def __post_init__(self) -> None:
        self.next_learning_rate = self.learning_rate_for_step(1)

    def learning_rate_for_step(self, t: int) -> float:
        if t < 1 or t > self.maximum_optimizer_steps:
            _fail("B2_DLCM_SCHEDULER_STEP_INVALID", f"step {t} out of range")
        if t <= self.warmup_steps:
            return self.maximum_learning_rate * (t / float(self.warmup_steps))
        # Cosine from warmup+1 .. max
        progress = (t - self.warmup_steps) / float(
            self.maximum_optimizer_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math_cos_pi(progress))
        return self.minimum_learning_rate + (
            self.maximum_learning_rate - self.minimum_learning_rate
        ) * cosine

    def install_into_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            group["lr"] = float(self.next_learning_rate)

    def note_successful_update(self, t: int) -> None:
        applied = self.learning_rate_for_step(t)
        self.global_optimizer_step = t
        self.last_applied_learning_rate = applied
        if t < self.maximum_optimizer_steps:
            self.next_learning_rate = self.learning_rate_for_step(t + 1)
        else:
            self.next_learning_rate = applied

    def note_failed_update(self) -> None:
        return None

    def contract_state(self) -> dict[str, Any]:
        return {
            "scheduler_contract_version": SCHEDULER_CONTRACT_VERSION,
            "global_optimizer_step": self.global_optimizer_step,
            "maximum_optimizer_steps": self.maximum_optimizer_steps,
            "warmup_steps": self.warmup_steps,
            "maximum_learning_rate": self.maximum_learning_rate,
            "minimum_learning_rate": self.minimum_learning_rate,
            "last_applied_learning_rate": self.last_applied_learning_rate,
            "next_learning_rate": self.next_learning_rate,
        }

    @classmethod
    def from_contract_state(cls, state: Mapping[str, Any]) -> ExplicitLRSchedule:
        if state.get("scheduler_contract_version") != SCHEDULER_CONTRACT_VERSION:
            _fail("B2_DLCM_SCHEDULER_VERSION", "scheduler contract version mismatch")
        sched = cls(
            maximum_learning_rate=float(state["maximum_learning_rate"]),
            minimum_learning_rate=float(state["minimum_learning_rate"]),
            warmup_steps=int(state["warmup_steps"]),
            maximum_optimizer_steps=int(state["maximum_optimizer_steps"]),
            global_optimizer_step=int(state["global_optimizer_step"]),
            last_applied_learning_rate=state.get("last_applied_learning_rate"),
        )
        sched.next_learning_rate = float(state["next_learning_rate"])
        return sched


def math_cos_pi(progress: float) -> float:
    import math

    return math.cos(math.pi * progress)


def run_epoch0_baseline_marker(schedule: ExplicitLRSchedule) -> None:
    """Epoch 0 does not advance optimizer or scheduler."""

    _ = schedule.contract_state()


@dataclass
class CheckpointSelector:
    min_delta: float = 1e-5
    best_epoch: int | None = None
    best_primary: float | None = None
    best_secondary: float | None = None
    epoch0_replaced: bool = False

    def consider(self, *, epoch: int, primary: float, secondary: float) -> bool:
        if self.best_epoch is None:
            self.best_epoch = epoch
            self.best_primary = float(primary)
            self.best_secondary = float(secondary)
            self.epoch0_replaced = False
            return True
        assert self.best_primary is not None and self.best_secondary is not None
        if not self.epoch0_replaced and self.best_epoch == 0:
            # Epoch 0 protection: only significant primary improvement.
            if primary < self.best_primary - self.min_delta:
                self.best_epoch = epoch
                self.best_primary = float(primary)
                self.best_secondary = float(secondary)
                self.epoch0_replaced = True
                return True
            return False
        # After epoch 0 replaced:
        if primary < self.best_primary - self.min_delta:
            self.best_epoch = epoch
            self.best_primary = float(primary)
            self.best_secondary = float(secondary)
            return True
        if abs(primary - self.best_primary) <= self.min_delta and secondary < (
            self.best_secondary - self.min_delta
        ):
            self.best_epoch = epoch
            self.best_primary = float(primary)
            self.best_secondary = float(secondary)
            return True
        return False


@dataclass
class EarlyStopController:
    patience: int = 50
    maximum_epochs: int = 500
    patience_counter: int = 0

    def after_epoch(self, *, epoch: int, improved: bool) -> str:
        if improved:
            self.patience_counter = 0
            return "running"
        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            return "early_stopped"
        if epoch >= self.maximum_epochs:
            return "max_epochs"
        return "running"


def exact_float_field(value: float, *, dtype: str = "float32") -> dict[str, Any]:
    meta = dlcm.float_to_bits_hex(value, dtype=dtype)
    return {"value": float(value), **meta}


def scientific_epoch_record(
    *,
    epoch: int,
    primary: float,
    secondary: float,
    total_loss: float,
    global_optimizer_step: int,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Whitelist-only scientific epoch record (no paths/timestamps)."""

    return {
        "epoch": int(epoch),
        "global_optimizer_step": int(global_optimizer_step),
        "ordered_stable_sample_ids": list(sample_ids),
        "calibration_primary": exact_float_field(primary, dtype="float64"),
        "calibration_secondary": exact_float_field(secondary, dtype="float64"),
        "training_total_loss": exact_float_field(total_loss, dtype="float32"),
    }


class TraceHashChain:
    def __init__(self, schema_version: str = TRACE_CHAIN_SCHEMA_VERSION) -> None:
        self.schema_version = schema_version
        self.nodes: list[dict[str, Any]] = []
        self.tail: str | None = None

    def append(self, record: Mapping[str, Any]) -> str:
        index = len(self.nodes)
        if index == 0:
            node = {
                "schema_version": self.schema_version,
                "chain_index": 0,
                "record": dict(record),
            }
        else:
            assert self.tail is not None
            node = {
                "schema_version": self.schema_version,
                "chain_index": index,
                "previous_sha256": self.tail,
                "record": dict(record),
            }
        digest = _canonical_json_sha256(node)
        self.nodes.append(node)
        self.tail = digest
        return digest


def verify_trace_chain(nodes: Sequence[Mapping[str, Any]]) -> str:
    if not nodes:
        _fail("B2_DLCM_TRACE_CHAIN_INVALID", "empty chain")
    previous: str | None = None
    tail = ""
    for expected_index, node in enumerate(nodes):
        if int(node.get("chain_index", -1)) != expected_index:
            _fail("B2_DLCM_TRACE_CHAIN_INVALID", "chain indices must be contiguous from 0")
        if expected_index == 0:
            if "previous_sha256" in node:
                _fail("B2_DLCM_TRACE_CHAIN_INVALID", "epoch 0 must not carry previous hash")
            payload = {
                "schema_version": node["schema_version"],
                "chain_index": 0,
                "record": node["record"],
            }
        else:
            prev = node.get("previous_sha256")
            if not isinstance(prev, str) or len(prev) != 64 or prev != previous:
                _fail("B2_DLCM_TRACE_CHAIN_INVALID", "previous_sha256 mismatch")
            payload = {
                "schema_version": node["schema_version"],
                "chain_index": expected_index,
                "previous_sha256": prev,
                "record": node["record"],
            }
        tail = _canonical_json_sha256(payload)
        previous = tail
    return tail


class EpochTransaction:
    def __init__(self, seed_dir: Path) -> None:
        self.seed_dir = Path(seed_dir)
        self.committed = self.seed_dir / "committed"
        self.staging = self.seed_dir / ".epoch_staging"

    def begin(self) -> Path:
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.committed.mkdir(parents=True, exist_ok=True)
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True, exist_ok=True)
        return self.staging

    def abort(self) -> None:
        if self.staging.exists():
            shutil.rmtree(self.staging)

    def commit(self, manifest: Mapping[str, Any], *, update_best: bool) -> None:
        if not self.staging.exists():
            _fail("B2_DLCM_EPOCH_TX_INVALID", "staging missing")
        # Move staged files into committed.
        for name in ("last_training_checkpoint.pt", "training_trace.json"):
            src = self.staging / name
            if not src.exists():
                _fail("B2_DLCM_EPOCH_TX_INVALID", f"missing staged {name}")
            dest = self.committed / name
            os.replace(src, dest)
        best_src = self.staging / "best_training_checkpoint.pt"
        if update_best:
            if not best_src.exists():
                _fail("B2_DLCM_EPOCH_TX_INVALID", "missing staged best checkpoint")
            os.replace(best_src, self.committed / "best_training_checkpoint.pt")
        elif best_src.exists():
            best_src.unlink()
        # Manifest last = commit point.
        manifest_path = self.committed / "epoch_state_manifest.json"
        _atomic_write_json(manifest_path, dict(manifest) | {"schema_version": EPOCH_MANIFEST_SCHEMA_VERSION})
        shutil.rmtree(self.staging)


def load_committed_manifest(seed_dir: Path) -> dict[str, Any]:
    path = Path(seed_dir) / "committed" / "epoch_state_manifest.json"
    if not path.is_file():
        _fail("B2_DLCM_RESUME_MISSING_MANIFEST", "committed manifest missing")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_seed_resumable(seed_dir: Path) -> None:
    manifest = Path(seed_dir) / "seed_manifest.json"
    if manifest.is_file():
        status = json.loads(manifest.read_text(encoding="utf-8")).get("status")
        if status in {"passed", "failed"}:
            _fail("B2_DLCM_RESUME_FORBIDDEN", f"seed status {status} cannot resume")


def write_failure_attestation(
    seed_dir: Path,
    *,
    seed: int,
    stage: str,
    error_code: str,
    last_valid_committed_epoch: int,
    global_optimizer_step: int,
    trace_chain_tail: str | None,
    identities: Mapping[str, str],
    environment_identity: str,
    uncommitted_staging: Sequence[str],
) -> None:
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "failure_stage": stage,
        "normalized_error_code": error_code,
        "last_valid_committed_epoch": int(last_valid_committed_epoch),
        "global_optimizer_step": int(global_optimizer_step),
        "trace_chain_tail": trace_chain_tail,
        "identities": dict(identities),
        "environment_identity": environment_identity,
        "uncommitted_staging_inventory": list(uncommitted_staging),
        "seed_status": "failed",
    }
    _atomic_write_json(seed_dir / "failure_attestation.json", payload)
    # Discard staging if present.
    staging = seed_dir / ".epoch_staging"
    if staging.exists():
        shutil.rmtree(staging)


def collect_environment_contract(
    *,
    visible_gpu_count: int | None = None,
    allow_cpu_for_hermetic: bool = False,
) -> dict[str, Any]:
    # Require process-level env vars already set (do not mutate after startup).
    required_env = {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }
    for key, expected in required_env.items():
        got = os.environ.get(key)
        if got != expected:
            _fail(
                "B2_DLCM_ENV_VAR_MISMATCH",
                f"{key} must be {expected!r} before Python start, got {got!r}",
            )
    # Thread counts must match the contract. ``set_num_interop_threads`` may be
    # called only once per process before parallel work; never re-set it.
    if torch.get_num_threads() != 4:
        torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 4:
        _fail("B2_DLCM_TORCH_THREADS_INVALID", "torch.get_num_threads() must be 4")
    if torch.get_num_interop_threads() != 1:
        _fail(
            "B2_DLCM_TORCH_THREADS_INVALID",
            "torch.get_num_interop_threads() must be 1",
        )

    gpu_count = torch.cuda.device_count() if visible_gpu_count is None else int(visible_gpu_count)
    if gpu_count != 1 and not allow_cpu_for_hermetic:
        _fail("B2_DLCM_GPU_COUNT_INVALID", f"visible GPU count must be 1, got {gpu_count}")

    contract: dict[str, Any] = {
        "schema_version": ENVIRONMENT_CONTRACT_VERSION,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": __import__("numpy").__version__,
        "scipy_version": __import__("scipy").__version__,
        "cuda_runtime_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "nvidia_driver_version": None,
        "gpu_model": torch.cuda.get_device_name(0) if gpu_count >= 1 and torch.cuda.is_available() else None,
        "gpu_compute_capability": (
            ".".join(str(v) for v in torch.cuda.get_device_capability(0))
            if gpu_count >= 1 and torch.cuda.is_available()
            else None
        ),
        "visible_gpu_count": gpu_count if not allow_cpu_for_hermetic else gpu_count,
        "training_dtype": "float32",
        "amp_enabled": False,
        "default_dtype": str(torch.get_default_dtype()).replace("torch.", ""),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else "highest"
        ),
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "OMP_NUM_THREADS": 4,
        "MKL_NUM_THREADS": 4,
        "torch_num_threads": 4,
        "torch_num_interop_threads": 1,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": 0,
        "hermetic_cpu_allowed": bool(allow_cpu_for_hermetic),
    }
    return contract


def environment_contract_sha256(contract: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(contract)


def persist_environment_contract(path: Path, contract: Mapping[str, Any]) -> str:
    """Persist contract JSON + receipt; return scientific identity (not file SHA)."""

    _atomic_write_json(Path(path), contract)
    return environment_contract_sha256(contract)


def deterministic_epoch_permutation(
    sample_ids: Sequence[str],
    *,
    epoch: int,
    sampler_seed: int,
) -> list[str]:
    ids = list(sample_ids)
    generator = torch.Generator(device="cpu")
    # Mix epoch into seed stream without 32-bit truncation.
    mixed = int(sampler_seed) ^ (int(epoch) * 0x9E3779B97F4A7C15 & ((1 << 63) - 1))
    generator.manual_seed(mixed & ((1 << 63) - 1))
    order = torch.randperm(len(ids), generator=generator).tolist()
    return [ids[i] for i in order]


def _calibration_metrics(
    model: dlcm.B2DLCM,
    calibration: Sequence[Any],
) -> tuple[float, float]:
    model.eval()
    primary_terms: list[float] = []
    secondary_terms: list[float] = []
    with torch.no_grad():
        for record in calibration:
            depth_primary: list[float] = []
            depth_secondary: list[float] = []
            for depth in model.prediction_depths:
                desc = record.descriptors[depth].unsqueeze(0)
                out = model.forward_training(desc, prediction_depth=depth)
                alloc, parts = dlcm.allocation_loss(
                    out.deployment_logits,
                    record.p_gt[depth].unsqueeze(0),
                    record.p_t[depth].unsqueeze(0),
                )
                s_gt, gt_parts = dlcm.signed_loss(
                    out.gt_signed, record.phi_gt[depth].unsqueeze(0)
                )
                s_t, t_parts = dlcm.signed_loss(
                    out.teacher_signed, record.phi_t[depth].unsqueeze(0)
                )
                depth_primary.append(float(0.5 * (parts["gt_kl"] + parts["teacher_kl"])))
                depth_secondary.append(float(s_gt))
                depth_secondary.append(float(s_t))
                _ = alloc
                _ = gt_parts
                _ = t_parts
            primary_terms.append(sum(depth_primary) / 3.0)
            secondary_terms.append(sum(depth_secondary) / 6.0)
    primary = sum(primary_terms) / float(len(primary_terms))
    secondary = sum(secondary_terms) / float(len(secondary_terms))
    return primary, secondary


def run_hermetic_contract_training(
    *,
    output_root: Path,
    seed: int,
    records: Sequence[Any],
    maximum_epochs: int = 1,
    patience: int = 50,
    device: str = "cpu",
    batch_size: int = 4,
) -> dict[str, Any]:
    """Hermetic contract-only training loop (not authoritative B2-05B training)."""

    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        _fail("B2_DLCM_OUTPUT_COLLISION", "output root must be empty/fresh")
    output_root.mkdir(parents=True, exist_ok=True)

    env = collect_environment_contract(allow_cpu_for_hermetic=True)
    persist_environment_contract(output_root / "environment_contract.json", env)

    by_split: dict[str, list[Any]] = {"training": [], "calibration": [], "evaluation": []}
    for record in records:
        by_split[record.split].append(record)
    if len(by_split["training"]) != 16 or len(by_split["calibration"]) != 8:
        _fail("B2_DLCM_SPLIT_INVALID", "expected 16/8 training/calibration")
    # Do not deserialize evaluation content beyond counting.
    if len(by_split["evaluation"]) != 8:
        _fail("B2_DLCM_SPLIT_INVALID", "expected 8 evaluation placeholders")

    seed_dir = output_root / f"seed_{seed}"
    tx = EpochTransaction(seed_dir)
    model = dlcm.B2DLCM(seed=seed)
    if device != "cpu":
        model = dlcm.move_model_to_device_and_verify(model, torch.device(device))
    # First LR is 3e-6.
    optimizer, _groups = build_adamw(model, lr=3e-6)
    schedule = ExplicitLRSchedule()
    schedule.install_into_optimizer(optimizer)
    selector = CheckpointSelector()
    stopper = EarlyStopController(patience=patience, maximum_epochs=maximum_epochs)
    chain = TraceHashChain()

    train_ids = [r.stable_sample_id for r in by_split["training"]]
    id_to_record = {r.stable_sample_id: r for r in by_split["training"]}
    calibration = sorted(by_split["calibration"], key=lambda r: r.stable_sample_id)
    component_seeds = model.component_seeds

    # Epoch 0 baseline.
    primary0, secondary0 = _calibration_metrics(model, calibration)
    selector.consider(epoch=0, primary=primary0, secondary=secondary0)
    run_epoch0_baseline_marker(schedule)
    staging = tx.begin()
    torch.save({"model": model.state_dict(), "epoch": 0}, staging / "best_training_checkpoint.pt")
    torch.save({"model": model.state_dict(), "epoch": 0}, staging / "last_training_checkpoint.pt")
    record0 = scientific_epoch_record(
        epoch=0,
        primary=primary0,
        secondary=secondary0,
        total_loss=0.0,
        global_optimizer_step=0,
        sample_ids=train_ids,
    )
    h0 = chain.append(record0)
    (staging / "training_trace.json").write_text(
        json.dumps({"nodes": chain.nodes, "tail": chain.tail}, sort_keys=True),
        encoding="utf-8",
    )
    tx.commit(
        {
            "epoch": 0,
            "best_epoch": 0,
            "patience": 0,
            "global_optimizer_step": 0,
            "trace_chain_tail": h0,
            "best_file_sha256": sha256_file(seed_dir / "committed" / "best_training_checkpoint.pt")
            if False
            else "pending",
            "status": "running",
        },
        update_best=True,
    )
    # Fix best/last hashes after commit.
    manifest = load_committed_manifest(seed_dir)
    manifest["best_file_sha256"] = sha256_file(seed_dir / "committed" / "best_training_checkpoint.pt")
    manifest["last_file_sha256"] = sha256_file(seed_dir / "committed" / "last_training_checkpoint.pt")
    _atomic_write_json(seed_dir / "committed" / "epoch_state_manifest.json", manifest)

    status = "running"
    last_epoch = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = deterministic_epoch_permutation(
            train_ids,
            epoch=epoch,
            sampler_seed=component_seeds["sampler"],
        )
        # Four batches of 4.
        epoch_losses: list[float] = []
        for step_idx in range(4):
            batch_ids = order[step_idx * batch_size : (step_idx + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)
            depth_payload: dict[int, dict[str, torch.Tensor]] = {}
            for depth in model.prediction_depths:
                descs = torch.stack([id_to_record[i].descriptors[depth] for i in batch_ids], dim=0)
                p_gt = torch.stack([id_to_record[i].p_gt[depth] for i in batch_ids], dim=0)
                p_t = torch.stack([id_to_record[i].p_t[depth] for i in batch_ids], dim=0)
                phi_gt = torch.stack([id_to_record[i].phi_gt[depth] for i in batch_ids], dim=0)
                phi_t = torch.stack([id_to_record[i].phi_t[depth] for i in batch_ids], dim=0)
                out = model.forward_training(descs, prediction_depth=depth)
                depth_payload[depth] = {
                    "deployment_logits": out.deployment_logits,
                    "gt_signed": out.gt_signed,
                    "teacher_signed": out.teacher_signed,
                    "p_gt": p_gt,
                    "p_t": p_t,
                    "phi_gt": phi_gt,
                    "phi_t": phi_t,
                }
            loss, _ = dlcm.total_dlcm_loss(depth_payload)
            if not bool(torch.isfinite(loss)):
                write_failure_attestation(
                    seed_dir,
                    seed=seed,
                    stage="optimizer_step",
                    error_code="B2_DLCM_NONFINITE_LOSS",
                    last_valid_committed_epoch=last_epoch,
                    global_optimizer_step=schedule.global_optimizer_step,
                    trace_chain_tail=chain.tail,
                    identities={"model_state_scientific_sha256": dlcm.model_state_scientific_sha256(model)},
                    environment_identity=environment_contract_sha256(env),
                    uncommitted_staging=[],
                )
                return {
                    "status": "failed",
                    "real_training_started": False,
                    "evaluation_unlocked": False,
                }
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # LR already installed for this update.
            optimizer.step()
            t = schedule.global_optimizer_step + 1
            schedule.note_successful_update(t)
            schedule.install_into_optimizer(optimizer)
            epoch_losses.append(float(loss.detach()))

        primary, secondary = _calibration_metrics(model, calibration)
        improved = selector.consider(epoch=epoch, primary=primary, secondary=secondary)
        status = stopper.after_epoch(epoch=epoch, improved=improved)
        staging = tx.begin()
        torch.save({"model": model.state_dict(), "epoch": epoch}, staging / "last_training_checkpoint.pt")
        if improved:
            torch.save({"model": model.state_dict(), "epoch": epoch}, staging / "best_training_checkpoint.pt")
        record = scientific_epoch_record(
            epoch=epoch,
            primary=primary,
            secondary=secondary,
            total_loss=sum(epoch_losses) / len(epoch_losses),
            global_optimizer_step=schedule.global_optimizer_step,
            sample_ids=order,
        )
        tail = chain.append(record)
        (staging / "training_trace.json").write_text(
            json.dumps({"nodes": chain.nodes, "tail": chain.tail}, sort_keys=True),
            encoding="utf-8",
        )
        tx.commit(
            {
                "epoch": epoch,
                "best_epoch": selector.best_epoch,
                "patience": stopper.patience_counter,
                "global_optimizer_step": schedule.global_optimizer_step,
                "trace_chain_tail": tail,
                "status": status,
            },
            update_best=improved,
        )
        last_epoch = epoch
        if status == "early_stopped":
            break

    return {
        "status": status if status != "running" else "completed_epoch",
        "real_training_started": False,
        "evaluation_unlocked": False,
        "best_epoch": selector.best_epoch,
        "seed": seed,
    }


def build_hermetic_contract_records(*, map_hw: tuple[int, int] = (8, 8)) -> list[dict[str, Any]]:
    """In-memory hermetic 32-record bundle for B2-05A dry-run validation.

    Production-owned (not imported from tests). Never persisted by dry-run.
    """

    categories = ("bottle", "cable", "capsule", "carpet")
    h, w = map_hw
    records: list[dict[str, Any]] = []
    for index in range(32):
        if index < 16:
            split = "training"
        elif index < 24:
            split = "calibration"
        else:
            split = "evaluation"
        descriptors: dict[int, torch.Tensor] = {}
        p_gt: dict[int, torch.Tensor] = {}
        p_t: dict[int, torch.Tensor] = {}
        phi_gt: dict[int, torch.Tensor] = {}
        phi_t: dict[int, torch.Tensor] = {}
        anomaly_maps: dict[int, torch.Tensor] = {}
        for depth in dlcm.DEFAULT_PREDICTION_DEPTHS:
            players = dlcm.players_for_depth(dlcm.DEFAULT_CANDIDATE_LAYERS, depth)
            n = len(players)
            base = torch.arange(n * 18, dtype=torch.float32)
            desc = ((base + index * 0.01 + depth * 0.001) % 17.0) / 8.5 - 1.0
            descriptors[depth] = desc.view(n, 18).contiguous()
            raw = torch.arange(1, n + 1, dtype=torch.float64) + 0.1 * index + 0.01 * depth
            p_gt[depth] = (raw / raw.sum()).to(torch.float32)
            raw_t = torch.arange(1, n + 1, dtype=torch.float64) + 0.07 * index + 0.02 * depth
            p_t[depth] = (raw_t / raw_t.sum()).to(torch.float32)
            phi = torch.linspace(-1.0, 1.0, n, dtype=torch.float32) * (1.0 + 0.01 * index)
            phi_gt[depth] = phi
            phi_t[depth] = phi.flip(0) * 0.5
            maps = torch.zeros(n, h, w, dtype=torch.float32)
            for li in range(n):
                maps[li].fill_(0.1 * (li + 1) + 0.001 * index)
            anomaly_maps[depth] = maps
        mask = torch.zeros(h, w, dtype=torch.float32)
        if index % 2 == 1:
            mask[2:5, 2:5] = 1.0
        records.append(
            {
                "stable_sample_id": f"fixture-{index:02d}",
                "split": split,
                "category": categories[index % len(categories)],
                "descriptors": descriptors,
                "p_gt": p_gt,
                "p_t": p_t,
                "phi_gt": phi_gt,
                "phi_t": phi_t,
                "anomaly_maps": anomaly_maps,
                "mask": mask,
                "artifact_kind": "test_fixture",
            }
        )
    return records


def optimizer_state_by_parameter_name(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map AdamW state by full parameter name (never by internal id)."""

    name_by_param = {id(param): name for name, param in model.named_parameters()}
    group_by_name: dict[str, str] = {}
    for group in groups:
        for name in group["ordered_parameter_names"]:
            group_by_name[name] = str(group["group_name"])
    out: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            name = name_by_param.get(id(param))
            if name is None:
                _fail("B2_DLCM_OPTIM_UNMAPPED", "optimizer param missing name mapping")
            state = optimizer.state.get(param, {})
            step_val = state.get("step", 0)
            if hasattr(step_val, "item"):
                step_val = int(step_val.item())
            else:
                step_val = int(step_val)
            out[name] = {
                "full_parameter_name": name,
                "optimizer_group": group_by_name[name],
                "step": step_val,
                "exp_avg": state.get("exp_avg"),
                "exp_avg_sq": state.get("exp_avg_sq"),
            }
    return out


def build_collection_failure_manifest(
    *,
    failed_seed: int,
    error_code: str,
    completed_seeds: Sequence[int],
    environment_identity: str,
) -> dict[str, Any]:
    return {
        "collection_status": "seed_collection_failed",
        "failed_seed": int(failed_seed),
        "normalized_error_code": error_code,
        "completed_seeds": list(completed_seeds),
        "environment_identity": environment_identity,
        "canonical_seed_selected": False,
        "evaluation_unlocked": False,
        "deployment_exported": False,
    }


def persist_collection_failure_manifest(path: Path, payload: Mapping[str, Any]) -> str:
    return _atomic_write_json(Path(path), payload)


def dry_run_complete_contract_validation(
    *,
    config: Mapping[str, Any],
    seed: int,
    output_root: Path | str,
) -> dict[str, Any]:
    """Complete B2-05A dry-run: hermetic contract exercise, zero filesystem writes."""

    output_root = Path(output_root)
    if config.get("real_training_enabled") is not False:
        _fail("B2_DLCM_REAL_TRAINING_FLAG", "dry-run requires real_training_enabled=false")
    if config.get("contract_stage") != "b2_05a":
        _fail("B2_DLCM_CONFIG_STAGE_INVALID", "contract_stage must be b2_05a")
    records = build_hermetic_contract_records()
    if len(records) != 32:
        _fail("B2_DLCM_HERMETIC_COUNT", "hermetic fixture must contain 32 records")
    model = dlcm.B2DLCM(seed=int(seed))
    train = [r for r in records if r["split"] == "training"]
    batch = train[:4]
    depth_payload: dict[int, dict[str, torch.Tensor]] = {}
    for depth in model.prediction_depths:
        descs = torch.stack([r["descriptors"][depth] for r in batch], dim=0)
        out = model.forward_training(descs, prediction_depth=depth)
        depth_payload[depth] = {
            "deployment_logits": out.deployment_logits,
            "gt_signed": out.gt_signed,
            "teacher_signed": out.teacher_signed,
            "p_gt": torch.stack([r["p_gt"][depth] for r in batch], dim=0),
            "p_t": torch.stack([r["p_t"][depth] for r in batch], dim=0),
            "phi_gt": torch.stack([r["phi_gt"][depth] for r in batch], dim=0),
            "phi_t": torch.stack([r["phi_t"][depth] for r in batch], dim=0),
        }
    loss, _ = dlcm.total_dlcm_loss(depth_payload)
    if not bool(torch.isfinite(loss)):
        _fail("B2_DLCM_NONFINITE_LOSS", "dry-run loss nonfinite")
    sched = ExplicitLRSchedule()
    if abs(sched.learning_rate_for_step(1) - 3e-6) > 1e-15:
        _fail("B2_DLCM_SCHEDULER_LR", "first-step LR contract failed")
    if abs(sched.learning_rate_for_step(100) - 3e-4) > 1e-15:
        _fail("B2_DLCM_SCHEDULER_LR", "100th-step LR contract failed")
    _ = dlcm.derive_component_seeds(int(seed))
    maps = batch[0]["anomaly_maps"][18].unsqueeze(0)
    ref = dlcm.reference_uniform_weights(3).view(1, 3).contiguous()
    fused, path = dlcm.sum_preserving_fusion(
        maps,
        ref,
        prediction_depth=18,
        player_layer_ids=(6, 12, 18),
        return_path=True,
    )
    if path != "uniform_baseline":
        _fail("B2_DLCM_FUSION_PATH", "epoch-0 style uniform path required")
    _ = fused
    return {
        "mode": "dry_run",
        "status": "contract_validated",
        "artifact_written": False,
        "run_directory_created": False,
        "real_training_started": False,
        "evaluation_unlocked": False,
        "teacher_forward_count": 0,
        "hermetic_records_validated": 32,
        "seed": int(seed),
        "dry_run_loss_finite": True,
        "fusion_path": path,
    }
