"""B2-05A DLCM deployment loading, qualification, and accepted-manifest contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import torch
import torch.nn as nn

from rad.phase_b import b2_dlcm as dlcm
from rad.phase_b.b2_dlcm import (
    ARCHITECTURE_CONTRACT_VERSION,
    DEFAULT_CANDIDATE_LAYERS,
    DEFAULT_PREDICTION_DEPTHS,
    MODEL_CLASS_ID,
    B2DLCMDeploymentTrunk,
)

LOADER_CONTRACT_VERSION = "b2_dlcm_loader_v1"
DEPLOYMENT_SCHEMA_VERSION = "b2_dlcm_deployment_checkpoint_v1"
FORMAL_LOCALIZATION_ADAPTER_ID = "b2_dlcm_formal_localization_v1"


class B2DLCMDeploymentError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMDeploymentError(code, detail)


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


def build_golden_input(case_id: str, prediction_depth: int, player_count: int) -> torch.Tensor:
    """Authoritative golden synthetic inputs (float32)."""

    n = int(player_count)
    feat = 18
    if case_id == "all_zero":
        return torch.zeros(1, n, feat, dtype=torch.float32)
    if case_id == "monotonic_ramp":
        total = n * feat
        flat = -1.0 + 2.0 * torch.arange(total, dtype=torch.float32) / float(total - 1)
        return flat.view(1, n, feat).contiguous()
    if case_id == "alternating_signed":
        out = torch.empty(1, n, feat, dtype=torch.float32)
        for layer in range(n):
            for k in range(feat):
                sign = -1.0 if ((layer + k) % 2) else 1.0
                out[0, layer, k] = sign * (0.25 + 0.05 * k + 0.1 * layer)
        return out.contiguous()
    _fail("B2_DLCM_GOLDEN_CASE_INVALID", f"unknown case {case_id}")


def generate_golden_cases(
    model: nn.Module,
    *,
    candidate_layers: Sequence[int] = DEFAULT_CANDIDATE_LAYERS,
    prediction_depths: Sequence[int] = DEFAULT_PREDICTION_DEPTHS,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for depth in prediction_depths:
            players = dlcm.players_for_depth(candidate_layers, depth)
            n = len(players)
            for case_id in ("all_zero", "monotonic_ramp", "alternating_signed"):
                inp = build_golden_input(case_id, depth, n)
                forward_fn = getattr(model, "forward_deployment", None)
                if callable(forward_fn):
                    logits, weights = cast(
                        tuple[torch.Tensor, torch.Tensor],
                        forward_fn(inp, prediction_depth=depth, player_layer_ids=players),
                    )
                else:
                    logits, weights = cast(
                        tuple[torch.Tensor, torch.Tensor],
                        model(inp, prediction_depth=depth, player_layer_ids=players),
                    )
                cases.append(
                    {
                        "case_id": case_id,
                        "prediction_depth": int(depth),
                        "player_count": n,
                        "input_dtype": "float32",
                        "input_shape": [1, n, 18],
                        "input_tensor": inp.cpu(),
                        "input_tensor_sha256": dlcm.tensor_sha256(inp),
                        "expected_logits_bits": [
                            dlcm.float_to_bits_hex(float(v), dtype="float32")
                            for v in logits.view(-1).tolist()
                        ],
                        "expected_weights_bits": [
                            dlcm.float_to_bits_hex(float(v), dtype="float32")
                            for v in weights.view(-1).tolist()
                        ],
                    }
                )
    return cases


def embed_normalization(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Embed accepted B2-03B normalization (axes format or flat fixture format)."""

    if "axes" in stats:
        required_axes = (
            "normalization_contract_version",
            "normalization_statistics_scientific_sha256",
            "normalization_training_coverage_sha256",
            "axes",
        )
        missing = [k for k in required_axes if k not in stats]
        if missing:
            _fail("B2_DLCM_NORM_INCOMPLETE", f"missing normalization fields {missing}")
        # Flatten per-depth mean/std matrices [n_d, 18] for deployment apply.
        feature_order: list[str] | None = None
        by_depth: dict[str, dict[str, Any]] = {}
        for depth_key, depth_block in dict(stats["axes"]).items():
            layers = list(depth_block["layers"])
            means: list[list[float]] = []
            stds: list[list[float]] = []
            counts: list[list[int]] = []
            zero_var: list[list[bool]] = []
            for layer in layers:
                feats = list(layer["features"])
                names = [str(f["descriptor_feature_name"]) for f in feats]
                if feature_order is None:
                    feature_order = names
                elif names != feature_order:
                    _fail("B2_DLCM_NORM_FEATURE_ORDER", "feature order drifted across axes")
                means.append([float(f["mean"]) for f in feats])
                stds.append([float(f["std"]) for f in feats])
                counts.append([int(f["count"]) for f in feats])
                zero_var.append([bool(float(f["std"]) == 0.0) for f in feats])
            by_depth[str(int(depth_key))] = {
                "mean": means,
                "std": stds,
                "count": counts,
                "zero_variance": zero_var,
                "player_layer_ids": [int(layer["candidate_layer_id"]) for layer in layers],
            }
        return {
            "format": "b2_03b_axes_v1",
            "normalization_contract_version": stats["normalization_contract_version"],
            "descriptor_normalization_scientific_sha256": stats[
                "normalization_statistics_scientific_sha256"
            ],
            "descriptor_normalization_training_coverage_sha256": stats[
                "normalization_training_coverage_sha256"
            ],
            "feature_order": list(feature_order or []),
            "axis_order": ["prediction_depth", "candidate_layer", "feature"],
            "by_depth": by_depth,
            "mean": by_depth["24"]["mean"],  # default depth-24 matrix for legacy checks
            "std": by_depth["24"]["std"],
            "count": by_depth["24"]["count"],
            "zero_variance": by_depth["24"]["zero_variance"],
        }

    required = (
        "mean",
        "std",
        "count",
        "zero_variance",
        "feature_order",
        "axis_order",
        "normalization_contract_version",
        "descriptor_normalization_scientific_sha256",
        "descriptor_normalization_training_coverage_sha256",
    )
    missing = [k for k in required if k not in stats]
    if missing:
        _fail("B2_DLCM_NORM_INCOMPLETE", f"missing normalization fields {missing}")
    return {key: stats[key] for key in required}


def apply_embedded_normalization(
    raw: torch.Tensor,
    embedded: Mapping[str, Any],
    *,
    prediction_depth: int | None = None,
) -> torch.Tensor:
    """Raw CPU float32 → float64 standardize → float32."""

    if raw.device.type != "cpu":
        _fail("B2_DLCM_NORM_DEVICE", "raw descriptors must be CPU")
    if raw.dtype != torch.float32:
        _fail("B2_DLCM_NORM_DTYPE", "raw descriptors must be float32")
    if not raw.is_contiguous():
        _fail("B2_DLCM_NORM_CONTIGUITY", "raw descriptors must be contiguous")
    if raw.ndim != 3 or raw.shape[-1] != 18:
        _fail("B2_DLCM_NORM_SHAPE", "expected [B,n,18]")
    if not bool(torch.isfinite(raw).all()):
        _fail("B2_DLCM_NORM_NONFINITE", "raw descriptors nonfinite")

    if embedded.get("format") == "b2_03b_axes_v1":
        if prediction_depth is None:
            _fail("B2_DLCM_NORM_DEPTH_REQUIRED", "axes normalization requires prediction_depth")
        depth_block = embedded["by_depth"].get(str(int(prediction_depth)))
        if depth_block is None:
            _fail("B2_DLCM_NORM_DEPTH_MISSING", f"no stats for depth {prediction_depth}")
        mean = torch.tensor(depth_block["mean"], dtype=torch.float64)
        std = torch.tensor(depth_block["std"], dtype=torch.float64)
        zero_var = [bool(flag) for row in depth_block["zero_variance"] for flag in row]
        # Rebuild zero-var mask as [n,18]
        zero_mask = torch.tensor(depth_block["zero_variance"], dtype=torch.bool)
    else:
        mean = torch.tensor(embedded["mean"], dtype=torch.float64)
        std = torch.tensor(embedded["std"], dtype=torch.float64)
        zero_mask = None
        zero_var = list(embedded["zero_variance"])

    x64 = raw.to(torch.float64)
    # Broadcast mean/std over batch.
    while mean.ndim < x64.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    out = (x64 - mean) / std
    if zero_mask is not None:
        while zero_mask.ndim < out.ndim:
            zero_mask = zero_mask.unsqueeze(0)
        out = torch.where(zero_mask, torch.zeros_like(out), out)
    else:
        # Legacy flat zero_variance list applied on last dims if shapes match.
        _ = zero_var
    if zero_mask is None and zero_var:
        # Legacy flat zero_variance: treat as last-dim flags when length matches F.
        if len(zero_var) == int(out.shape[-1]):
            for axis, is_zero in enumerate(zero_var):
                if is_zero:
                    out[..., axis] = 0.0
        elif len(zero_var) == int(out.shape[-2] * out.shape[-1]):
            mask = torch.tensor(zero_var, dtype=torch.bool).view(out.shape[-2], out.shape[-1])
            while mask.ndim < out.ndim:
                mask = mask.unsqueeze(0)
            out = torch.where(mask, torch.zeros_like(out), out)
    if not bool(torch.isfinite(out).all()):
        _fail("B2_DLCM_NORM_NONFINITE", "normalized descriptors nonfinite")
    return out.to(dtype=torch.float32).contiguous()


def deployment_scientific_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    whitelist = (
        "schema_version",
        "architecture_contract_version",
        "model_class_id",
        "candidate_layers",
        "prediction_depths",
        "descriptor_feature_names",
        "layer_embedding_dimension",
        "depth_embedding_dimension",
        "hidden_dimension",
        "dropout_contract",
        "head_contract",
        "state_dict_digests",
        "embedded_normalization",
        "golden_cases_scientific",
        "canonical_seed",
        "source_original_best_training_identity",
        "source_reproduction_best_training_identity",
        "descriptor_normalization_scientific_sha256",
        "contribution_target_collection_scientific_sha256",
    )
    return {key: checkpoint[key] for key in whitelist if key in checkpoint}


def deployment_scientific_sha256(checkpoint: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(deployment_scientific_payload(checkpoint))


def export_deployment_checkpoint(
    *,
    training_model: dlcm.B2DLCM,
    normalization: Mapping[str, Any],
    canonical_seed: int,
    source_original_best_identity: str,
    source_reproduction_best_identity: str,
    contribution_target_collection_scientific_sha256: str,
) -> dict[str, Any]:
    if source_original_best_identity != source_reproduction_best_identity:
        _fail(
            "B2_DLCM_DEPLOY_SOURCE_MISMATCH",
            "original and reproduction best identities must match",
        )
    deploy_state = dlcm.extract_deployment_state_dict(training_model)
    full = training_model
    trunk = B2DLCMDeploymentTrunk(
        seed=None,
        initialize=False,
        candidate_layers=full.candidate_layers,
        prediction_depths=full.prediction_depths,
        descriptor_dimension=full.descriptor_dimension,
        layer_embedding_dimension=full.layer_embedding_dimension,
        depth_embedding_dimension=full.depth_embedding_dimension,
        hidden_dimension=full.hidden_dimension,
        dropout_probability=full.dropout_probability,
    )
    trunk.dropout_site_seeds = dict(full.dropout_site_seeds)
    trunk._reset_dropout_generators(device="cpu")
    trunk.load_state_dict(deploy_state, strict=True)
    trunk.eval()

    embedded = embed_normalization(normalization)
    golden = generate_golden_cases(trunk)
    golden_scientific = []
    for case in golden:
        golden_scientific.append(
            {
                "case_id": case["case_id"],
                "prediction_depth": case["prediction_depth"],
                "player_count": case["player_count"],
                "input_dtype": case["input_dtype"],
                "input_shape": case["input_shape"],
                "input_tensor_sha256": case["input_tensor_sha256"],
                "expected_logits_bits": case["expected_logits_bits"],
                "expected_weights_bits": case["expected_weights_bits"],
            }
        )
    state_digests = {
        name: dlcm.tensor_sha256(tensor) for name, tensor in sorted(deploy_state.items())
    }
    checkpoint: dict[str, Any] = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "architecture_contract_version": ARCHITECTURE_CONTRACT_VERSION,
        "model_class_id": MODEL_CLASS_ID,
        "candidate_layers": list(full.candidate_layers),
        "prediction_depths": list(full.prediction_depths),
        "descriptor_feature_names": [f"f{i:02d}" for i in range(18)],
        "layer_embedding_dimension": full.layer_embedding_dimension,
        "depth_embedding_dimension": full.depth_embedding_dimension,
        "hidden_dimension": full.hidden_dimension,
        "dropout_contract": {"p": full.dropout_probability, "inplace": False},
        "head_contract": {"deployment": "64->1", "auxiliary_deleted": True},
        "state_dict": deploy_state,
        "state_dict_digests": state_digests,
        "embedded_normalization": embedded,
        "golden_cases": golden,
        "golden_cases_scientific": golden_scientific,
        "canonical_seed": int(canonical_seed),
        "source_original_best_training_identity": source_original_best_identity,
        "source_reproduction_best_training_identity": source_reproduction_best_identity,
        "descriptor_normalization_scientific_sha256": embedded[
            "descriptor_normalization_scientific_sha256"
        ],
        "contribution_target_collection_scientific_sha256": (
            contribution_target_collection_scientific_sha256
        ),
        "input_dtype": "float32",
        "parameter_dtype": "float32",
        "output_dtype": "float32",
    }
    checkpoint["deployment_scientific_sha256"] = deployment_scientific_sha256(checkpoint)
    return checkpoint


def run_cpu_golden_self_test(checkpoint: Mapping[str, Any]) -> None:
    trunk = _instantiate_trunk_from_checkpoint(checkpoint)
    trunk.load_state_dict(checkpoint["state_dict"], strict=True)
    trunk.eval()
    for case in checkpoint["golden_cases"]:
        inp = case["input_tensor"]
        if not isinstance(inp, torch.Tensor):
            _fail("B2_DLCM_GOLDEN_INPUT_MISSING", "persisted tensor required")
        logits, weights = trunk.forward(
            inp,
            prediction_depth=case["prediction_depth"],
            player_layer_ids=tuple(
                dlcm.players_for_depth(checkpoint["candidate_layers"], case["prediction_depth"])
            ),
        )
        for value, meta in zip(logits.view(-1).tolist(), case["expected_logits_bits"], strict=True):
            got = dlcm.float_to_bits_hex(float(value), dtype="float32")
            if got["bits_hex"] != meta["bits_hex"]:
                _fail("B2_DLCM_GOLDEN_LOGIT_MISMATCH", "CPU golden logits bit mismatch")
        for value, meta in zip(weights.view(-1).tolist(), case["expected_weights_bits"], strict=True):
            got = dlcm.float_to_bits_hex(float(value), dtype="float32")
            if got["bits_hex"] != meta["bits_hex"]:
                _fail("B2_DLCM_GOLDEN_WEIGHT_MISMATCH", "CPU golden weights bit mismatch")
        if not bool(torch.isfinite(weights).all()):
            _fail("B2_DLCM_GOLDEN_NONFINITE", "weights nonfinite")
        if bool((weights < -1e-6).any()):
            _fail("B2_DLCM_GOLDEN_NEGATIVE", "weights negative")
        if not bool(torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[0]), atol=1e-6)):
            _fail("B2_DLCM_GOLDEN_SUM", "weight rows must sum to 1")


def _instantiate_trunk_from_checkpoint(checkpoint: Mapping[str, Any]) -> B2DLCMDeploymentTrunk:
    model = dlcm.B2DLCM(
        seed=None,
        candidate_layers=checkpoint["candidate_layers"],
        prediction_depths=checkpoint["prediction_depths"],
        initialize=False,
    )
    trunk = B2DLCMDeploymentTrunk.__new__(B2DLCMDeploymentTrunk)
    nn.Module.__init__(trunk)
    trunk.candidate_layers = tuple(checkpoint["candidate_layers"])
    trunk.prediction_depths = tuple(checkpoint["prediction_depths"])
    trunk.descriptor_dimension = 18
    trunk.layer_embedding_dimension = int(checkpoint["layer_embedding_dimension"])
    trunk.depth_embedding_dimension = int(checkpoint["depth_embedding_dimension"])
    trunk.hidden_dimension = int(checkpoint["hidden_dimension"])
    trunk.dropout_probability = float(checkpoint["dropout_contract"]["p"])
    trunk.layer_id_to_index = {layer: idx for idx, layer in enumerate(trunk.candidate_layers)}
    trunk.depth_id_to_index = {depth: idx for idx, depth in enumerate(trunk.prediction_depths)}
    trunk.layer_embedding = model.layer_embedding
    trunk.depth_embedding = model.depth_embedding
    trunk.layer_encoder = model.layer_encoder
    trunk.context_encoder = model.context_encoder
    trunk.deployment_head = model.deployment_head
    trunk.dropout_site_seeds = {site: 0 for site in dlcm.DROPOUT_SITE_NAMES}
    trunk.dropout_generators = {}
    trunk._reset_dropout_generators(device="cpu")
    return trunk


@dataclass
class QualificationCacheKey:
    accepted_deployment_scientific_sha256: str
    checkpoint_file_sha256: str
    environment_contract_sha256: str
    loader_contract_version: str
    gpu_device_index: int
    gpu_uuid: str
    gpu_model: str


@dataclass
class ImmutableDLCMInference:
    """Protected deployment wrapper — not a bare nn.Module for callers."""

    _model: B2DLCMDeploymentTrunk
    _device: torch.device
    _state_identity: str
    _checkpoint_sha: str
    _deployment_scientific_sha256: str
    _embedded_normalization: Mapping[str, Any]
    _candidate_layers: tuple[int, ...]
    _qualified: bool = True
    _cache_key: QualificationCacheKey | None = None

    def train(self, mode: bool = True) -> None:  # noqa: FBT001,FBT002
        if mode:
            _fail("B2_DLCM_IMMUTABLE_TRAIN", "immutable wrapper rejects train(True)")
        self._model.eval()

    def load_state_dict(self, *args: Any, **kwargs: Any) -> None:
        _fail("B2_DLCM_IMMUTABLE_LOAD", "immutable wrapper rejects load_state_dict")

    def to(self, *args: Any, **kwargs: Any) -> None:
        _fail("B2_DLCM_IMMUTABLE_DEVICE", "immutable wrapper rejects device moves")

    def _validate_alive(self) -> None:
        if not self._qualified:
            _fail("B2_DLCM_QUALIFICATION_INVALID", "qualification cache invalidated")
        live = dlcm.model_state_scientific_sha256(self._model)
        if live != self._state_identity:
            self._qualified = False
            _fail("B2_DLCM_STATE_MUTATION", "model state mutated after qualification")

    def forward(
        self,
        raw_descriptors: torch.Tensor,
        prediction_depth: int,
        player_layer_ids: Sequence[int],
    ) -> torch.Tensor:
        self._validate_alive()
        dlcm.validate_player_layer_ids(
            prediction_depth,
            player_layer_ids,
            candidate_layers=self._candidate_layers,
        )
        if raw_descriptors.device.type != "cpu":
            _fail("B2_DLCM_INPUT_DEVICE", "formal inputs must be CPU")
        if raw_descriptors.dtype != torch.float32 or not raw_descriptors.is_contiguous():
            _fail("B2_DLCM_INPUT_DTYPE", "formal inputs must be contiguous float32")
        if raw_descriptors.ndim != 3:
            _fail("B2_DLCM_INPUT_RANK", "formal inputs must be rank-3 [B,n,18]")
        if int(raw_descriptors.shape[0]) < 1:
            _fail("B2_DLCM_INPUT_BATCH", "batch size B must be >= 1")
        if not bool(torch.isfinite(raw_descriptors).all()):
            _fail("B2_DLCM_INPUT_NONFINITE", "formal inputs must be finite")
        normalized = apply_embedded_normalization(
            raw_descriptors,
            self._embedded_normalization,
            prediction_depth=prediction_depth,
        )
        normalized = normalized.to(self._device)
        with torch.inference_mode():
            _logits, weights = self._model.forward(
                normalized,
                prediction_depth=prediction_depth,
                player_layer_ids=player_layer_ids,
            )
        if weights.device != self._device or weights.dtype != torch.float32:
            _fail("B2_DLCM_OUTPUT_CONTRACT", "weights must be float32 on qualified GPU")
        if bool((weights < -1e-6).any()) or not bool(torch.isfinite(weights).all()):
            _fail("B2_DLCM_OUTPUT_INVALID", "invalid deployment weights")
        if not bool(torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[0], device=weights.device), atol=1e-6)):
            _fail("B2_DLCM_OUTPUT_SUM", "weight rows must sum to 1")
        return weights.contiguous()

    def forward_diagnostic(
        self,
        raw_descriptors: torch.Tensor,
        prediction_depth: int,
        player_layer_ids: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_alive()
        normalized = apply_embedded_normalization(
            raw_descriptors,
            self._embedded_normalization,
            prediction_depth=prediction_depth,
        )
        normalized = normalized.to(self._device)
        with torch.inference_mode():
            return self._model.forward(
                normalized,
                prediction_depth=prediction_depth,
                player_layer_ids=player_layer_ids,
            )


_PROCESS_QUAL_CACHE: dict[str, ImmutableDLCMInference] = {}


def _cache_key_str(key: QualificationCacheKey) -> str:
    return _canonical_json_sha256(key.__dict__)


def clear_qualification_cache() -> None:
    _PROCESS_QUAL_CACHE.clear()


def invalidate_qualification_cache_entry(wrapper: ImmutableDLCMInference) -> None:
    wrapper._qualified = False
    if wrapper._cache_key is not None:
        _PROCESS_QUAL_CACHE.pop(_cache_key_str(wrapper._cache_key), None)


def run_gpu_qualification(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
    gpu_atol: float = 1e-6,
) -> dict[str, Any]:
    """§40 GPU numerical qualification; errors recorded for runtime attestation."""

    if device.type != "cuda":
        _fail("B2_DLCM_GPU_QUAL_DEVICE", "GPU qualification requires CUDA device")
    run_cpu_golden_self_test(checkpoint)
    trunk = _instantiate_trunk_from_checkpoint(checkpoint)
    trunk.load_state_dict(checkpoint["state_dict"], strict=True)
    trunk.eval()
    for param in trunk.parameters():
        param.requires_grad_(False)
    trunk.to(device)
    trunk._reset_dropout_generators(device=device)
    max_logit = 0.0
    max_weight = 0.0
    for case in checkpoint["golden_cases"]:
        inp = case["input_tensor"].to(device, non_blocking=False)
        with torch.no_grad():
            logits, weights = trunk.forward(
                inp,
                prediction_depth=case["prediction_depth"],
                player_layer_ids=tuple(
                    dlcm.players_for_depth(
                        checkpoint["candidate_layers"], case["prediction_depth"]
                    )
                ),
            )
        if weights.dtype != torch.float32 or not bool(torch.isfinite(weights).all()):
            _fail("B2_DLCM_GPU_QUAL_FAIL", "GPU weights dtype/finite invalid")
        if bool((weights < -1e-6).any()):
            _fail("B2_DLCM_GPU_QUAL_FAIL", "GPU weights below -1e-6")
        if not bool(
            torch.allclose(
                weights.sum(dim=-1),
                torch.ones(weights.shape[0], device=weights.device),
                atol=1e-6,
            )
        ):
            _fail("B2_DLCM_GPU_QUAL_FAIL", "GPU weight row sums invalid")
        cpu_logits = torch.tensor(
            [dlcm.bits_hex_to_float(m) for m in case["expected_logits_bits"]],
            dtype=torch.float32,
        ).view_as(logits.cpu())
        cpu_weights = torch.tensor(
            [dlcm.bits_hex_to_float(m) for m in case["expected_weights_bits"]],
            dtype=torch.float32,
        ).view_as(weights.cpu())
        max_logit = max(max_logit, float((logits.cpu() - cpu_logits).abs().max()))
        max_weight = max(max_weight, float((weights.cpu() - cpu_weights).abs().max()))
    if max_logit > gpu_atol or max_weight > gpu_atol:
        _fail("B2_DLCM_GPU_QUAL_FAIL", "GPU vs CPU drift exceeds tolerance")
    return {
        "cases_run": 9,
        "passed": True,
        "max_logit_abs_error": max_logit,
        "max_weight_abs_error": max_weight,
        # Runtime attestation only — not scientific identity.
        "runtime_attestation_fields": {
            "gpu_device": str(device),
            "max_logit_abs_error": max_logit,
            "max_weight_abs_error": max_weight,
        },
    }


def load_qualified_deployment(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_file_sha256: str,
    environment_contract_sha256: str,
    device: torch.device,
    gpu_uuid: str = "hermetic",
    require_accepted_manifest: Mapping[str, Any] | None = None,
    gpu_atol: float = 1e-6,
) -> ImmutableDLCMInference:
    if require_accepted_manifest is not None:
        verify_accepted_manifest_for_loader(
            require_accepted_manifest,
            deployment_scientific_sha256=str(checkpoint["deployment_scientific_sha256"]),
            deployment_file_sha256=checkpoint_file_sha256,
            expected_accepted_identity=str(
                require_accepted_manifest.get("accepted_identity", "")
            ),
        )

    key = QualificationCacheKey(
        accepted_deployment_scientific_sha256=str(checkpoint["deployment_scientific_sha256"]),
        checkpoint_file_sha256=checkpoint_file_sha256,
        environment_contract_sha256=environment_contract_sha256,
        loader_contract_version=LOADER_CONTRACT_VERSION,
        gpu_device_index=(device.index or 0) if device.type == "cuda" else -1,
        gpu_uuid=gpu_uuid,
        gpu_model=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    )
    cache_id = _cache_key_str(key)
    cached = _PROCESS_QUAL_CACHE.get(cache_id)
    if cached is not None and cached._qualified:
        try:
            cached._validate_alive()
            return cached
        except B2DLCMDeploymentError:
            _PROCESS_QUAL_CACHE.pop(cache_id, None)

    # First accepted load in this process: manifest (optional) + checkpoint golden
    # CPU bit-exact + GPU numerical qualification when device is CUDA.
    run_cpu_golden_self_test(checkpoint)
    trunk = _instantiate_trunk_from_checkpoint(checkpoint)
    trunk.load_state_dict(checkpoint["state_dict"], strict=True)
    trunk.eval()
    for param in trunk.parameters():
        param.requires_grad_(False)

    if device.type == "cuda":
        run_gpu_qualification(checkpoint, device=device, gpu_atol=gpu_atol)
        trunk.to(device)
        trunk._reset_dropout_generators(device=device)
    elif device.type != "cpu":
        _fail("B2_DLCM_DEVICE_INVALID", f"unsupported device {device}")

    state_id = dlcm.model_state_scientific_sha256(trunk)
    wrapper = ImmutableDLCMInference(
        _model=trunk,
        _device=device,
        _state_identity=state_id,
        _checkpoint_sha=checkpoint_file_sha256,
        _deployment_scientific_sha256=str(checkpoint["deployment_scientific_sha256"]),
        _embedded_normalization=checkpoint["embedded_normalization"],
        _candidate_layers=tuple(checkpoint["candidate_layers"]),
        _cache_key=key,
    )
    _PROCESS_QUAL_CACHE[cache_id] = wrapper
    return wrapper


def select_canonical_seed(
    seed_summaries: Sequence[Mapping[str, Any]],
    *,
    min_delta: float = 1e-5,
) -> dict[str, Any]:
    """Deterministic canonical seed selection among verified best checkpoints."""

    if [s["seed"] for s in seed_summaries] != [17, 29, 43]:
        _fail("B2_DLCM_SEED_ORDER_INVALID", "seeds must be exactly [17,29,43]")
    # Special rule: all best remain epoch 0 → seed 17.
    if all(int(s["best_epoch"]) == 0 for s in seed_summaries):
        chosen = next(s for s in seed_summaries if int(s["seed"]) == 17)
        return {
            "canonical_seed": 17,
            "selected_best_epoch": 0,
            "selected_best_model_state_identity": chosen["best_model_state_identity"],
            "rule": "epoch0_cross_seed_seed17",
            "min_delta": min_delta,
        }
    ranked = sorted(
        seed_summaries,
        key=lambda s: (
            float(s["calibration_primary"]),
            float(s["calibration_secondary"]),
            int(s["seed"]),
        ),
    )
    # Apply min_delta dominance manually for clarity.
    best = ranked[0]
    for cand in ranked[1:]:
        p_diff = float(cand["calibration_primary"]) - float(best["calibration_primary"])
        if p_diff < -min_delta:
            best = cand
        elif abs(p_diff) <= min_delta:
            s_diff = float(cand["calibration_secondary"]) - float(best["calibration_secondary"])
            if s_diff < -min_delta or (
                abs(s_diff) <= min_delta and int(cand["seed"]) < int(best["seed"])
            ):
                best = cand
    return {
        "canonical_seed": int(best["seed"]),
        "selected_best_epoch": int(best["best_epoch"]),
        "selected_best_model_state_identity": best["best_model_state_identity"],
        "rule": "primary_secondary_seed",
        "min_delta": min_delta,
    }


def evaluate_qualification_gates(
    *,
    kl_dlcm_gt_macro: float,
    kl_uniform_gt_macro: float,
    kl_dlcm_teacher_macro: float,
    kl_uniform_teacher_macro: float,
    per_category_kl: Mapping[str, Mapping[str, float]],
    delta_pixel_ap_macro: float,
    delta_pixel_auroc_macro: float,
    delta_aupro_macro: float,
    per_category_localization: Mapping[str, Mapping[str, float]],
    best_epoch: int,
    localization_evidence: FormalLocalizationGateEvidence | None = None,
) -> dict[str, Any]:
    target_ok = True
    loc_ok = True
    reasons: list[str] = []
    if best_epoch == 0:
        target_ok = False
        reasons.append("epoch0_cannot_pass_target_learning")
    if not (
        kl_dlcm_gt_macro <= kl_uniform_gt_macro - 1e-5
        and kl_dlcm_teacher_macro <= kl_uniform_teacher_macro - 1e-5
    ):
        target_ok = False
        reasons.append("macro_kl_gate")
    for category, fam in per_category_kl.items():
        if fam["gt"] > fam["gt_uniform"] + 1e-4 or fam["teacher"] > fam["teacher_uniform"] + 1e-4:
            target_ok = False
            reasons.append(f"category_kl:{category}")
    if localization_evidence is None:
        loc_ok = False
        reasons.append("formal_localization_adapter_missing")
    else:
        if localization_evidence.metric_source_identity != FORMAL_LOCALIZATION_ADAPTER_ID:
            loc_ok = False
            reasons.append("formal_localization_adapter_identity_invalid")
        delta_pixel_ap_macro = localization_evidence.delta_pixel_ap_macro
        delta_pixel_auroc_macro = localization_evidence.delta_pixel_auroc_macro
        delta_aupro_macro = localization_evidence.delta_aupro_macro
        per_category_localization = localization_evidence.per_category_localization
    if not (
        delta_pixel_ap_macro >= 0
        and delta_pixel_auroc_macro >= -1e-4
        and delta_aupro_macro >= -1e-4
    ):
        loc_ok = False
        reasons.append("macro_localization_gate")
    for category, metrics in per_category_localization.items():
        if (
            metrics["delta_pixel_ap"] < -1e-3
            or metrics["delta_pixel_auroc"] < -1e-3
            or metrics["delta_aupro"] < -1e-3
        ):
            loc_ok = False
            reasons.append(f"category_localization:{category}")
    if target_ok and loc_ok:
        state = "deployment_qualified"
    elif target_ok and not loc_ok:
        state = "trained_but_not_deployment_qualified"
    elif loc_ok and not target_ok:
        state = "localized_but_target_fidelity_unqualified"
    else:
        state = "unqualified_both"
    return {
        "state": state,
        "deployment_qualified": state == "deployment_qualified",
        "target_learning_passed": target_ok,
        "localization_passed": loc_ok,
        "reasons": reasons,
    }


@dataclass(frozen=True)
class FormalLocalizationMetrics:
    metric_source_identity: str
    pixel_auroc: float
    pixel_ap: float
    aupro: float
    teacher_spearman: float
    teacher_top1_overlap: float
    invocation_proof: Mapping[str, bool]


@dataclass(frozen=True)
class FormalLocalizationGateEvidence:
    metric_source_identity: str
    delta_pixel_ap_macro: float
    delta_pixel_auroc_macro: float
    delta_aupro_macro: float
    per_category_localization: Mapping[str, Mapping[str, float]]


def compute_formal_localization_metrics(
    *,
    image_labels: Any,
    image_scores: Any,
    masks: Any,
    anomaly_maps: Any,
    teacher_map: Any | None = None,
) -> FormalLocalizationMetrics:
    """Formal adapter: production paper metrics + contribution teacher fidelity only."""

    import numpy as np

    from rad.evaluation import paper_metrics
    from rad.phase_b import b2_contribution_targets as contrib_mod

    proof = {
        "compute_paper_metrics": False,
        "spearman_fidelity": False,
        "top1_overlap": False,
    }
    try:
        metrics = paper_metrics.compute_paper_metrics(
            image_labels=np.asarray(image_labels),
            image_scores=np.asarray(image_scores),
            masks=np.asarray(masks),
            anomaly_maps=np.asarray(anomaly_maps),
            boundary_enabled=False,
        )
        proof["compute_paper_metrics"] = True
    except Exception as exc:  # noqa: BLE001 — fail closed with contract code
        _fail("B2_DLCM_EVAL_METRIC_UNDEFINED", f"production paper metrics failed: {exc}")
    if teacher_map is None:
        _fail("B2_DLCM_EVAL_METRIC_UNDEFINED", "teacher_map required for formal adapter")
    try:
        anomaly_t = (
            anomaly_maps
            if isinstance(anomaly_maps, torch.Tensor)
            else torch.as_tensor(anomaly_maps, dtype=torch.float32)
        )
        teacher_t = (
            teacher_map
            if isinstance(teacher_map, torch.Tensor)
            else torch.as_tensor(teacher_map, dtype=torch.float32)
        )
        if anomaly_t.ndim == 2:
            anomaly_t = anomaly_t.unsqueeze(0)
            teacher_t = teacher_t.unsqueeze(0)
        if anomaly_t.ndim != 3 or teacher_t.shape != anomaly_t.shape:
            _fail(
                "B2_DLCM_EVAL_METRIC_UNDEFINED",
                "anomaly/teacher maps must be [H,W] or [N,H,W] with equal shapes",
            )
        spearman_vals: list[float] = []
        overlap_vals: list[float] = []
        for idx in range(int(anomaly_t.shape[0])):
            spearman_vals.append(
                float(contrib_mod.spearman_fidelity(anomaly_t[idx], teacher_t[idx]).raw)
            )
            overlap_vals.append(float(contrib_mod.top1_overlap(anomaly_t[idx], teacher_t[idx])))
        proof["spearman_fidelity"] = True
        proof["top1_overlap"] = True
        spearman_raw = float(sum(spearman_vals) / len(spearman_vals))
        overlap_raw = float(sum(overlap_vals) / len(overlap_vals))
    except B2DLCMDeploymentError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail("B2_DLCM_EVAL_METRIC_UNDEFINED", f"teacher fidelity metrics failed: {exc}")
    if not all(proof.values()):
        _fail("B2_DLCM_EVAL_METRIC_UNDEFINED", "incomplete production metric invocation")
    return FormalLocalizationMetrics(
        metric_source_identity=FORMAL_LOCALIZATION_ADAPTER_ID,
        pixel_auroc=float(metrics.pixel_auroc),
        pixel_ap=float(metrics.pixel_ap),
        aupro=float(metrics.pixel_aupro),
        teacher_spearman=spearman_raw,
        teacher_top1_overlap=overlap_raw,
        invocation_proof=dict(proof),
    )


def build_formal_localization_gate_evidence(
    *,
    per_category_metrics: Mapping[str, FormalLocalizationMetrics],
    per_category_baseline: Mapping[str, FormalLocalizationMetrics],
) -> FormalLocalizationGateEvidence:
    if not per_category_metrics:
        _fail("B2_DLCM_EVAL_METRIC_UNDEFINED", "no formal category metrics")
    deltas: dict[str, dict[str, float]] = {}
    ap_vals: list[float] = []
    auroc_vals: list[float] = []
    aupro_vals: list[float] = []
    for category, metrics in sorted(per_category_metrics.items()):
        if metrics.metric_source_identity != FORMAL_LOCALIZATION_ADAPTER_ID:
            _fail("B2_DLCM_EVAL_METRIC_UNDEFINED", f"non-formal metrics for {category}")
        if category not in per_category_baseline:
            _fail("B2_DLCM_EVAL_METRIC_UNDEFINED", f"missing baseline for {category}")
        baseline = per_category_baseline[category]
        if baseline.metric_source_identity != FORMAL_LOCALIZATION_ADAPTER_ID:
            _fail("B2_DLCM_EVAL_METRIC_UNDEFINED", f"non-formal baseline for {category}")
        delta = {
            "delta_pixel_ap": float(metrics.pixel_ap - baseline.pixel_ap),
            "delta_pixel_auroc": float(metrics.pixel_auroc - baseline.pixel_auroc),
            "delta_aupro": float(metrics.aupro - baseline.aupro),
        }
        deltas[category] = delta
        ap_vals.append(delta["delta_pixel_ap"])
        auroc_vals.append(delta["delta_pixel_auroc"])
        aupro_vals.append(delta["delta_aupro"])
    return FormalLocalizationGateEvidence(
        metric_source_identity=FORMAL_LOCALIZATION_ADAPTER_ID,
        delta_pixel_ap_macro=float(sum(ap_vals) / len(ap_vals)),
        delta_pixel_auroc_macro=float(sum(auroc_vals) / len(auroc_vals)),
        delta_aupro_macro=float(sum(aupro_vals) / len(aupro_vals)),
        per_category_localization=deltas,
    )


def build_accepted_manifest_identities(
    *,
    deploy_identity: str,
    qualification_identity: str,
    selection_identity: str,
    upstream_identities: Mapping[str, str],
) -> dict[str, str]:
    accepted = _canonical_json_sha256(
        {
            "deploy": deploy_identity,
            "qualification": qualification_identity,
            "selection": selection_identity,
            "upstream": dict(sorted(upstream_identities.items())),
            "accepted_contract": "b2_dlcm_accepted_v1",
        }
    )
    return {
        "deploy_identity": deploy_identity,
        "qualification_identity": qualification_identity,
        "accepted_identity": accepted,
    }


def allocation_jsd(p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Natural-log JSD without epsilon; 0 log 0 := 0."""

    m = 0.5 * (p + w)
    return 0.5 * _kl_zero_safe(p, m) + 0.5 * _kl_zero_safe(w, m)


def _kl_zero_safe(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    positive = p > 0
    log_p = torch.where(positive, torch.log(torch.where(positive, p, torch.ones_like(p))), torch.zeros_like(p))
    log_q = torch.where(q > 0, torch.log(torch.where(q > 0, q, torch.ones_like(q))), torch.zeros_like(q))
    terms = torch.where(positive, p * (log_p - log_q), torch.zeros_like(p))
    return terms.sum(dim=-1)


def top1_set_agreement(target: torch.Tensor, weights: torch.Tensor, *, tol: float = 1e-6) -> float:
    t_max = float(target.max())
    p_max = float(weights.max())
    t_set = {i for i, v in enumerate(target.tolist()) if v >= t_max - tol}
    p_set = {i for i, v in enumerate(weights.tolist()) if v >= p_max - tol}
    return 1.0 if t_set.intersection(p_set) else 0.0


def spearman_average_ranks(x: torch.Tensor, y: torch.Tensor) -> float:
    if bool((x == x[0]).all()) and bool((y == y[0]).all()):
        return 1.0
    if bool((x == x[0]).all()) ^ bool((y == y[0]).all()):
        return 0.0
    def ranks(v: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(v)
        r = torch.empty_like(v, dtype=torch.float64)
        # Average ranks for ties: simple dense average via sorting groups.
        values = v.to(torch.float64)
        sorted_vals = values[order]
        avg = torch.empty(len(v), dtype=torch.float64)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            mean_rank = 0.5 * (i + j) + 1.0
            avg[i : j + 1] = mean_rank
            i = j + 1
        r[order] = avg
        return r
    rx = ranks(x)
    ry = ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx * rx).sum() * (ry * ry).sum())
    if float(denom) == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


# ---------------------------------------------------------------------------
# Evaluation unlock, seed collection, metrics aggregation, accepted manifests
# ---------------------------------------------------------------------------

EVALUATION_UNLOCK_SCHEMA_VERSION = "b2_dlcm_evaluation_unlock_v1"
SEED_COLLECTION_SCHEMA_VERSION = "b2_dlcm_seed_collection_v1"
EVALUATION_MANIFEST_SCHEMA_VERSION = "b2_dlcm_evaluation_manifest_v1"
ACCEPTED_MANIFEST_SCHEMA_VERSION = "b2_dlcm_accepted_deployment_v1"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    raw = encoded.encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def pairwise_ranking_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    tie_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Pairwise ranking accuracy; no_valid_pairs is not scored as 0/1."""

    if pred.ndim != 1 or target.ndim != 1 or pred.shape != target.shape:
        _fail("B2_DLCM_RANK_ACC_SHAPE", "pred/target must be 1-D equal shape")
    n = int(pred.numel())
    correct = 0
    valid = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = float(target[i] - target[j])
            if abs(diff) <= tie_tolerance:
                continue
            valid += 1
            pred_diff = float(pred[i] - pred[j])
            if (diff > 0 and pred_diff > 0) or (diff < 0 and pred_diff < 0):
                correct += 1
    if valid == 0:
        return {
            "status": "no_valid_pairs",
            "accuracy": None,
            "valid_pair_count": 0,
        }
    return {
        "status": "ok",
        "accuracy": correct / float(valid),
        "valid_pair_count": valid,
    }


def build_evaluation_unlock_artifact(
    *,
    canonical_selection_identity: str,
    reproduction_comparison: Mapping[str, Any],
    trace_node_comparisons: Sequence[Mapping[str, Any]],
    best_model_identity: str,
    last_model_identity: str,
    best_training_identity: str,
    last_training_identity: str,
    checkpoint_bytes_equal: bool,
    deployment_scientific_identity: str,
    environment_identity: str,
    descriptor_normalization_identity: str,
    contribution_target_identity: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": EVALUATION_UNLOCK_SCHEMA_VERSION,
        "canonical_selection_identity": canonical_selection_identity,
        "reproduction_comparison": dict(reproduction_comparison),
        "trace_node_comparisons": list(trace_node_comparisons),
        "best_model_identity": best_model_identity,
        "last_model_identity": last_model_identity,
        "best_training_identity": best_training_identity,
        "last_training_identity": last_training_identity,
        "checkpoint_bytes_equal": bool(checkpoint_bytes_equal),
        "deployment_scientific_identity": deployment_scientific_identity,
        "environment_identity": environment_identity,
        "descriptor_normalization_identity": descriptor_normalization_identity,
        "contribution_target_identity": contribution_target_identity,
        "evaluation_unlocked": True,
    }
    payload["evaluation_unlock_scientific_sha256"] = _canonical_json_sha256(
        {k: v for k, v in payload.items() if k != "evaluation_unlock_scientific_sha256"}
    )
    return payload


def persist_evaluation_unlock(path: Path | str, unlock: Mapping[str, Any]) -> str:
    return _atomic_write_json(Path(path), unlock)


def require_evaluation_unlocked(
    run_dir: Path | str,
    *,
    cli_unlock_flag: bool | None = None,
) -> dict[str, Any]:
    if cli_unlock_flag is True:
        _fail(
            "B2_DLCM_EVAL_CLI_BYPASS",
            "evaluation unlock cannot be granted by a CLI boolean",
        )
    path = Path(run_dir) / "evaluation_unlock.json"
    if not path.is_file():
        _fail("B2_DLCM_EVAL_LOCKED", "evaluation_unlock.json missing; evaluation locked")
    unlock = json.loads(path.read_text(encoding="utf-8"))
    if unlock.get("evaluation_unlocked") is not True:
        _fail("B2_DLCM_EVAL_LOCKED", "evaluation_unlocked is not true")
    claimed = unlock.get("evaluation_unlock_scientific_sha256")
    recomputed = _canonical_json_sha256(
        {k: v for k, v in unlock.items() if k != "evaluation_unlock_scientific_sha256"}
    )
    if claimed != recomputed:
        _fail("B2_DLCM_EVAL_UNLOCK_HASH", "evaluation unlock scientific hash mismatch")
    receipt = path.with_suffix(path.suffix + ".sha256")
    if not receipt.is_file():
        _fail("B2_DLCM_EVAL_UNLOCK_RECEIPT", "evaluation unlock receipt missing")
    return unlock


def seed_collection_scientific_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "ordered_seeds": list(manifest["ordered_seeds"]),
        "seed_scientific_identities": dict(manifest["seed_scientific_identities"]),
        "calibration_by_seed": dict(manifest["calibration_by_seed"]),
        "best_epoch_by_seed": dict(manifest["best_epoch_by_seed"]),
        "best_model_identity_by_seed": dict(manifest["best_model_identity_by_seed"]),
        "training_config_identity": manifest["training_config_identity"],
        "upstream_identities": dict(manifest["upstream_identities"]),
        "environment_identity": manifest["environment_identity"],
        "all_seeds_passed": manifest["all_seeds_passed"],
        "evaluation_unlocked": manifest["evaluation_unlocked"],
        "canonical_seed_selected": manifest["canonical_seed_selected"],
    }


def seed_collection_scientific_sha256(manifest: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(seed_collection_scientific_payload(manifest))


def build_seed_collection_manifest(
    seeds: Sequence[Mapping[str, Any]],
    *,
    training_config_identity: str,
    upstream_identities: Mapping[str, str],
) -> dict[str, Any]:
    if [int(s["seed"]) for s in seeds] != [17, 29, 43]:
        _fail("B2_DLCM_SEED_ORDER_INVALID", "seeds must be exactly [17,29,43]")
    if any(s.get("status") != "passed" for s in seeds):
        _fail("B2_DLCM_SEED_COLLECTION_INCOMPLETE", "all seeds must be passed")
    env_ids = {s["environment_identity"] for s in seeds}
    if len(env_ids) != 1:
        _fail("B2_DLCM_ENV_IDENTITY_MISMATCH", "seed environment identities must agree")
    manifest = {
        "schema_version": SEED_COLLECTION_SCHEMA_VERSION,
        "ordered_seeds": [17, 29, 43],
        "seed_scientific_identities": {
            str(s["seed"]): s["seed_scientific_sha256"] for s in seeds
        },
        "seed_file_sha256_by_seed": {str(s["seed"]): s["file_sha256"] for s in seeds},
        "calibration_by_seed": {
            str(s["seed"]): {
                "primary": s["calibration_primary"],
                "secondary": s["calibration_secondary"],
            }
            for s in seeds
        },
        "best_epoch_by_seed": {str(s["seed"]): int(s["best_epoch"]) for s in seeds},
        "best_model_identity_by_seed": {
            str(s["seed"]): s["best_model_state_identity"] for s in seeds
        },
        "training_config_identity": training_config_identity,
        "upstream_identities": dict(upstream_identities),
        "environment_identity": next(iter(env_ids)),
        "all_seeds_passed": True,
        "evaluation_unlocked": False,
        "canonical_seed_selected": False,
    }
    manifest["seed_collection_scientific_sha256"] = seed_collection_scientific_sha256(manifest)
    return manifest


def verify_seed_collection_file_hashes(
    manifest: Mapping[str, Any],
    expected_by_seed: Mapping[int, str],
) -> None:
    for seed, digest in expected_by_seed.items():
        got = manifest["seed_file_sha256_by_seed"].get(str(seed))
        if got != digest:
            _fail("B2_DLCM_SEED_FILE_SHA", f"seed {seed} file sha mismatch")


def compare_reproduction(
    original: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    orig_nodes = list(original.get("nodes", []))
    repro_nodes = list(reproduction.get("nodes", []))
    limit = max(len(orig_nodes), len(repro_nodes))
    for idx in range(limit):
        if idx >= len(orig_nodes) or idx >= len(repro_nodes):
            return {
                "status": "canonical_reproduction_failed",
                "first_mismatch": {"epoch": idx, "field": "nodes_length"},
                "nodes_equal": False,
            }
        if orig_nodes[idx] != repro_nodes[idx]:
            return {
                "status": "canonical_reproduction_failed",
                "first_mismatch": {
                    "epoch": int(orig_nodes[idx].get("epoch", idx)),
                    "field": "node",
                },
                "nodes_equal": False,
            }
    if original.get("model") != reproduction.get("model"):
        return {
            "status": "canonical_reproduction_failed",
            "first_mismatch": {"epoch": None, "field": "model"},
            "nodes_equal": True,
        }
    return {"status": "passed", "first_mismatch": None, "nodes_equal": True}


def aggregate_target_fidelity(
    per_sample: Sequence[Mapping[str, Any]],
    *,
    metric_path: Sequence[str],
) -> dict[str, Any]:
    def _dig(payload: Mapping[str, Any], path: Sequence[str]) -> float:
        cur: Any = payload
        for key in path:
            cur = cur[key]
        return float(cur)

    by_category: dict[str, list[float]] = {}
    all_values: list[float] = []
    for sample in per_sample:
        value = _dig(sample, metric_path)
        all_values.append(value)
        by_category.setdefault(str(sample["category"]), []).append(value)
    per_category = {
        category: sum(values) / float(len(values)) for category, values in sorted(by_category.items())
    }
    category_macro = sum(per_category.values()) / float(len(per_category))
    pooled = sum(all_values) / float(len(all_values))
    return {
        "category_macro": category_macro,
        "pooled_diagnostic": pooled,
        "per_category": per_category,
    }


def three_seed_summary(values: Sequence[float], *, ddof: int = 0) -> dict[str, float]:
    if ddof != 0:
        _fail("B2_DLCM_SEED_SUMMARY_DDOF", "seed summary requires ddof=0")
    if len(values) != 3:
        _fail("B2_DLCM_SEED_SUMMARY_COUNT", "exactly three seed values required")
    acc = [float(v) for v in values]
    mean = sum(acc) / 3.0
    var = sum((v - mean) ** 2 for v in acc) / 3.0
    return {"mean": mean, "std": var**0.5}


def build_evaluation_record(
    *,
    evaluated_checkpoint_scientific_identity: str,
    evaluation_unlock_identity: str,
    evaluation_split_coverage_sha256: str,
    no_parameter_update_proof: bool,
    depth_results: Mapping[int, Mapping[str, Any]],
    per_category: Mapping[str, Mapping[str, Any]],
    pooled: Mapping[str, Any],
) -> dict[str, Any]:
    if no_parameter_update_proof is not True:
        _fail("B2_DLCM_EVAL_PARAM_UPDATE", "evaluation must prove no parameter updates")
    return {
        "evaluated_checkpoint_scientific_identity": evaluated_checkpoint_scientific_identity,
        "evaluation_unlock_identity": evaluation_unlock_identity,
        "evaluation_split_coverage_sha256": evaluation_split_coverage_sha256,
        "no_parameter_update_proof": True,
        "depth_results": {str(k): dict(v) for k, v in depth_results.items()},
        "per_category": dict(per_category),
        "pooled": dict(pooled),
    }


def build_evaluation_manifest(
    *,
    records: Mapping[str, Mapping[str, Any]],
    unlock_identity: str,
) -> dict[str, Any]:
    required = {
        "seed_17",
        "seed_29",
        "seed_43",
        "canonical_deployment",
    }
    if set(records) != required:
        _fail("B2_DLCM_EVAL_MANIFEST_KEYS", f"evaluation records must be exactly {sorted(required)}")
    manifest = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "evaluation_unlock_identity": unlock_identity,
        "record_identities": {
            key: _canonical_json_sha256(dict(records[key])) for key in sorted(records)
        },
        "canonical_reproduction_excluded_from_statistics": True,
        "seed_summary_ddof": 0,
    }
    manifest["evaluation_manifest_scientific_sha256"] = _canonical_json_sha256(
        {k: v for k, v in manifest.items() if k != "evaluation_manifest_scientific_sha256"}
    )
    return manifest


def persist_evaluation_bundle(
    evaluation_dir: Path | str,
    manifest: Mapping[str, Any],
    record_files: Mapping[str, Mapping[str, Any]],
) -> None:
    out = Path(evaluation_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in record_files.items():
        _atomic_write_json(out / name, payload)
    _atomic_write_json(out / "evaluation_manifest.json", manifest)


def build_accepted_deployment_manifest(
    *,
    deploy_identity: str,
    qualification_identity: str,
    accepted_identity: str,
    selection_identity: str,
    deployment_scientific_sha256: str,
    upstream_identities: Mapping[str, str],
    deployment_qualified: bool,
) -> dict[str, Any]:
    return {
        "schema_version": ACCEPTED_MANIFEST_SCHEMA_VERSION,
        "deploy_identity": deploy_identity,
        "qualification_identity": qualification_identity,
        "accepted_identity": accepted_identity,
        "selection_identity": selection_identity,
        "deployment_scientific_sha256": deployment_scientific_sha256,
        "upstream_identities": dict(upstream_identities),
        "deployment_qualified": bool(deployment_qualified),
    }


def verify_accepted_manifest_for_loader(
    manifest: Mapping[str, Any],
    *,
    deployment_scientific_sha256: str,
    deployment_file_sha256: str,
    expected_accepted_identity: str,
) -> None:
    if not isinstance(manifest, Mapping):
        _fail("B2_DLCM_NOT_ACCEPTED", "accepted manifest required")
    if manifest.get("deployment_qualified") is not True:
        _fail("B2_DLCM_NOT_ACCEPTED", "deployment_qualified must be true")
    if not deployment_file_sha256 or len(deployment_file_sha256) != 64:
        _fail("B2_DLCM_NOT_ACCEPTED", "deployment file sha required")
    if manifest.get("deployment_scientific_sha256") != deployment_scientific_sha256:
        _fail("B2_DLCM_NOT_ACCEPTED", "deployment scientific identity mismatch")
    if expected_accepted_identity and manifest.get("accepted_identity") != expected_accepted_identity:
        _fail("B2_DLCM_NOT_ACCEPTED", "accepted identity mismatch")
    for key in ("deploy_identity", "qualification_identity", "accepted_identity"):
        if key not in manifest:
            _fail("B2_DLCM_NOT_ACCEPTED", f"missing {key}")


def require_formal_localization_defined(
    per_category: Mapping[str, Mapping[str, Any]],
) -> None:
    required_metrics = ("pixel_ap", "pixel_auroc", "aupro")
    for category, metrics in per_category.items():
        for name in required_metrics:
            value = metrics.get(name)
            if value is None or (isinstance(value, float) and value != value):
                _fail(
                    "B2_DLCM_EVAL_METRIC_UNDEFINED",
                    f"formal localization metric {name} undefined for {category}",
                )


def require_passed_seed_collection_for_canonical(run_dir: Path | str) -> None:
    failure = Path(run_dir) / "collection_failure_manifest.json"
    if failure.is_file():
        _fail("B2_DLCM_COLLECTION_FAILED", "seed collection failed; canonical selection blocked")
    # Absence of failure is insufficient alone for production; B2-05A contract gate:
    collection = Path(run_dir) / "seed_collection_manifest.json"
    if not collection.is_file():
        _fail("B2_DLCM_COLLECTION_FAILED", "passed seed collection manifest required")


def require_reproduction_passed_for_evaluation(comparison: Mapping[str, Any]) -> None:
    if comparison.get("status") != "passed":
        _fail(
            "B2_DLCM_REPRO_BLOCKED",
            "canonical reproduction failure blocks evaluation/deployment export",
        )


def dual_family_diagnostic(
    gt_metrics: Mapping[str, Any],
    teacher_metrics: Mapping[str, Any],
    *,
    metric: str,
) -> dict[str, Any]:
    if gt_metrics.get("valid") is False or teacher_metrics.get("valid") is False:
        return {"status": "invalid", "value": None, "metric": metric}
    if metric not in gt_metrics or metric not in teacher_metrics:
        return {"status": "invalid", "value": None, "metric": metric}
    return {
        "status": "ok",
        "value": 0.5 * (float(gt_metrics[metric]) + float(teacher_metrics[metric])),
        "metric": metric,
    }


def scientific_hash_rejecting_runtime_fields(
    payload: Mapping[str, Any],
    *,
    whitelist: Sequence[str],
) -> str:
    banned = {
        "hostname",
        "gpu_uuid",
        "absolute_path",
        "path",
        "timestamp",
        "pid",
        "start_time",
        "end_time",
    }
    runtime = [key for key in payload if key in banned]
    if runtime:
        _fail("B2_DLCM_SCI_RUNTIME_FIELD", f"runtime fields forbidden in scientific hash: {runtime}")
    unknown = [key for key in payload if key not in whitelist]
    if unknown:
        _fail("B2_DLCM_SCI_RUNTIME_FIELD", f"undeclared scientific fields: {unknown}")
    projected = {key: payload[key] for key in whitelist}
    return _canonical_json_sha256(projected)
