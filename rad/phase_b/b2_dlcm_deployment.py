"""B2-05A DLCM deployment loading, qualification, and accepted-manifest contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    mean = torch.tensor(embedded["mean"], dtype=torch.float64)
    std = torch.tensor(embedded["std"], dtype=torch.float64)
    zero_var = list(embedded["zero_variance"])
    x64 = raw.to(torch.float64)
    out = (x64 - mean) / std
    for axis, is_zero in enumerate(zero_var):
        if is_zero:
            out[..., axis] = 0.0
    if not bool(torch.isfinite(out).all()):
        _fail("B2_DLCM_NORM_NONFINITE", "normalized descriptors nonfinite")
    return out.to(torch.float32).contiguous()


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
        normalized = apply_embedded_normalization(raw_descriptors, self._embedded_normalization)
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
        normalized = apply_embedded_normalization(raw_descriptors, self._embedded_normalization)
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
        if not require_accepted_manifest.get("deployment_qualified", False):
            _fail("B2_DLCM_NOT_ACCEPTED", "formal loader requires accepted qualified manifest")
    run_cpu_golden_self_test(checkpoint)
    trunk = _instantiate_trunk_from_checkpoint(checkpoint)
    trunk.load_state_dict(checkpoint["state_dict"], strict=True)
    trunk.eval()
    for param in trunk.parameters():
        param.requires_grad_(False)

    if device.type == "cuda":
        trunk.to(device)
        trunk._reset_dropout_generators(device=device)
        # GPU qualification against golden CPU expectations.
        for case in checkpoint["golden_cases"]:
            inp = case["input_tensor"].to(device)
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
            cpu_logits = torch.tensor(
                [dlcm.bits_hex_to_float(m) for m in case["expected_logits_bits"]],
                dtype=torch.float32,
            ).view_as(logits.cpu())
            cpu_weights = torch.tensor(
                [dlcm.bits_hex_to_float(m) for m in case["expected_weights_bits"]],
                dtype=torch.float32,
            ).view_as(weights.cpu())
            if float((logits.cpu() - cpu_logits).abs().max()) > gpu_atol:
                _fail("B2_DLCM_GPU_QUAL_FAIL", "GPU logits drift exceeds 1e-6")
            if float((weights.cpu() - cpu_weights).abs().max()) > gpu_atol:
                _fail("B2_DLCM_GPU_QUAL_FAIL", "GPU weights drift exceeds 1e-6")
    elif device.type != "cpu":
        _fail("B2_DLCM_DEVICE_INVALID", f"unsupported device {device}")

    state_id = dlcm.model_state_scientific_sha256(trunk)
    key = QualificationCacheKey(
        accepted_deployment_scientific_sha256=str(checkpoint["deployment_scientific_sha256"]),
        checkpoint_file_sha256=checkpoint_file_sha256,
        environment_contract_sha256=environment_contract_sha256,
        loader_contract_version=LOADER_CONTRACT_VERSION,
        gpu_device_index=device.index or 0 if device.type == "cuda" else -1,
        gpu_uuid=gpu_uuid,
        gpu_model=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    )
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
    _PROCESS_QUAL_CACHE[_cache_key_str(key)] = wrapper
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
