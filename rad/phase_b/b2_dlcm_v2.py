"""B2-05C1 decoupled DLCM V2 architecture, losses, and deployment trunk.

Four training heads: GT deployment allocation, teacher auxiliary allocation,
GT signed, teacher signed. Deployment artifact retains trunk + GT deployment
head only. Reuses V1 KL/Huber/ranking/fusion/RNG semantics without alteration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F

from rad.phase_b import b2_dlcm as v1

DEFAULT_CANDIDATE_LAYERS = v1.DEFAULT_CANDIDATE_LAYERS
DEFAULT_PREDICTION_DEPTHS = v1.DEFAULT_PREDICTION_DEPTHS
DEFAULT_DESCRIPTOR_DIMENSION = v1.DEFAULT_DESCRIPTOR_DIMENSION
DEFAULT_LAYER_EMBEDDING_DIM = v1.DEFAULT_LAYER_EMBEDDING_DIM
DEFAULT_DEPTH_EMBEDDING_DIM = v1.DEFAULT_DEPTH_EMBEDDING_DIM
DEFAULT_HIDDEN_DIMENSION = v1.DEFAULT_HIDDEN_DIMENSION
DEFAULT_DROPOUT_P = v1.DEFAULT_DROPOUT_P
ARCHITECTURE_CONTRACT_VERSION = "b2_dlcm_architecture_v2"
MODEL_CLASS_ID = "rad.phase_b.b2_dlcm_v2.B2DLCMV2"

TEACHER_ALLOC_WEIGHT = 0.25
GT_SIGNED_WEIGHT = 0.25
TEACHER_SIGNED_WEIGHT = 0.0625

# Immutable V1 evidence identity pins (must never change).
V1_EVIDENCE_TAG = "b2-dlcm-unqualified-evidence-v1"
V1_EVIDENCE_COMMIT = "43d856f5ff771957f9f39d0909b1bc87d6b7081b"
V1_ACCEPTED_TRAINING_PLAN = "59e20f4cb337ef42384f70bb8b3dad5211d906341b0a2d41f7e6847610635980"
V1_SEED_COLLECTION = "94a6a9332a0694889c7a0255814ac13fe8316c601529197063165ce14ec1277f"
V1_CANONICAL_SEED = 17
V1_DEPLOYMENT_SCIENTIFIC = "4cbc6fb88f39ed86deacfbbe48580f7682453b94becb046ec6ef1b1302df378a"
V1_EVALUATION_UNLOCK = "19dca41e9f647d12afce9877a7340f5af58bf9a23997d7339dded26d89fe73dd"
V1_QUALIFICATION_SCIENTIFIC = "da51e5fc1302cf507bc844f87e82cb66f7d2fa0a13e61f28a0dba14333201c49"
V1_VERDICT = "localized_but_target_fidelity_unqualified"


class B2DLCMV2Error(RuntimeError):
    """Fail-closed V2 DLCM contract error."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV2Error(code, detail)


@dataclass
class B2DLCMV2Outputs:
    gt_deployment_logits: torch.Tensor
    gt_deployment_weights: torch.Tensor
    teacher_allocation_logits: torch.Tensor
    teacher_allocation_weights: torch.Tensor
    gt_signed: torch.Tensor
    teacher_signed: torch.Tensor
    player_features: torch.Tensor


def v1_immutable_identity() -> dict[str, Any]:
    return {
        "tag": V1_EVIDENCE_TAG,
        "commit": V1_EVIDENCE_COMMIT,
        "accepted_training_plan": V1_ACCEPTED_TRAINING_PLAN,
        "seed_collection": V1_SEED_COLLECTION,
        "canonical_seed": V1_CANONICAL_SEED,
        "deployment_scientific_sha256": V1_DEPLOYMENT_SCIENTIFIC,
        "evaluation_unlock": V1_EVALUATION_UNLOCK,
        "qualification_scientific_sha256": V1_QUALIFICATION_SCIENTIFIC,
        "verdict": V1_VERDICT,
    }


class B2DLCMV2(nn.Module):
    """Decoupled V2 DLCM with four training heads."""

    def __init__(
        self,
        seed: int | None,
        *,
        candidate_layers: Sequence[int] = DEFAULT_CANDIDATE_LAYERS,
        prediction_depths: Sequence[int] = DEFAULT_PREDICTION_DEPTHS,
        descriptor_dimension: int = DEFAULT_DESCRIPTOR_DIMENSION,
        layer_embedding_dimension: int = DEFAULT_LAYER_EMBEDDING_DIM,
        depth_embedding_dimension: int = DEFAULT_DEPTH_EMBEDDING_DIM,
        hidden_dimension: int = DEFAULT_HIDDEN_DIMENSION,
        dropout_probability: float = DEFAULT_DROPOUT_P,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.candidate_layers = tuple(int(layer) for layer in candidate_layers)
        self.prediction_depths = tuple(int(depth) for depth in prediction_depths)
        if not self.candidate_layers:
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "candidate_layers must be non-empty")
        if sorted(self.candidate_layers) != list(self.candidate_layers):
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "candidate_layers must be ascending")
        for depth in self.prediction_depths:
            if depth not in self.candidate_layers:
                _fail("B2_DLCM_V2_CONTRACT_MISMATCH", f"prediction depth {depth} missing")
        self.descriptor_dimension = int(descriptor_dimension)
        self.layer_embedding_dimension = int(layer_embedding_dimension)
        self.depth_embedding_dimension = int(depth_embedding_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.dropout_probability = float(dropout_probability)
        self.model_seed = seed

        self.layer_id_to_index = {layer: idx for idx, layer in enumerate(self.candidate_layers)}
        self.depth_id_to_index = {depth: idx for idx, depth in enumerate(self.prediction_depths)}

        self.layer_embedding = nn.Embedding(len(self.candidate_layers), self.layer_embedding_dimension)
        self.depth_embedding = nn.Embedding(len(self.prediction_depths), self.depth_embedding_dimension)
        encoder_in = (
            self.descriptor_dimension + self.layer_embedding_dimension + self.depth_embedding_dimension
        )
        self.layer_encoder = v1.LayerEncoder(
            input_dim=encoder_in,
            hidden_dim=self.hidden_dimension,
            dropout_p=self.dropout_probability,
        )
        self.context_encoder = v1.ContextEncoder(
            input_dim=self.hidden_dimension * 3,
            hidden_dim=self.hidden_dimension,
            dropout_p=self.dropout_probability,
        )
        self.gt_deployment_head = nn.Linear(self.hidden_dimension, 1)
        self.teacher_allocation_head = nn.Linear(self.hidden_dimension, 1)
        self.gt_signed_head = nn.Linear(self.hidden_dimension, 1)
        self.teacher_signed_head = nn.Linear(self.hidden_dimension, 1)

        self.component_seeds: dict[str, int] = {}
        self.dropout_site_seeds: dict[str, int] = {}
        self.dropout_generators: dict[str, torch.Generator] = {}
        self._init_generator: torch.Generator | None = None

        if initialize:
            if seed is None:
                _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "seed required for initialization")
            self._initialize_from_seed(int(seed))

    def players_for_depth(self, depth: int) -> tuple[int, ...]:
        return v1.players_for_depth(self.candidate_layers, depth)

    def _initialize_from_seed(self, seed: int) -> None:
        self.component_seeds = v1.derive_component_seeds(seed)
        self.dropout_site_seeds = v1.derive_dropout_site_seeds(self.component_seeds["dropout"])
        init_seed = self.component_seeds["model_initialization"]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(init_seed)
        self._init_generator = generator

        ordered_steps: list[tuple[str, Any]] = [
            ("layer_embedding", self.layer_embedding),
            ("depth_embedding", self.depth_embedding),
            ("layer_encoder.block_1.linear", self.layer_encoder.block_1.linear),
            ("layer_encoder.block_1.norm", self.layer_encoder.block_1.norm),
            ("layer_encoder.block_2.linear", self.layer_encoder.block_2.linear),
            ("layer_encoder.block_2.norm", self.layer_encoder.block_2.norm),
            ("context_encoder.block_1.linear", self.context_encoder.block_1.linear),
            ("context_encoder.block_1.norm", self.context_encoder.block_1.norm),
            ("context_encoder.block_2.linear", self.context_encoder.block_2.linear),
            ("context_encoder.block_2.norm", self.context_encoder.block_2.norm),
            ("gt_signed_head", self.gt_signed_head),
            ("teacher_signed_head", self.teacher_signed_head),
            ("gt_deployment_head", self.gt_deployment_head),
            ("teacher_allocation_head", self.teacher_allocation_head),
        ]
        for name, module in ordered_steps:
            if isinstance(module, nn.Embedding):
                with torch.no_grad():
                    module.weight.normal_(0.0, 0.02, generator=generator)
            elif isinstance(module, nn.Linear):
                if name in {"gt_deployment_head", "teacher_allocation_head"}:
                    v1._zeros_(module.weight)
                    v1._zeros_(module.bias)
                else:
                    v1._xavier_uniform_(module.weight, generator)
                    v1._zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                v1._ones_(module.weight)
                v1._zeros_(module.bias)
            else:
                _fail("B2_DLCM_V2_CONTRACT_MISMATCH", f"unknown init target {name}")
        self._reset_dropout_generators(device="cpu")

    def _reset_dropout_generators(self, *, device: str | torch.device) -> None:
        device_t = torch.device(device)
        gens: dict[str, torch.Generator] = {}
        for site, site_seed in self.dropout_site_seeds.items():
            gen = torch.Generator(device=device_t)
            gen.manual_seed(int(site_seed))
            gens[site] = gen
        self.dropout_generators = gens

    def forward_training(
        self,
        descriptors: torch.Tensor,
        *,
        prediction_depth: int,
        player_layer_ids: Sequence[int] | None = None,
    ) -> B2DLCMV2Outputs:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.descriptor_dimension:
            _fail(
                "B2_DLCM_V2_CONTRACT_MISMATCH",
                f"expected [B,n,{self.descriptor_dimension}], got {tuple(descriptors.shape)}",
            )
        if not bool(torch.isfinite(descriptors).all()):
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "descriptors must be finite")
        if descriptors.dtype != torch.float32:
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "descriptors must be float32")

        expected = self.players_for_depth(prediction_depth)
        if player_layer_ids is None:
            player_layer_ids = expected
        else:
            got = tuple(int(layer) for layer in player_layer_ids)
            if got != expected:
                _fail(
                    "B2_DLCM_V2_CONTRACT_MISMATCH",
                    f"depth {prediction_depth} expects {expected}, got {got}",
                )
            player_layer_ids = got

        batch, n_players, _ = descriptors.shape
        if n_players != len(expected):
            _fail(
                "B2_DLCM_V2_CONTRACT_MISMATCH",
                f"depth {prediction_depth} expects {len(expected)} players, got {n_players}",
            )
        if prediction_depth not in self.depth_id_to_index:
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", f"unsupported prediction depth {prediction_depth}")

        layer_indices = torch.tensor(
            [self.layer_id_to_index[layer] for layer in expected],
            dtype=torch.long,
            device=descriptors.device,
        )
        depth_index = torch.tensor(
            self.depth_id_to_index[prediction_depth],
            dtype=torch.long,
            device=descriptors.device,
        )
        layer_emb = self.layer_embedding(layer_indices).unsqueeze(0).expand(batch, -1, -1)
        depth_emb = self.depth_embedding(depth_index).view(1, 1, -1).expand(batch, n_players, -1)
        enc_in = torch.cat([descriptors, layer_emb, depth_emb], dim=-1)
        player_h = self.layer_encoder(enc_in, generators=self.dropout_generators)
        mean_ctx = player_h.mean(dim=1)
        max_ctx = player_h.max(dim=1).values
        g = torch.cat([mean_ctx, max_ctx], dim=-1)
        ctx_in = torch.cat([player_h, g.unsqueeze(1).expand(-1, n_players, -1)], dim=-1)
        features = self.context_encoder(ctx_in, generators=self.dropout_generators)

        gt_logits = self.gt_deployment_head(features).squeeze(-1)
        t_logits = self.teacher_allocation_head(features).squeeze(-1)
        return B2DLCMV2Outputs(
            gt_deployment_logits=gt_logits,
            gt_deployment_weights=F.softmax(gt_logits, dim=-1),
            teacher_allocation_logits=t_logits,
            teacher_allocation_weights=F.softmax(t_logits, dim=-1),
            gt_signed=self.gt_signed_head(features).squeeze(-1),
            teacher_signed=self.teacher_signed_head(features).squeeze(-1),
            player_features=features,
        )

    def forward_deployment(
        self,
        descriptors: torch.Tensor,
        *,
        prediction_depth: int,
        player_layer_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward_training(
            descriptors,
            prediction_depth=prediction_depth,
            player_layer_ids=player_layer_ids,
        )
        return out.gt_deployment_logits, out.gt_deployment_weights


class B2DLCMV2DeploymentTrunk(nn.Module):
    """Deployment-only trunk: shared trunk + GT deployment head."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        initialize: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        do_init = (seed is not None) if initialize is None else bool(initialize)
        full = B2DLCMV2(seed=seed, initialize=do_init, **kwargs)
        self.candidate_layers = full.candidate_layers
        self.prediction_depths = full.prediction_depths
        self.descriptor_dimension = full.descriptor_dimension
        self.layer_embedding_dimension = full.layer_embedding_dimension
        self.depth_embedding_dimension = full.depth_embedding_dimension
        self.hidden_dimension = full.hidden_dimension
        self.dropout_probability = full.dropout_probability
        self.layer_id_to_index = full.layer_id_to_index
        self.depth_id_to_index = full.depth_id_to_index
        self.layer_embedding = full.layer_embedding
        self.depth_embedding = full.depth_embedding
        self.layer_encoder = full.layer_encoder
        self.context_encoder = full.context_encoder
        self.gt_deployment_head = full.gt_deployment_head
        self.dropout_site_seeds = {
            site: v1.derive_u63_seed(model_seed=0, component=site) for site in v1.DROPOUT_SITE_NAMES
        }
        self.dropout_generators: dict[str, torch.Generator] = {}
        self._reset_dropout_generators(device="cpu")

    def _reset_dropout_generators(self, *, device: str | torch.device) -> None:
        device_t = torch.device(device)
        self.dropout_generators = {}
        for site, site_seed in self.dropout_site_seeds.items():
            gen = torch.Generator(device=device_t)
            gen.manual_seed(int(site_seed))
            self.dropout_generators[site] = gen

    def players_for_depth(self, depth: int) -> tuple[int, ...]:
        return v1.players_for_depth(self.candidate_layers, depth)

    def forward(
        self,
        descriptors: torch.Tensor,
        *,
        prediction_depth: int,
        player_layer_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.players_for_depth(prediction_depth)
        if player_layer_ids is None:
            player_layer_ids = expected
        v1.validate_player_layer_ids(
            prediction_depth,
            player_layer_ids,
            candidate_layers=self.candidate_layers,
        )
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.descriptor_dimension:
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "bad descriptor shape")
        if not bool(torch.isfinite(descriptors).all()):
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "descriptors must be finite")
        batch, n_players, _ = descriptors.shape
        if n_players != len(expected):
            _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "player count mismatch")
        layer_indices = torch.tensor(
            [self.layer_id_to_index[layer] for layer in expected],
            dtype=torch.long,
            device=descriptors.device,
        )
        depth_index = torch.tensor(
            self.depth_id_to_index[prediction_depth],
            dtype=torch.long,
            device=descriptors.device,
        )
        layer_emb = self.layer_embedding(layer_indices).unsqueeze(0).expand(batch, -1, -1)
        depth_emb = self.depth_embedding(depth_index).view(1, 1, -1).expand(batch, n_players, -1)
        enc_in = torch.cat([descriptors, layer_emb, depth_emb], dim=-1)
        if not self.dropout_generators:
            self._reset_dropout_generators(device=descriptors.device)
        player_h = self.layer_encoder(enc_in, generators=self.dropout_generators)
        mean_ctx = player_h.mean(dim=1)
        max_ctx = player_h.max(dim=1).values
        g = torch.cat([mean_ctx, max_ctx], dim=-1)
        ctx_in = torch.cat([player_h, g.unsqueeze(1).expand(-1, n_players, -1)], dim=-1)
        features = self.context_encoder(ctx_in, generators=self.dropout_generators)
        logits = self.gt_deployment_head(features).squeeze(-1)
        return logits, F.softmax(logits, dim=-1)


AUXILIARY_HEAD_PREFIXES = (
    "teacher_allocation_head",
    "gt_signed_head",
    "teacher_signed_head",
)


def extract_deployment_state_dict(model: B2DLCMV2) -> dict[str, torch.Tensor]:
    """Extract deployment-only parameters (no auxiliary heads)."""

    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if any(name.startswith(prefix) for prefix in AUXILIARY_HEAD_PREFIXES):
            continue
        state[name] = tensor.detach().cpu().clone()
    return state


def total_dlcm_v2_loss(
    depth_batch: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    teacher_alloc_weight: float = TEACHER_ALLOC_WEIGHT,
    gt_signed_weight: float = GT_SIGNED_WEIGHT,
    teacher_signed_weight: float = TEACHER_SIGNED_WEIGHT,
    ranking_weight: float = 0.25,
    huber_delta: float = 1.0,
    tie_tolerance: float = 1e-6,
    depth_weights: Mapping[int, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if depth_weights is None:
        depth_weights = {12: 1 / 3, 18: 1 / 3, 24: 1 / 3}
    depths = sorted(depth_batch)
    if not depths:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "no depths provided")
    total: torch.Tensor | None = None
    depth_parts: dict[int, Any] = {}
    for depth in depths:
        payload = depth_batch[depth]
        gt_deploy = v1.allocation_kl(payload["p_gt"], payload["gt_deployment_logits"])
        t_alloc = v1.allocation_kl(payload["p_t"], payload["teacher_allocation_logits"])
        s_gt, gt_parts = v1.signed_loss(
            payload["gt_signed"],
            payload["phi_gt"],
            huber_delta=huber_delta,
            ranking_weight=ranking_weight,
            tie_tolerance=tie_tolerance,
        )
        s_t, t_parts = v1.signed_loss(
            payload["teacher_signed"],
            payload["phi_t"],
            huber_delta=huber_delta,
            ranking_weight=ranking_weight,
            tie_tolerance=tie_tolerance,
        )
        depth_loss = (
            gt_deploy
            + float(gt_signed_weight) * s_gt
            + float(teacher_alloc_weight) * t_alloc
            + float(teacher_signed_weight) * s_t
        )
        weight = float(depth_weights[depth])
        total = depth_loss * weight if total is None else total + depth_loss * weight
        depth_parts[depth] = {
            "loss": depth_loss,
            "gt_deploy_kl": gt_deploy,
            "teacher_alloc_kl": t_alloc,
            "gt_signed": gt_parts,
            "teacher_signed": t_parts,
            "weight": weight,
        }
    assert total is not None
    return total, {"depths": depth_parts, "depth_weights": dict(depth_weights)}


def model_state_scientific_sha256(model: nn.Module) -> str:
    return v1.model_state_scientific_sha256(model)


def move_model_to_device_and_verify(model: B2DLCMV2, device: torch.device) -> B2DLCMV2:
    """Move synchronously to CUDA and require bit-exact CPU round-trip identity."""

    if device.type != "cuda":
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "canonical move verification requires CUDA")
    before = model_state_scientific_sha256(model)
    model.to(device)
    torch.cuda.synchronize(device)
    cpu_clone = B2DLCMV2(
        seed=None,
        candidate_layers=model.candidate_layers,
        prediction_depths=model.prediction_depths,
        descriptor_dimension=model.descriptor_dimension,
        layer_embedding_dimension=model.layer_embedding_dimension,
        depth_embedding_dimension=model.depth_embedding_dimension,
        hidden_dimension=model.hidden_dimension,
        dropout_probability=model.dropout_probability,
        initialize=False,
    )
    cpu_clone.load_state_dict(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        strict=True,
    )
    cpu_clone.component_seeds = dict(model.component_seeds)
    cpu_clone.dropout_site_seeds = dict(model.dropout_site_seeds)
    cpu_clone._reset_dropout_generators(device="cpu")
    after = model_state_scientific_sha256(cpu_clone)
    if after != before:
        _fail("B2_DLCM_V2_CONTRACT_MISMATCH", "CPU→GPU parameter identity mismatch")
    model._reset_dropout_generators(device=device)
    return model


def probe_gradient_isolation(
    model: B2DLCMV2,
    *,
    prediction_depth: int = 12,
) -> dict[str, dict[str, bool]]:
    """Return whether each loss term produces nonzero grads on each head group.

    Allocation heads are contractually zero-initialized, which blocks trunk grads
    through those Linear maps. For isolation probing we apply a temporary
    nonzero scale to allocation head weights so trunk connectivity is observable
    while still verifying cross-head isolation.
    """

    n = len(model.players_for_depth(prediction_depth))
    x = torch.randn(2, n, model.descriptor_dimension, dtype=torch.float32)
    p_gt = torch.softmax(torch.randn(2, n), dim=-1).detach()
    p_t = torch.softmax(torch.randn(2, n), dim=-1).detach()
    phi_gt = torch.randn(2, n, dtype=torch.float32).detach()
    phi_t = torch.randn(2, n, dtype=torch.float32).detach()

    # Preserve contractual zero-init while probing connectivity.
    saved = {
        "gt_w": model.gt_deployment_head.weight.detach().clone(),
        "gt_b": model.gt_deployment_head.bias.detach().clone(),
        "t_w": model.teacher_allocation_head.weight.detach().clone(),
        "t_b": model.teacher_allocation_head.bias.detach().clone(),
    }
    with torch.no_grad():
        model.gt_deployment_head.weight.add_(0.05)
        model.teacher_allocation_head.weight.add_(0.05)

    groups = {
        "gt_deployment_head": list(model.gt_deployment_head.parameters()),
        "teacher_allocation_head": list(model.teacher_allocation_head.parameters()),
        "gt_signed_head": list(model.gt_signed_head.parameters()),
        "teacher_signed_head": list(model.teacher_signed_head.parameters()),
        "shared_trunk": (
            list(model.layer_embedding.parameters())
            + list(model.depth_embedding.parameters())
            + list(model.layer_encoder.parameters())
            + list(model.context_encoder.parameters())
        ),
    }

    def _nonzero(params: list[nn.Parameter]) -> bool:
        return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in params)

    results: dict[str, dict[str, bool]] = {}
    loss_builders = {
        "gt_deploy": lambda out: v1.allocation_kl(p_gt, out.gt_deployment_logits),
        "teacher_alloc": lambda out: v1.allocation_kl(p_t, out.teacher_allocation_logits),
        "gt_signed": lambda out: v1.signed_loss(out.gt_signed, phi_gt)[0],
        "teacher_signed": lambda out: v1.signed_loss(out.teacher_signed, phi_t)[0],
    }
    try:
        for loss_name, builder in loss_builders.items():
            model.zero_grad(set_to_none=True)
            out = model.forward_training(x, prediction_depth=prediction_depth)
            loss = builder(out)
            loss.backward()
            results[loss_name] = {group: _nonzero(params) for group, params in groups.items()}
    finally:
        with torch.no_grad():
            model.gt_deployment_head.weight.copy_(saved["gt_w"])
            model.gt_deployment_head.bias.copy_(saved["gt_b"])
            model.teacher_allocation_head.weight.copy_(saved["t_w"])
            model.teacher_allocation_head.bias.copy_(saved["t_b"])
    return results
