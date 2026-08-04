"""B2-05A DLCM architecture, deterministic init, losses, and fusion contract.

Contract-only production domain module. Candidate layers and prediction depths
are configuration-driven defaults matching the frozen B2 architecture; callers
may pass explicit vocabularies rather than hard-coding four layers inside
reusable helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CANDIDATE_LAYERS: tuple[int, ...] = (6, 12, 18, 24)
DEFAULT_PREDICTION_DEPTHS: tuple[int, ...] = (12, 18, 24)
DEFAULT_DESCRIPTOR_DIMENSION = 18
DEFAULT_LAYER_EMBEDDING_DIM = 8
DEFAULT_DEPTH_EMBEDDING_DIM = 8
DEFAULT_HIDDEN_DIMENSION = 64
DEFAULT_DROPOUT_P = 0.1
SEED_DERIVATION_SCHEMA_VERSION = "b2_dlcm_seed_derivation_v1"
ARCHITECTURE_CONTRACT_VERSION = "b2_dlcm_architecture_v1"
MODEL_CLASS_ID = "rad.phase_b.b2_dlcm.B2DLCM"

DROPOUT_SITE_NAMES: tuple[str, ...] = (
    "layer_encoder.block_1.dropout",
    "layer_encoder.block_2.dropout",
    "context_encoder.block_1.dropout",
    "context_encoder.block_2.dropout",
)

COMPONENT_SEED_NAMES: tuple[str, ...] = (
    "model_initialization",
    "sampler",
    "dropout",
    "dataloader",
)


class B2DLCMError(RuntimeError):
    """Fail-closed B2 DLCM contract error with a stable code prefix."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMError(code, detail)


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_u63_seed(*, model_seed: int, component: str, schema_version: str = SEED_DERIVATION_SCHEMA_VERSION) -> int:
    """Versioned SHA-256 → big-endian 63-bit seed derivation."""

    if not isinstance(model_seed, int) or isinstance(model_seed, bool):
        _fail("B2_DLCM_SEED_INVALID", "model_seed must be an int")
    if not isinstance(component, str) or not component:
        _fail("B2_DLCM_SEED_INVALID", "component name required")
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema_version": schema_version,
                "model_seed": model_seed,
                "component": component,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value & ((1 << 63) - 1)


def derive_component_seeds(
    model_seed: int,
    components: Sequence[str] = COMPONENT_SEED_NAMES,
    *,
    collision_force: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Derive independent component seeds; fail on collisions."""

    seeds: dict[str, int] = {}
    seen: dict[int, str] = {}
    for name in components:
        if collision_force is not None and name in collision_force:
            seed = int(collision_force[name])
        else:
            seed = derive_u63_seed(model_seed=model_seed, component=name)
        if seed in seen:
            _fail(
                "B2_DLCM_SEED_COLLISION",
                f"components {seen[seed]!r} and {name!r} derived identical seed",
            )
        seen[seed] = name
        seeds[name] = seed
    return seeds


def derive_dropout_site_seeds(dropout_component_seed: int) -> dict[str, int]:
    """Derive four independent dropout site seeds from the dropout component seed."""

    seeds: dict[str, int] = {}
    seen: dict[int, str] = {}
    for site in DROPOUT_SITE_NAMES:
        seed = derive_u63_seed(model_seed=dropout_component_seed, component=site)
        if seed in seen:
            _fail(
                "B2_DLCM_SEED_COLLISION",
                f"dropout sites {seen[seed]!r} and {site!r} collided",
            )
        seen[seed] = site
        seeds[site] = seed
    return seeds


def players_for_depth(
    candidate_layers: Sequence[int],
    depth: int,
) -> tuple[int, ...]:
    """Return causal players for a prediction depth from candidate layers."""

    layers = tuple(int(layer) for layer in candidate_layers)
    if depth not in layers:
        _fail("B2_DLCM_DEPTH_INVALID", f"depth {depth} not in candidate layers {layers}")
    return tuple(layer for layer in layers if layer <= depth)


def validate_player_layer_ids(
    prediction_depth: int,
    player_layer_ids: Sequence[int],
    *,
    candidate_layers: Sequence[int] = DEFAULT_CANDIDATE_LAYERS,
) -> tuple[int, ...]:
    expected = players_for_depth(candidate_layers, prediction_depth)
    got = tuple(int(layer) for layer in player_layer_ids)
    if got != expected:
        _fail(
            "B2_DLCM_PLAYER_VOCABULARY_MISMATCH",
            f"depth {prediction_depth} expects {expected}, got {got}",
        )
    return got


class DeterministicDropout(nn.Module):
    """Generator-aware dropout; never consumes the default RNG."""

    def __init__(self, p: float = DEFAULT_DROPOUT_P) -> None:
        super().__init__()
        if not 0.0 <= float(p) < 1.0:
            _fail("B2_DLCM_DROPOUT_INVALID", "dropout p must be in [0, 1)")
        self.p = float(p)

    def forward(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        training: bool | None = None,
    ) -> torch.Tensor:
        is_training = self.training if training is None else bool(training)
        if not is_training or self.p == 0.0:
            return x
        if generator is None:
            _fail("B2_DLCM_DROPOUT_GENERATOR_REQUIRED", "explicit generator required")
        keep = 1.0 - self.p
        # Bernoulli(keep) mask on the same device as x.
        mask = torch.empty_like(x).bernoulli_(keep, generator=generator)
        return x * mask / keep


class _EncoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_p: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = DeterministicDropout(p=dropout_p)

    def forward(self, x: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        return self.dropout(self.act(self.norm(self.linear(x))), generator=generator)


class LayerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = DEFAULT_HIDDEN_DIMENSION,
        dropout_p: float = DEFAULT_DROPOUT_P,
    ) -> None:
        super().__init__()
        self.block_1 = _EncoderBlock(input_dim, hidden_dim, dropout_p)
        self.block_2 = _EncoderBlock(hidden_dim, hidden_dim, dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        *,
        generators: Mapping[str, torch.Generator],
    ) -> torch.Tensor:
        h = self.block_1(x, generator=generators["layer_encoder.block_1.dropout"])
        return self.block_2(h, generator=generators["layer_encoder.block_2.dropout"])


class ContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = DEFAULT_HIDDEN_DIMENSION,
        dropout_p: float = DEFAULT_DROPOUT_P,
    ) -> None:
        super().__init__()
        self.block_1 = _EncoderBlock(input_dim, hidden_dim, dropout_p)
        self.block_2 = _EncoderBlock(hidden_dim, hidden_dim, dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        *,
        generators: Mapping[str, torch.Generator],
    ) -> torch.Tensor:
        h = self.block_1(x, generator=generators["context_encoder.block_1.dropout"])
        return self.block_2(h, generator=generators["context_encoder.block_2.dropout"])


@dataclass(frozen=True)
class B2DLCMOutputs:
    deployment_logits: torch.Tensor
    deployment_weights: torch.Tensor
    gt_signed: torch.Tensor
    teacher_signed: torch.Tensor
    player_features: torch.Tensor


def _xavier_uniform_(tensor: torch.Tensor, generator: torch.Generator) -> None:
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)  # noqa: SLF001
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-a, a, generator=generator)


def _zeros_(tensor: torch.Tensor) -> None:
    with torch.no_grad():
        tensor.zero_()


def _ones_(tensor: torch.Tensor) -> None:
    with torch.no_grad():
        tensor.fill_(1.0)


class B2DLCM(nn.Module):
    """Frozen B2 DLCM trunk with deployment and dual signed heads."""

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
            _fail("B2_DLCM_LAYER_VOCAB_EMPTY", "candidate_layers must be non-empty")
        if sorted(self.candidate_layers) != list(self.candidate_layers):
            _fail("B2_DLCM_LAYER_VOCAB_ORDER", "candidate_layers must be ascending")
        for depth in self.prediction_depths:
            if depth not in self.candidate_layers:
                _fail("B2_DLCM_DEPTH_INVALID", f"prediction depth {depth} missing from layers")
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
        encoder_in = self.descriptor_dimension + self.layer_embedding_dimension + self.depth_embedding_dimension
        self.layer_encoder = LayerEncoder(
            input_dim=encoder_in,
            hidden_dim=self.hidden_dimension,
            dropout_p=self.dropout_probability,
        )
        self.context_encoder = ContextEncoder(
            input_dim=self.hidden_dimension * 3,
            hidden_dim=self.hidden_dimension,
            dropout_p=self.dropout_probability,
        )
        self.deployment_head = nn.Linear(self.hidden_dimension, 1)
        self.gt_signed_head = nn.Linear(self.hidden_dimension, 1)
        self.teacher_signed_head = nn.Linear(self.hidden_dimension, 1)

        self.component_seeds: dict[str, int] = {}
        self.dropout_site_seeds: dict[str, int] = {}
        self.dropout_generators: dict[str, torch.Generator] = {}
        self._init_generator: torch.Generator | None = None

        if initialize:
            if seed is None:
                _fail("B2_DLCM_SEED_REQUIRED", "seed required for initialization")
            self._initialize_from_seed(int(seed))

    def players_for_depth(self, depth: int) -> tuple[int, ...]:
        return players_for_depth(self.candidate_layers, depth)

    def _initialize_from_seed(self, seed: int) -> None:
        # All initialization on CPU with a dedicated generator.
        self.component_seeds = derive_component_seeds(seed)
        self.dropout_site_seeds = derive_dropout_site_seeds(self.component_seeds["dropout"])
        init_seed = self.component_seeds["model_initialization"]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(init_seed)
        self._init_generator = generator

        # Explicit frozen order independent of module registration traversal.
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
            ("deployment_head", self.deployment_head),
        ]
        for name, module in ordered_steps:
            if isinstance(module, nn.Embedding):
                with torch.no_grad():
                    module.weight.normal_(0.0, 0.02, generator=generator)
            elif isinstance(module, nn.Linear):
                if name == "deployment_head":
                    _zeros_(module.weight)
                    _zeros_(module.bias)
                else:
                    _xavier_uniform_(module.weight, generator)
                    _zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                _ones_(module.weight)
                _zeros_(module.bias)
            else:
                _fail("B2_DLCM_INIT_UNKNOWN_MODULE", f"unknown init target {name}")

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
    ) -> B2DLCMOutputs:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.descriptor_dimension:
            _fail(
                "B2_DLCM_DESCRIPTOR_SHAPE_INVALID",
                f"expected [B,n,{self.descriptor_dimension}], got {tuple(descriptors.shape)}",
            )
        if not bool(torch.isfinite(descriptors).all()):
            _fail("B2_DLCM_INVALID_DESCRIPTOR", "descriptors must be finite")
        if descriptors.dtype != torch.float32:
            _fail("B2_DLCM_DTYPE_INVALID", "descriptors must be float32")

        expected = self.players_for_depth(prediction_depth)
        if player_layer_ids is None:
            player_layer_ids = expected
        else:
            got = tuple(int(layer) for layer in player_layer_ids)
            if got != expected:
                # Distinguish mixed-depth style mismatches when count matches another depth.
                if len(got) == len(expected) and got != expected:
                    _fail(
                        "B2_DLCM_MIXED_DEPTH_BATCH",
                        f"player ids {got} do not match depth {prediction_depth}",
                    )
                _fail(
                    "B2_DLCM_PLAYER_VOCABULARY_MISMATCH",
                    f"depth {prediction_depth} expects {expected}, got {got}",
                )
            player_layer_ids = got

        batch, n_players, _ = descriptors.shape
        if n_players != len(expected):
            _fail(
                "B2_DLCM_PLAYER_VOCABULARY_MISMATCH",
                f"depth {prediction_depth} expects {len(expected)} players, got {n_players}",
            )

        if prediction_depth not in self.depth_id_to_index:
            _fail("B2_DLCM_DEPTH_INVALID", f"unsupported prediction depth {prediction_depth}")

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

        # Mean/Max only along player dimension — never across batch.
        mean_ctx = player_h.mean(dim=1)
        max_ctx = player_h.max(dim=1).values
        g = torch.cat([mean_ctx, max_ctx], dim=-1)  # [B, 128]
        g_expand = g.unsqueeze(1).expand(-1, n_players, -1)
        ctx_in = torch.cat([player_h, g_expand], dim=-1)  # [B, n, 192]
        features = self.context_encoder(ctx_in, generators=self.dropout_generators)

        deploy_logits = self.deployment_head(features).squeeze(-1)
        weights = F.softmax(deploy_logits, dim=-1)
        gt_signed = self.gt_signed_head(features).squeeze(-1)
        teacher_signed = self.teacher_signed_head(features).squeeze(-1)
        return B2DLCMOutputs(
            deployment_logits=deploy_logits,
            deployment_weights=weights,
            gt_signed=gt_signed,
            teacher_signed=teacher_signed,
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
        return out.deployment_logits, out.deployment_weights


class B2DLCMDeploymentTrunk(nn.Module):
    """Deployment-only trunk: no auxiliary signed heads."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        initialize: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Build full model without signed heads for state compatibility.
        do_init = (seed is not None) if initialize is None else bool(initialize)
        full = B2DLCM(seed=seed, initialize=do_init, **kwargs)
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
        self.deployment_head = full.deployment_head
        self.dropout_site_seeds = {
            site: derive_u63_seed(model_seed=0, component=site) for site in DROPOUT_SITE_NAMES
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
        return players_for_depth(self.candidate_layers, depth)

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
        validate_player_layer_ids(
            prediction_depth,
            player_layer_ids,
            candidate_layers=self.candidate_layers,
        )
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.descriptor_dimension:
            _fail("B2_DLCM_DESCRIPTOR_SHAPE_INVALID", "bad descriptor shape")
        if not bool(torch.isfinite(descriptors).all()):
            _fail("B2_DLCM_INVALID_DESCRIPTOR", "descriptors must be finite")
        batch, n_players, _ = descriptors.shape
        if n_players != len(expected):
            _fail("B2_DLCM_PLAYER_VOCABULARY_MISMATCH", "player count mismatch")
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
        # Ensure generators exist (eval path still requires objects).
        if not self.dropout_generators:
            self._reset_dropout_generators(device=descriptors.device)
        player_h = self.layer_encoder(enc_in, generators=self.dropout_generators)
        mean_ctx = player_h.mean(dim=1)
        max_ctx = player_h.max(dim=1).values
        g = torch.cat([mean_ctx, max_ctx], dim=-1)
        ctx_in = torch.cat([player_h, g.unsqueeze(1).expand(-1, n_players, -1)], dim=-1)
        features = self.context_encoder(ctx_in, generators=self.dropout_generators)
        logits = self.deployment_head(features).squeeze(-1)
        weights = F.softmax(logits, dim=-1)
        return logits, weights


def extract_deployment_state_dict(model: B2DLCM) -> dict[str, torch.Tensor]:
    """Extract deployment-only parameters (no auxiliary heads)."""

    banned = ("gt_signed_head", "teacher_signed_head")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if any(name.startswith(prefix) for prefix in banned):
            continue
        state[name] = tensor.detach().cpu().clone()
    return state


def tensor_sha256(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor):
        _fail("B2_DLCM_TENSOR_INVALID", "expected torch.Tensor")
    cpu = tensor.detach().contiguous().cpu()
    payload = (
        str(cpu.dtype).encode("utf-8")
        + b"|"
        + ",".join(str(int(s)) for s in cpu.shape).encode("utf-8")
        + b"|"
        + cpu.numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def model_state_scientific_payload(model: nn.Module) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        entries.append(
            {
                "kind": "parameter",
                "name": name,
                "dtype": str(param.dtype).replace("torch.", ""),
                "shape": [int(s) for s in param.shape],
                "tensor_sha256": tensor_sha256(param),
            }
        )
    for name, buf in sorted(model.named_buffers(), key=lambda item: item[0]):
        # persistent buffers only (named_buffers excludes non-persistent)
        entries.append(
            {
                "kind": "buffer",
                "name": name,
                "dtype": str(buf.dtype).replace("torch.", ""),
                "shape": [int(s) for s in buf.shape],
                "tensor_sha256": tensor_sha256(buf),
            }
        )
    return {"schema_version": "b2_dlcm_model_state_v1", "entries": entries}


def model_state_scientific_sha256_from_payload(payload: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(payload)


def model_state_scientific_sha256(model: nn.Module) -> str:
    return model_state_scientific_sha256_from_payload(model_state_scientific_payload(model))


def move_model_to_device_and_verify(model: B2DLCM, device: torch.device) -> B2DLCM:
    """Move synchronously to device and require bit-exact CPU round-trip identity."""

    if device.type != "cuda":
        _fail("B2_DLCM_DEVICE_INVALID", "canonical move verification requires CUDA")
    before = model_state_scientific_sha256(model)
    model.to(device)
    # Sync and copy back.
    torch.cuda.synchronize(device)
    cpu_clone = B2DLCM(
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
        _fail("B2_DLCM_INIT_DEVICE_DRIFT", "CPU→GPU parameter identity mismatch")
    # Refresh generators on the live device for subsequent training.
    model._reset_dropout_generators(device=device)
    return model


def reference_uniform_weights(player_count: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    logits = torch.zeros(player_count, dtype=dtype)
    return F.softmax(logits, dim=0)


def allocation_kl(p: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Target-weighted KL averaged equally across the batch.

    ``D_KL(p || w)`` with ``log w = log_softmax(logits)``. Zero targets contribute
    exactly zero. No epsilon, smoothing, or renormalization.
    """

    if p.shape != logits.shape:
        _fail("B2_DLCM_LOSS_SHAPE_MISMATCH", "p and logits shapes must match")
    if p.requires_grad:
        _fail("B2_DLCM_TARGET_REQUIRES_GRAD", "allocation targets must not require grad")
    log_w = F.log_softmax(logits, dim=-1)
    # Explicit zero handling: only p>0 contributes.
    positive = p > 0
    # p log p - p log w  for positive entries; 0 * log 0 := 0
    kl_terms = torch.zeros_like(p)
    kl_terms = torch.where(
        positive,
        p * (torch.log(p.clamp_min(0.0)) - log_w),
        torch.zeros_like(p),
    )
    # For p>0, log(p) is fine; for safety rewrite using where on log argument.
    log_p = torch.zeros_like(p)
    log_p = torch.where(positive, torch.log(torch.where(positive, p, torch.ones_like(p))), log_p)
    kl_terms = torch.where(positive, p * (log_p - log_w), torch.zeros_like(p))
    per_sample = kl_terms.sum(dim=-1)
    return per_sample.mean()


def allocation_loss(
    logits: torch.Tensor,
    p_gt: torch.Tensor,
    p_teacher: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    gt_kl = allocation_kl(p_gt, logits)
    teacher_kl = allocation_kl(p_teacher, logits)
    loss = 0.5 * gt_kl + 0.5 * teacher_kl
    return loss, {"gt_kl": gt_kl, "teacher_kl": teacher_kl}


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    if pred.shape != target.shape:
        _fail("B2_DLCM_LOSS_SHAPE_MISMATCH", "pred/target shape mismatch")
    if target.requires_grad:
        _fail("B2_DLCM_TARGET_REQUIRES_GRAD", "signed targets must not require grad")
    err = pred - target
    abs_err = err.abs()
    delta_t = torch.tensor(delta, dtype=pred.dtype, device=pred.device)
    quadratic = 0.5 * err * err
    linear = abs_err - 0.5 * delta_t
    per_player = torch.where(abs_err <= delta_t, quadratic, linear)
    per_sample = per_player.mean(dim=-1)
    return per_sample.mean()


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    tie_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, int]]:
    if pred.shape != target.shape:
        _fail("B2_DLCM_LOSS_SHAPE_MISMATCH", "pred/target shape mismatch")
    if target.requires_grad:
        _fail("B2_DLCM_TARGET_REQUIRES_GRAD", "signed targets must not require grad")
    batch, n = pred.shape
    total = pred.new_zeros(())
    valid_pairs = 0
    valid_samples = 0
    for b in range(batch):
        sample_loss = pred.new_zeros(())
        sample_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = target[b, i] - target[b, j]
                if abs(float(diff)) <= tie_tolerance:
                    continue
                sign = torch.sign(diff)
                pair = F.softplus(-(sign * (pred[b, i] - pred[b, j])))
                sample_loss = sample_loss + pair
                sample_pairs += 1
        if sample_pairs > 0:
            total = total + sample_loss / float(sample_pairs)
            valid_samples += 1
            valid_pairs += sample_pairs
    if valid_samples == 0:
        return pred.new_zeros(()), {"valid_pair_count": 0, "valid_sample_count": 0}
    return total / float(valid_samples), {
        "valid_pair_count": valid_pairs,
        "valid_sample_count": valid_samples,
    }


def signed_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    huber_delta: float = 1.0,
    ranking_weight: float = 0.25,
    tie_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    huber = huber_loss(pred, target, delta=huber_delta)
    ranking, _meta = pairwise_ranking_loss(pred, target, tie_tolerance=tie_tolerance)
    loss = huber + float(ranking_weight) * ranking
    return loss, {"huber": huber, "ranking": ranking}


def total_dlcm_loss(
    depth_batch: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    signed_loss_weight: float = 0.25,
    ranking_weight: float = 0.25,
    huber_delta: float = 1.0,
    tie_tolerance: float = 1e-6,
    depth_weights: Mapping[int, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if depth_weights is None:
        depth_weights = {12: 1 / 3, 18: 1 / 3, 24: 1 / 3}
    depths = sorted(depth_batch)
    if not depths:
        _fail("B2_DLCM_LOSS_EMPTY", "no depths provided")
    total = None
    depth_parts: dict[int, Any] = {}
    for depth in depths:
        payload = depth_batch[depth]
        alloc, alloc_parts = allocation_loss(
            payload["deployment_logits"],
            payload["p_gt"],
            payload["p_t"],
        )
        s_gt, gt_parts = signed_loss(
            payload["gt_signed"],
            payload["phi_gt"],
            huber_delta=huber_delta,
            ranking_weight=ranking_weight,
            tie_tolerance=tie_tolerance,
        )
        s_t, t_parts = signed_loss(
            payload["teacher_signed"],
            payload["phi_t"],
            huber_delta=huber_delta,
            ranking_weight=ranking_weight,
            tie_tolerance=tie_tolerance,
        )
        depth_loss = alloc + float(signed_loss_weight) * 0.5 * (s_gt + s_t)
        weight = float(depth_weights[depth])
        total = depth_loss * weight if total is None else total + depth_loss * weight
        depth_parts[depth] = {
            "loss": depth_loss,
            "allocation": alloc_parts,
            "gt_signed": gt_parts,
            "teacher_signed": t_parts,
            "weight": weight,
        }
    assert total is not None
    return total, {"depths": depth_parts, "depth_weights": dict(depth_weights)}


def sum_preserving_fusion(
    anomaly_maps: torch.Tensor,
    weights: torch.Tensor,
    *,
    prediction_depth: int,
    player_layer_ids: Sequence[int],
    candidate_layers: Sequence[int] = DEFAULT_CANDIDATE_LAYERS,
    return_path: bool = False,
    clamp_negative_eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, str]:
    """B2 formal sum-preserving fusion with exact uniform fast path.

    Reuses the production sum-preserving formula ``A_d = n_d * sum_i w_i A_i``
    with fixed layer-ID order FP32 accumulation. Does not modify
    ``rad.models.dlcm.sum_preserving_fusion``.
    """

    validate_player_layer_ids(
        prediction_depth,
        player_layer_ids,
        candidate_layers=candidate_layers,
    )
    if anomaly_maps.ndim != 4:
        _fail("B2_DLCM_FUSION_SHAPE_INVALID", "anomaly_maps must be [B,n,H,W]")
    if weights.ndim != 2:
        _fail("B2_DLCM_FUSION_SHAPE_INVALID", "weights must be [B,n]")
    if anomaly_maps.shape[:2] != weights.shape:
        _fail("B2_DLCM_FUSION_SHAPE_INVALID", "maps/weights batch-player mismatch")
    if anomaly_maps.dtype != torch.float32 or weights.dtype != torch.float32:
        _fail("B2_DLCM_FUSION_DTYPE_INVALID", "fusion requires float32")
    if not anomaly_maps.is_contiguous() or not weights.is_contiguous():
        _fail("B2_DLCM_FUSION_CONTIGUITY", "inputs must be contiguous")
    if not bool(torch.isfinite(anomaly_maps).all()) or not bool(torch.isfinite(weights).all()):
        _fail("B2_DLCM_FUSION_NONFINITE", "fusion inputs must be finite")
    if anomaly_maps.shape[2] < 1 or anomaly_maps.shape[3] < 1:
        _fail("B2_DLCM_FUSION_SHAPE_INVALID", "H,W must be >= 1")

    n_d = anomaly_maps.shape[1]
    # Weight validity.
    if bool((weights < -clamp_negative_eps).any()):
        _fail("B2_DLCM_WEIGHT_NEGATIVE", "weights below -1e-6")
    clamped = torch.where(
        (weights >= -clamp_negative_eps) & (weights < 0),
        torch.zeros_like(weights),
        weights,
    )
    row_sum = clamped.sum(dim=-1)
    if bool((row_sum - 1.0).abs().gt(clamp_negative_eps).any()):
        _fail("B2_DLCM_WEIGHT_SUM_INVALID", "weight rows must sum to 1 within 1e-6")

    ref = reference_uniform_weights(n_d, dtype=torch.float32).to(device=weights.device)
    # Exact bit-pattern match against reference uniform vector.
    ref_row = ref.view(1, -1).expand_as(clamped)
    exact_uniform = bool(torch.equal(clamped, ref_row))
    if exact_uniform:
        # Uniform baseline: add maps directly in fixed layer order.
        fused = anomaly_maps[:, 0].clone()
        for idx in range(1, n_d):
            fused = fused + anomaly_maps[:, idx]
        path = "uniform_baseline"
    else:
        fused = torch.zeros(
            anomaly_maps.shape[0],
            anomaly_maps.shape[2],
            anomaly_maps.shape[3],
            dtype=torch.float32,
            device=anomaly_maps.device,
        )
        for idx in range(n_d):
            fused = fused + clamped[:, idx].view(-1, 1, 1) * anomaly_maps[:, idx]
        fused = fused * float(n_d)
        path = "dynamic_weighted"
    if return_path:
        return fused, path
    return fused


def float_to_bits_hex(value: float, *, dtype: str) -> dict[str, str]:
    """Exact IEEE-754 metadata for decision-critical floats."""

    if dtype == "float32":
        packed = struct.pack(">f", float(value))
        bits = packed.hex()
        if len(bits) != 8:
            _fail("B2_DLCM_FLOAT_BITS_INVALID", "float32 bits must be 8 hex chars")
    elif dtype == "float64":
        packed = struct.pack(">d", float(value))
        bits = packed.hex()
        if len(bits) != 16:
            _fail("B2_DLCM_FLOAT_BITS_INVALID", "float64 bits must be 16 hex chars")
    else:
        _fail("B2_DLCM_FLOAT_BITS_INVALID", f"unsupported dtype {dtype}")
    if not math.isfinite(float(value)):
        _fail("B2_DLCM_FLOAT_NONFINITE", "NaN/Inf are never valid trace values")
    return {"dtype": dtype, "bits_hex": bits}


def bits_hex_to_float(meta: Mapping[str, str]) -> float:
    dtype = meta["dtype"]
    bits = meta["bits_hex"]
    if dtype == "float32":
        if len(bits) != 8:
            _fail("B2_DLCM_FLOAT_BITS_INVALID", "float32 bits width")
        return struct.unpack(">f", bytes.fromhex(bits))[0]
    if dtype == "float64":
        if len(bits) != 16:
            _fail("B2_DLCM_FLOAT_BITS_INVALID", "float64 bits width")
        return struct.unpack(">d", bytes.fromhex(bits))[0]
    _fail("B2_DLCM_FLOAT_BITS_INVALID", f"unsupported dtype {dtype}")
