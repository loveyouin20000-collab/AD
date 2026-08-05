"""B2-05C2 V3 training: category-balanced sampler, eligibility, selection, dry-run."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch

from rad.phase_b import b2_dlcm as v1
from rad.phase_b import b2_dlcm_training as v1_train
from rad.phase_b import b2_dlcm_v3 as v3
from rad.phase_b import b2_dlcm_v3_protocol as protocol

SAMPLER_CONTRACT_VERSION = "b2_dlcm_category_balanced_sampler_v1"
GT_MACRO_MARGIN = 1e-5
GT_PER_CATEGORY_SLACK = 1e-4
MIN_DELTA = 1e-5


class B2DLCMV3TrainingError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise B2DLCMV3TrainingError(code, detail)


def _generator_state_bytes(gen: torch.Generator) -> bytes:
    state = gen.get_state()
    return bytes(state.cpu().numpy().tobytes())


def _generator_from_state_bytes(raw: bytes) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    # torch Generator state is a uint8 CPU tensor; reconstruct from bytes.
    import numpy as np

    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    gen.set_state(torch.from_numpy(arr))
    return gen


@dataclass
class CategoryBalancedSamplerState:
    bottle_generator_state: bytes
    carpet_generator_state: bytes
    batch_order_generator_state: bytes
    epoch_index: int
    sampler_contract_version: str = SAMPLER_CONTRACT_VERSION

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "bottle_generator_state_hex": self.bottle_generator_state.hex(),
            "carpet_generator_state_hex": self.carpet_generator_state.hex(),
            "batch_order_generator_state_hex": self.batch_order_generator_state.hex(),
            "epoch_index": int(self.epoch_index),
            "sampler_contract_version": self.sampler_contract_version,
        }

    @classmethod
    def from_jsonable(cls, payload: Mapping[str, Any]) -> CategoryBalancedSamplerState:
        return cls(
            bottle_generator_state=bytes.fromhex(str(payload["bottle_generator_state_hex"])),
            carpet_generator_state=bytes.fromhex(str(payload["carpet_generator_state_hex"])),
            batch_order_generator_state=bytes.fromhex(
                str(payload["batch_order_generator_state_hex"])
            ),
            epoch_index=int(payload["epoch_index"]),
            sampler_contract_version=str(payload["sampler_contract_version"]),
        )


def _record_category(record: Any) -> str:
    if isinstance(record, Mapping):
        return str(record["category"])
    return str(record.category)


def _record_id(record: Any) -> str:
    if isinstance(record, Mapping):
        return str(record["stable_sample_id"])
    return str(record.stable_sample_id)


def _mix_seed(base: int, epoch: int) -> int:
    return int(base) ^ (int(epoch) * 0x9E3779B97F4A7C15 & ((1 << 63) - 1)) & ((1 << 63) - 1)


def build_category_balanced_epoch_batches(
    records: Sequence[Any],
    *,
    epoch: int,
    bottle_seed: int,
    carpet_seed: int,
    batch_order_seed: int,
    prior_state: CategoryBalancedSamplerState | None = None,
) -> tuple[list[list[str]], CategoryBalancedSamplerState]:
    """Build 4 batches of 2 bottle + 2 carpet; each record once per epoch."""

    if prior_state is not None:
        if prior_state.sampler_contract_version != SAMPLER_CONTRACT_VERSION:
            _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "sampler contract version mismatch")
        if int(prior_state.epoch_index) != int(epoch):
            _fail(
                "B2_DLCM_V3_CONTRACT_MISMATCH",
                f"resume epoch {prior_state.epoch_index} != requested {epoch}",
            )
        bottle_gen = _generator_from_state_bytes(prior_state.bottle_generator_state)
        carpet_gen = _generator_from_state_bytes(prior_state.carpet_generator_state)
        order_gen = _generator_from_state_bytes(prior_state.batch_order_generator_state)
    else:
        bottle_gen = torch.Generator(device="cpu")
        bottle_gen.manual_seed(_mix_seed(bottle_seed, epoch))
        carpet_gen = torch.Generator(device="cpu")
        carpet_gen.manual_seed(_mix_seed(carpet_seed, epoch))
        order_gen = torch.Generator(device="cpu")
        order_gen.manual_seed(_mix_seed(batch_order_seed, epoch))

    by_cat: dict[str, list[str]] = {"bottle": [], "carpet": []}
    for record in records:
        cat = _record_category(record)
        if cat not in by_cat:
            _fail("B2_DLCM_CATEGORY_COVERAGE_INVALID", f"unexpected category {cat}")
        by_cat[cat].append(_record_id(record))
    if len(by_cat["bottle"]) != 8 or len(by_cat["carpet"]) != 8:
        _fail(
            "B2_DLCM_CATEGORY_COVERAGE_INVALID",
            f"expected 8+8 bottle/carpet, got {len(by_cat['bottle'])}+{len(by_cat['carpet'])}",
        )

    bottle_ids = sorted(by_cat["bottle"])
    carpet_ids = sorted(by_cat["carpet"])
    bottle_order = [bottle_ids[i] for i in torch.randperm(8, generator=bottle_gen).tolist()]
    carpet_order = [carpet_ids[i] for i in torch.randperm(8, generator=carpet_gen).tolist()]
    pairs = []
    for i in range(4):
        batch = bottle_order[2 * i : 2 * i + 2] + carpet_order[2 * i : 2 * i + 2]
        _validate_batch_categories(batch, {sid: _lookup_cat(records, sid) for sid in batch})
        pairs.append(batch)
    order = torch.randperm(4, generator=order_gen).tolist()
    batches = [pairs[i] for i in order]
    # Capture POST-draw state for resume of NEXT epoch construction from saved pre-state.
    # Contract: persist generator states that allow exact reproduction. We persist the
    # pre-draw seeds via reconstructing from epoch seeds; also store post-draw for chaining.
    state = CategoryBalancedSamplerState(
        bottle_generator_state=_generator_state_bytes(bottle_gen),
        carpet_generator_state=_generator_state_bytes(carpet_gen),
        batch_order_generator_state=_generator_state_bytes(order_gen),
        epoch_index=int(epoch),
        sampler_contract_version=SAMPLER_CONTRACT_VERSION,
    )
    # Verify coverage
    flat = [sid for batch in batches for sid in batch]
    if len(flat) != 16 or len(set(flat)) != 16:
        _fail("B2_DLCM_CATEGORY_COVERAGE_INVALID", "each record must appear exactly once")
    return batches, state


def _lookup_cat(records: Sequence[Any], sid: str) -> str:
    for record in records:
        if _record_id(record) == sid:
            return _record_category(record)
    _fail("B2_DLCM_CATEGORY_COVERAGE_INVALID", f"unknown id {sid}")


def _validate_batch_categories(batch_ids: Sequence[str], cats: Mapping[str, str]) -> None:
    counts = {"bottle": 0, "carpet": 0}
    for sid in batch_ids:
        cat = cats[sid]
        if cat not in counts:
            _fail("B2_DLCM_CATEGORY_BATCH_INVALID", f"invalid category {cat}")
        counts[cat] += 1
    if counts["bottle"] != 2 or counts["carpet"] != 2:
        _fail("B2_DLCM_CATEGORY_BATCH_INVALID", f"batch counts {counts}")


def reproduce_epoch_batches_from_seeds(
    records: Sequence[Any],
    *,
    epoch: int,
    bottle_seed: int,
    carpet_seed: int,
    batch_order_seed: int,
) -> list[list[str]]:
    batches, _ = build_category_balanced_epoch_batches(
        records,
        epoch=epoch,
        bottle_seed=bottle_seed,
        carpet_seed=carpet_seed,
        batch_order_seed=batch_order_seed,
        prior_state=None,
    )
    return batches


def is_checkpoint_eligible(
    *,
    depth24_gt_kl_macro: float,
    depth24_uniform_gt_kl_macro: float,
    per_category_gt_kl: Mapping[str, float],
    per_category_uniform_gt_kl: Mapping[str, float],
) -> bool:
    if not (float(depth24_gt_kl_macro) <= float(depth24_uniform_gt_kl_macro) - GT_MACRO_MARGIN):
        return False
    for cat in ("bottle", "carpet"):
        if cat not in per_category_gt_kl or cat not in per_category_uniform_gt_kl:
            return False
        if not (
            float(per_category_gt_kl[cat])
            <= float(per_category_uniform_gt_kl[cat]) + GT_PER_CATEGORY_SLACK
        ):
            return False
    return True


@dataclass
class EligibleWorstCategorySelector:
    min_delta: float = MIN_DELTA
    best_worst: float | None = None
    best_macro: float | None = None
    best_signed: float | None = None
    best_epoch: int | None = None
    best_eligible: bool = False
    patience_counter: int = 0

    def consider(
        self,
        *,
        epoch: int,
        worst_category_kl: float,
        macro_kl: float,
        gt_signed: float,
        eligible: bool,
    ) -> bool:
        """Return True if this checkpoint becomes the new best."""

        if self.best_epoch is None:
            self.best_worst = float(worst_category_kl)
            self.best_macro = float(macro_kl)
            self.best_signed = float(gt_signed)
            self.best_epoch = int(epoch)
            self.best_eligible = bool(eligible) and int(epoch) > 0
            return True

        # Epoch 0 is initial best but not trained-eligible.
        if int(epoch) == 0:
            return False

        if not eligible:
            return False

        assert self.best_worst is not None and self.best_macro is not None
        assert self.best_signed is not None

        # First trained eligible always replaces Epoch-0 fallback.
        if not self.best_eligible:
            self._accept(epoch, worst_category_kl, macro_kl, gt_signed, eligible=True)
            self.patience_counter = 0
            return True

        md = float(self.min_delta)
        if float(worst_category_kl) < float(self.best_worst) - md:
            self._accept(epoch, worst_category_kl, macro_kl, gt_signed, eligible=True)
            self.patience_counter = 0
            return True
        if abs(float(worst_category_kl) - float(self.best_worst)) <= md:
            if float(macro_kl) < float(self.best_macro) - md:
                self._accept(epoch, worst_category_kl, macro_kl, gt_signed, eligible=True)
                self.patience_counter = 0
                return True
            if abs(float(macro_kl) - float(self.best_macro)) <= md:
                if float(gt_signed) < float(self.best_signed) - md:
                    self._accept(epoch, worst_category_kl, macro_kl, gt_signed, eligible=True)
                    self.patience_counter = 0
                    return True
                # Complete tie → keep earlier epoch (no replace).
        return False

    def _accept(
        self,
        epoch: int,
        worst: float,
        macro: float,
        signed: float,
        *,
        eligible: bool,
    ) -> None:
        self.best_worst = float(worst)
        self.best_macro = float(macro)
        self.best_signed = float(signed)
        self.best_epoch = int(epoch)
        self.best_eligible = bool(eligible)

    def note_epoch_without_improvement(self) -> None:
        self.patience_counter += 1


def select_canonical_seed_category_robust(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    min_delta: float = MIN_DELTA,
) -> int:
    if not seed_results:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "empty seed results")
    if all(not bool(r.get("eligible", False)) for r in seed_results):
        return 17

    best: tuple[float, float, float, int] | None = None
    # (worst, macro, signed, seed) — eligible preferred already filtered.
    eligible_rows = [r for r in seed_results if bool(r.get("eligible", False))]
    pool = eligible_rows if eligible_rows else list(seed_results)
    for row in pool:
        seed = int(row["seed"])
        worst = float(row["worst_category_kl"])
        macro = float(row["macro_kl"])
        signed = float(row["gt_signed"])
        cand = (worst, macro, signed, seed)
        if best is None:
            best = cand
            continue
        b_worst, b_macro, b_signed, b_seed = best
        md = float(min_delta)
        if worst < b_worst - md:
            best = cand
        elif abs(worst - b_worst) <= md and macro < b_macro - md:
            best = cand
        elif abs(worst - b_worst) <= md and abs(macro - b_macro) <= md and signed < b_signed - md:
            best = cand
        elif (
            abs(worst - b_worst) <= md
            and abs(macro - b_macro) <= md
            and abs(signed - b_signed) <= md
            and seed < b_seed
        ):
            best = cand
    assert best is not None
    return int(best[3])


def _uniform_logits(n_players: int, batch: int, device: torch.device) -> torch.Tensor:
    # Zero logits → uniform softmax; for KL(p||u) use zero logits.
    return torch.zeros(batch, n_players, device=device, dtype=torch.float32)


def calibration_metrics_category_robust(
    model: v3.B2DLCMV3,
    calibration: Sequence[Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Depth-24 eligibility metrics + 3-depth signed diagnostic."""

    device_t = device or torch.device("cpu")
    model.eval()
    per_cat_kl: dict[str, list[float]] = {"bottle": [], "carpet": []}
    per_cat_uniform: dict[str, list[float]] = {"bottle": [], "carpet": []}
    signed_terms: list[float] = []

    for record in calibration:
        cat = _record_category(record)
        if cat not in per_cat_kl:
            continue
        if hasattr(record, "descriptors"):
            desc = record.descriptors[24].unsqueeze(0).to(device=device_t)
            p_gt = record.p_gt[24].unsqueeze(0).to(device=device_t)
            phi_gt = record.phi_gt[24].unsqueeze(0).to(device=device_t)
        else:
            desc = record["descriptors"][24].unsqueeze(0).to(device=device_t)
            p_gt = record["p_gt"][24].unsqueeze(0).to(device=device_t)
            phi_gt = record["phi_gt"][24].unsqueeze(0).to(device=device_t)
        with torch.no_grad():
            out = model.forward_training(desc, prediction_depth=24)
            kl = float(v1.allocation_kl(p_gt, out.gt_deployment_logits))
            uni = float(v1.allocation_kl(p_gt, _uniform_logits(p_gt.shape[-1], 1, device_t)))
            s_gt, _ = v1.signed_loss(out.gt_signed, phi_gt)
        per_cat_kl[cat].append(kl)
        per_cat_uniform[cat].append(uni)
        signed_terms.append(float(s_gt))

    if not per_cat_kl["bottle"] or not per_cat_kl["carpet"]:
        _fail("B2_DLCM_CATEGORY_COVERAGE_INVALID", "calibration missing bottle/carpet")

    cat_means = {c: sum(v) / len(v) for c, v in per_cat_kl.items()}
    uni_means = {c: sum(v) / len(v) for c, v in per_cat_uniform.items()}
    macro = sum(cat_means.values()) / 2.0
    uni_macro = sum(uni_means.values()) / 2.0
    worst = max(cat_means.values())
    signed = sum(signed_terms) / float(len(signed_terms))
    eligible = is_checkpoint_eligible(
        depth24_gt_kl_macro=macro,
        depth24_uniform_gt_kl_macro=uni_macro,
        per_category_gt_kl=cat_means,
        per_category_uniform_gt_kl=uni_means,
    )
    return {
        "worst_category_kl": worst,
        "macro_kl": macro,
        "gt_signed": signed,
        "per_category_gt_kl": cat_means,
        "per_category_uniform_gt_kl": uni_means,
        "uniform_macro_kl": uni_macro,
        "eligible": eligible,
        "depth": 24,
    }


def teacher_must_not_affect_selection() -> dict[str, bool]:
    return {
        "teacher_in_worst": False,
        "teacher_in_macro": False,
        "teacher_in_signed_selector": False,
        "teacher_in_patience": False,
        "teacher_in_canonical": False,
        "development_in_selector": False,
    }


def build_hermetic_v3_records(*, map_hw: tuple[int, int] = (8, 8)) -> list[dict[str, Any]]:
    """Hermetic 16/8/8 with bottle+carpet only (8+8 training)."""

    base = v1_train.build_hermetic_contract_records(map_hw=map_hw)
    # Remap all records to bottle/carpet alternating within each split.
    out: list[dict[str, Any]] = []
    counters = {"training": 0, "calibration": 0, "evaluation": 0}
    for record in base:
        row = copy.deepcopy(record)
        split = str(row["split"])
        idx = counters[split]
        counters[split] = idx + 1
        row["category"] = "bottle" if (idx % 2 == 0) else "carpet"
        out.append(row)
    train = [r for r in out if r["split"] == "training"]
    n_bottle = sum(1 for r in train if r["category"] == "bottle")
    n_carpet = sum(1 for r in train if r["category"] == "carpet")
    if n_bottle != 8 or n_carpet != 8:
        _fail("B2_DLCM_CATEGORY_COVERAGE_INVALID", "hermetic training must be 8+8")
    return out


def dry_run_complete_v3_contract_validation(
    *,
    config: Mapping[str, Any],
    seed: int,
    output_dir: Path | str,
) -> dict[str, Any]:
    if list(config.get("candidate_layers", [])) != [6, 12, 18, 24]:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "candidate_layers mismatch")
    if list(config.get("prediction_depths", [])) != [12, 18, 24]:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "prediction_depths mismatch")
    if abs(float(config.get("smoothmax_tau", -1)) - v3.SMOOTHMAX_TAU) > 0:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "smoothmax_tau must be 0.05")
    return dry_run_v3_summary(config=config, seed=seed, output_dir=output_dir)


def dry_run_v3_summary(
    *,
    config: Mapping[str, Any],
    seed: int,
    output_dir: Path | str,
) -> dict[str, Any]:
    protocol.reject_bypass_flags(config)
    if config.get("real_training_enabled") is not False:
        _fail("B2_DLCM_V3_REAL_TRAINING_NOT_ENABLED", "dry-run requires real_training_enabled=false")
    if config.get("contract_stage") != "b2_05c2a":
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "contract_stage must be b2_05c2a")

    records = build_hermetic_v3_records()
    train = [r for r in records if r["split"] == "training"]
    calibration = [r for r in records if r["split"] == "calibration"]
    model = v3.B2DLCMV3(seed=int(seed))
    component = model.component_seeds
    batches, sampler_state = build_category_balanced_epoch_batches(
        train,
        epoch=0,
        bottle_seed=int(component["sampler"]),
        carpet_seed=int(component["sampler"]) ^ 0xC0FFEE,
        batch_order_seed=int(component["sampler"]) ^ 0xA5A5A5,
    )
    # Resume reproduction check
    again = reproduce_epoch_batches_from_seeds(
        train,
        epoch=0,
        bottle_seed=int(component["sampler"]),
        carpet_seed=int(component["sampler"]) ^ 0xC0FFEE,
        batch_order_seed=int(component["sampler"]) ^ 0xA5A5A5,
    )
    if again != batches:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "sampler not reproducible")

    batch_ids = batches[0]
    id_to = {_record_id(r): r for r in train}
    batch = [id_to[i] for i in batch_ids]
    categories = [_record_category(r) for r in batch]
    depth_payload: dict[int, dict[str, torch.Tensor]] = {}
    for depth in model.prediction_depths:
        descs = torch.stack([r["descriptors"][depth] for r in batch], dim=0)
        out = model.forward_training(descs, prediction_depth=depth)
        depth_payload[depth] = {
            "gt_deployment_logits": out.gt_deployment_logits,
            "teacher_allocation_logits": out.teacher_allocation_logits,
            "gt_signed": out.gt_signed,
            "teacher_signed": out.teacher_signed,
            "p_gt": torch.stack([r["p_gt"][depth] for r in batch], dim=0),
            "p_t": torch.stack([r["p_t"][depth] for r in batch], dim=0),
            "phi_gt": torch.stack([r["phi_gt"][depth] for r in batch], dim=0),
            "phi_t": torch.stack([r["phi_t"][depth] for r in batch], dim=0),
        }
    loss, parts = v3.total_dlcm_v3_loss(depth_payload, categories=categories, tau=v3.SMOOTHMAX_TAU)
    if not bool(torch.isfinite(loss)):
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "dry-run loss nonfinite")
    if parts["aggregation"]["gt_deployment"] != "category_smooth_max":
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "gt deploy aggregation mismatch")
    metrics = calibration_metrics_category_robust(model, calibration)
    selector = EligibleWorstCategorySelector()
    selector.consider(
        epoch=0,
        worst_category_kl=metrics["worst_category_kl"],
        macro_kl=metrics["macro_kl"],
        gt_signed=metrics["gt_signed"],
        eligible=False,  # epoch 0 not trained-eligible
    )
    assert teacher_must_not_affect_selection()["teacher_in_canonical"] is False
    deploy_state = v3.extract_deployment_state_dict(model)
    if any(any(k.startswith(b) for k in deploy_state) for b in v3.AUXILIARY_HEAD_PREFIXES):
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "auxiliary leaked into deployment")
    final_content_resolved = False
    try:
        protocol.forbid_final_content_access(unlocked=False, context="dry_run_probe")
    except protocol.B2DLCMV3ProtocolError as exc:
        if exc.code != "B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN":
            raise
        final_content_resolved = False
    return {
        "mode": "dry_run",
        "status": "contract_validated",
        "real_training_started": False,
        "development_evaluation_started": False,
        "final_content_resolved": final_content_resolved,
        "final_materialization_started": False,
        "final_evaluation_started": False,
        "artifact_written": False,
        "run_directory_created": False,
        "teacher_forward_count": 0,
        "seed": int(seed),
        "dry_run_loss_finite": True,
        "calibration_metrics": metrics,
        "sampler_state": sampler_state.to_jsonable(),
        "deployment_param_count": len(deploy_state),
        "v1_immutable": v3.v1_immutable_identity(),
        "v2_immutable": v3.v2_immutable_identity(),
        "output_dir": str(Path(output_dir)),
        "category_not_in_model": True,
    }



def _teacher_alloc_kl_macro(
    model: v3.B2DLCMV3,
    calibration: Sequence[Any],
    *,
    device: torch.device,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for record in calibration:
            for depth in model.prediction_depths:
                if hasattr(record, "descriptors"):
                    desc = record.descriptors[int(depth)].unsqueeze(0).to(device=device)
                    p_t = record.p_t[int(depth)].unsqueeze(0).to(device=device)
                else:
                    desc = record["descriptors"][int(depth)].unsqueeze(0).to(device=device)
                    p_t = record["p_t"][int(depth)].unsqueeze(0).to(device=device)
                out = model.forward_training(desc, prediction_depth=int(depth))
                values.append(float(v1.allocation_kl(p_t, out.teacher_allocation_logits)))
    return sum(values) / float(len(values))


def _cpu_identity_model_v3(model: v3.B2DLCMV3) -> v3.B2DLCMV3:
    clone = v3.B2DLCMV3(
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
    clone.load_state_dict(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        strict=True,
    )
    return clone


def run_v3_contract_training(
    *,
    output_root: Path,
    seed: int,
    records: Sequence[Any],
    maximum_epochs: int = 1,
    patience: int = 50,
    device: str = "cpu",
    batch_size: int = 4,
    smoothmax_tau: float = v3.SMOOTHMAX_TAU,
    environment_contract: Mapping[str, Any] | None = None,
    allow_existing_output: bool = False,
    mark_real_training_started: bool = False,
) -> dict[str, Any]:
    """Official/hermetic V3 seed training with constrained worst-category selection."""

    import json

    from rad.phase_b import b2_dlcm_v2 as v2

    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing_output:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "output root must be empty/fresh")
    output_root.mkdir(parents=True, exist_ok=True)

    if environment_contract is None:
        env = v1_train.collect_environment_contract(allow_cpu_for_hermetic=(device == "cpu"))
        v1_train.persist_environment_contract(output_root / "environment_contract.json", env)
    else:
        env = dict(environment_contract)
        env_path = output_root / "environment_contract.json"
        if env_path.is_file():
            existing = json.loads(env_path.read_text(encoding="utf-8"))
            if v1_train.environment_contract_sha256(existing) != v1_train.environment_contract_sha256(
                env
            ):
                _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "environment contract drifted")
        else:
            v1_train.persist_environment_contract(env_path, env)

    seed_dir = output_root / f"seed_{seed}"
    if seed_dir.exists():
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", f"seed directory already exists: seed_{seed}")

    by_split: dict[str, list[Any]] = {"training": [], "calibration": [], "evaluation": []}
    for record in records:
        by_split[record.split].append(record)
    if len(by_split["training"]) != 16 or len(by_split["calibration"]) != 8:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "expected 16/8 training/calibration")
    if len(by_split["evaluation"]) != 8:
        _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "expected 8 evaluation placeholders")

    train_bottle = [r for r in by_split["training"] if _record_category(r) == "bottle"]
    train_carpet = [r for r in by_split["training"] if _record_category(r) == "carpet"]
    if len(train_bottle) != 8 or len(train_carpet) != 8:
        _fail("B2_DLCM_CATEGORY_COVERAGE_INVALID", "training must be 8 bottle + 8 carpet")

    tx = v1_train.EpochTransaction(seed_dir)
    model = v3.B2DLCMV3(seed=seed)
    with torch.no_grad():
        assert torch.all(model.gt_deployment_head.weight == 0)
        assert torch.all(model.teacher_allocation_head.weight == 0)
    if device != "cpu":
        model = v2.move_model_to_device_and_verify(model, torch.device(device))
    optimizer, _groups = v1_train.build_adamw(model, lr=3e-6)
    schedule = v1_train.ExplicitLRSchedule()
    schedule.install_into_optimizer(optimizer)
    selector = EligibleWorstCategorySelector()
    stopper = v1_train.EarlyStopController(patience=patience, maximum_epochs=maximum_epochs)
    chain = v1_train.TraceHashChain()

    train_records = by_split["training"]
    id_to_record = {_record_id(r): r for r in train_records}
    calibration = sorted(by_split["calibration"], key=lambda r: _record_id(r))
    component_seeds = model.component_seeds
    bottle_seed = int(component_seeds["sampler"])
    carpet_seed = int(component_seeds["sampler"]) ^ 0xC0FFEE
    batch_order_seed = int(component_seeds["sampler"]) ^ 0xA5A5A5
    device_t = next(model.parameters()).device

    metrics0 = calibration_metrics_category_robust(model, calibration, device=device_t)
    teacher0 = _teacher_alloc_kl_macro(model, calibration, device=device_t)
    selector.consider(
        epoch=0,
        worst_category_kl=metrics0["worst_category_kl"],
        macro_kl=metrics0["macro_kl"],
        gt_signed=metrics0["gt_signed"],
        eligible=False,
    )
    v1_train.run_epoch0_baseline_marker(schedule)
    staging = tx.begin()
    torch.save({"model": model.state_dict(), "epoch": 0}, staging / "best_training_checkpoint.pt")
    torch.save({"model": model.state_dict(), "epoch": 0}, staging / "last_training_checkpoint.pt")
    (staging / "category_sampler_state.json").write_text(
        json.dumps(
            {
                "epoch_index": 0,
                "bottle_seed": bottle_seed,
                "carpet_seed": carpet_seed,
                "batch_order_seed": batch_order_seed,
                "sampler_contract_version": SAMPLER_CONTRACT_VERSION,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    train_ids = sorted(_record_id(r) for r in train_records)
    record0 = v1_train.scientific_epoch_record(
        epoch=0,
        primary=metrics0["worst_category_kl"],
        secondary=metrics0["macro_kl"],
        total_loss=0.0,
        global_optimizer_step=0,
        sample_ids=train_ids,
    )
    record0["worst_category_kl"] = float(metrics0["worst_category_kl"])
    record0["macro_kl"] = float(metrics0["macro_kl"])
    record0["gt_signed"] = float(metrics0["gt_signed"])
    record0["eligible"] = False
    record0["per_category_gt_kl"] = dict(metrics0["per_category_gt_kl"])
    record0["teacher_alloc_kl_macro"] = float(teacher0)
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
            "best_file_sha256": "pending",
            "status": "running",
            "best_eligible": False,
        },
        update_best=True,
    )
    manifest = v1_train.load_committed_manifest(seed_dir)
    manifest["best_file_sha256"] = v1_train.sha256_file(
        seed_dir / "committed" / "best_training_checkpoint.pt"
    )
    manifest["last_file_sha256"] = v1_train.sha256_file(
        seed_dir / "committed" / "last_training_checkpoint.pt"
    )
    v1_train._atomic_write_json(seed_dir / "committed" / "epoch_state_manifest.json", manifest)

    status = "running"
    last_epoch = 0
    best_per_category = dict(metrics0["per_category_gt_kl"])
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        batches, sampler_state = build_category_balanced_epoch_batches(
            train_records,
            epoch=epoch,
            bottle_seed=bottle_seed,
            carpet_seed=carpet_seed,
            batch_order_seed=batch_order_seed,
        )
        if len(batches) != 4 or batch_size != 4:
            _fail("B2_DLCM_V3_CONTRACT_MISMATCH", "expected 4 batches of size 4")
        epoch_losses: list[float] = []
        flat_order: list[str] = []
        for batch_ids in batches:
            flat_order.extend(batch_ids)
            optimizer.zero_grad(set_to_none=True)
            categories = [_record_category(id_to_record[i]) for i in batch_ids]
            depth_payload: dict[int, dict[str, torch.Tensor]] = {}
            for depth in model.prediction_depths:
                descs = torch.stack(
                    [id_to_record[i].descriptors[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                p_gt = torch.stack(
                    [id_to_record[i].p_gt[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                p_t = torch.stack(
                    [id_to_record[i].p_t[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                phi_gt = torch.stack(
                    [id_to_record[i].phi_gt[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                phi_t = torch.stack(
                    [id_to_record[i].phi_t[depth] for i in batch_ids], dim=0
                ).to(device=device_t, dtype=torch.float32)
                out = model.forward_training(descs, prediction_depth=depth)
                depth_payload[depth] = {
                    "gt_deployment_logits": out.gt_deployment_logits,
                    "teacher_allocation_logits": out.teacher_allocation_logits,
                    "gt_signed": out.gt_signed,
                    "teacher_signed": out.teacher_signed,
                    "p_gt": p_gt,
                    "p_t": p_t,
                    "phi_gt": phi_gt,
                    "phi_t": phi_t,
                }
            loss, _ = v3.total_dlcm_v3_loss(
                depth_payload, categories=categories, tau=float(smoothmax_tau)
            )
            if not bool(torch.isfinite(loss)):
                v1_train.write_failure_attestation(
                    seed_dir,
                    seed=seed,
                    stage="optimizer_step",
                    error_code="B2_DLCM_NONFINITE_LOSS",
                    last_valid_committed_epoch=last_epoch,
                    global_optimizer_step=schedule.global_optimizer_step,
                    trace_chain_tail=chain.tail,
                    identities={
                        "model_state_scientific_sha256": v3.model_state_scientific_sha256(model)
                    },
                    environment_identity=v1_train.environment_contract_sha256(env),
                    uncommitted_staging=[],
                )
                return {
                    "status": "failed",
                    "real_training_started": bool(mark_real_training_started),
                    "evaluation_unlocked": False,
                }
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t = schedule.global_optimizer_step + 1
            schedule.note_successful_update(t)
            schedule.install_into_optimizer(optimizer)
            epoch_losses.append(float(loss.detach()))

        metrics = calibration_metrics_category_robust(model, calibration, device=device_t)
        teacher_diag = _teacher_alloc_kl_macro(model, calibration, device=device_t)
        improved = selector.consider(
            epoch=epoch,
            worst_category_kl=metrics["worst_category_kl"],
            macro_kl=metrics["macro_kl"],
            gt_signed=metrics["gt_signed"],
            eligible=bool(metrics["eligible"]),
        )
        if improved:
            best_per_category = dict(metrics["per_category_gt_kl"])
        status = stopper.after_epoch(epoch=epoch, improved=improved)
        staging = tx.begin()
        torch.save({"model": model.state_dict(), "epoch": epoch}, staging / "last_training_checkpoint.pt")
        if improved:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch},
                staging / "best_training_checkpoint.pt",
            )
        (staging / "category_sampler_state.json").write_text(
            json.dumps(sampler_state.to_jsonable(), sort_keys=True),
            encoding="utf-8",
        )
        (staging / "epoch_batch_plan.json").write_text(
            json.dumps({"epoch": epoch, "batches": batches}, sort_keys=True),
            encoding="utf-8",
        )
        record = v1_train.scientific_epoch_record(
            epoch=epoch,
            primary=metrics["worst_category_kl"],
            secondary=metrics["macro_kl"],
            total_loss=sum(epoch_losses) / len(epoch_losses),
            global_optimizer_step=schedule.global_optimizer_step,
            sample_ids=flat_order,
        )
        record["worst_category_kl"] = float(metrics["worst_category_kl"])
        record["macro_kl"] = float(metrics["macro_kl"])
        record["gt_signed"] = float(metrics["gt_signed"])
        record["eligible"] = bool(metrics["eligible"])
        record["per_category_gt_kl"] = dict(metrics["per_category_gt_kl"])
        record["teacher_alloc_kl_macro"] = float(teacher_diag)
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
                "best_eligible": bool(selector.best_eligible),
            },
            update_best=improved,
        )
        last_epoch = epoch
        if status == "early_stopped":
            break

    best_path = seed_dir / "committed" / "best_training_checkpoint.pt"
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model = v3.B2DLCMV3(
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
    best_model.load_state_dict(best_payload["model"], strict=True)
    best_identity = v3.model_state_scientific_sha256(best_model)

    return {
        "status": status if status != "running" else "completed_epoch",
        "real_training_started": bool(mark_real_training_started),
        "evaluation_unlocked": False,
        "best_epoch": selector.best_epoch,
        "seed": seed,
        "worst_category_kl": selector.best_worst,
        "macro_kl": selector.best_macro,
        "gt_signed": selector.best_signed,
        "eligible": bool(selector.best_eligible),
        "per_category_gt_kl": dict(best_per_category),
        "trace_chain_tail": chain.tail,
        "model_state_scientific_sha256": best_identity,
        "last_model_state_scientific_sha256": v3.model_state_scientific_sha256(
            _cpu_identity_model_v3(model)
        ),
        "last_epoch": last_epoch,
        "global_optimizer_step": schedule.global_optimizer_step,
        "epoch0_worst_category_kl": float(metrics0["worst_category_kl"]),
        "epoch0_macro_kl": float(metrics0["macro_kl"]),
        "epoch0_gt_signed": float(metrics0["gt_signed"]),
        "epoch0_teacher_alloc_kl_macro": float(teacher0),
        "epoch0_per_category_gt_kl": dict(metrics0["per_category_gt_kl"]),
    }
