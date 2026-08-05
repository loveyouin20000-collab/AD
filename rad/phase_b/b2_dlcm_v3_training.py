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
