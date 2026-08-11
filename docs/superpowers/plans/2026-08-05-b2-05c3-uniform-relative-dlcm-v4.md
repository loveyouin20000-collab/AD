# B2-05C3 Uniform-Relative DLCM V4 Implementation Plan

## Base

- Worktree: `/root/autodl-tmp/AD-phase-b2-dlcm-uniform-relative-contract-v4`
- Branch: `phase-b2-dlcm-uniform-relative-contract-v4`
- Base HEAD: `99c26de94ba7fa5358a7670473876c4a4cf1829d` (C2B development-unqualified)
- Base evidence tag: `b2-dlcm-category-robust-unqualified-evidence-v1` → `99c26de94ba7fa5358a7670473876c4a4cf1829d`
- Python: `/root/miniconda3/envs/rad-visualad/bin/python` (3.10)
- Architecture: `docs/phase_b/b2_05c3_uniform_relative_dlcm_v4_architecture.md`

## Deliverables

| Path | Role |
|------|------|
| `rad/phase_b/b2_dlcm_v4.py` | Re-export V3/V2 four-head model under V4 identity; batch-matched relative regret; relative Smooth-Max; `total_dlcm_v4_loss` |
| `rad/phase_b/b2_dlcm_v4_training.py` | Category-balanced sampler (reuse V3); eligibility; worst-relative-regret selector; canonical seed; C3A dry-run |
| `rad/phase_b/b2_dlcm_v4_protocol.py` | V4 error codes; reject bypass; forbid final content; require real-training gate |
| `rad/phase_b/b2_dlcm_v4_evaluation.py` | Development/final gates with unchanged thresholds; C1/C2 comparison diagnostic only |
| `rad/phase_b/b2_dlcm_v4_deployment.py` | Export trunk+GT head with V4 architecture pins |
| `rad/phase_b/b2_dlcm_v4_roster_adoption.py` | Adopt C1 roster; prove no reselection/no paths/no unlock |
| `tools/train_b2_dlcm_v4.py` | C3A CLI (`--config --seed --output-dir --dry-run`) |
| `tools/adopt_b2_dlcm_final_roster_v4.py` | Write adoption manifest + sha256 |
| `tools/materialize_b2_dlcm_final_v4.py` | Fail-closed stub |
| `tools/evaluate_b2_dlcm_final_v4.py` | Fail-closed stub |
| `tools/verify_b2_dlcm_v4_artifacts.py` | Receipt/schema verification |
| `configs/phase_b/b2_dlcm_uniform_relative_contract_v4.json` | `real_training_enabled=false`, `tau=0.05`, sampler contract version |
| `configs/phase_b/b2_dlcm_uniform_relative_official_v4.json` | Official stub for C3B; C3A leaves training disabled |
| `tests/rad/test_b2_dlcm_v4_*.py` | RED→GREEN suites |
| `tests/rad/b2_dlcm_v4_fixtures.py` | Hermetic bottle/carpet 8+8 training fixtures |
| `docs/phase_b/b2_05c3_final_roster_adoption_manifest.json(+.sha256)` | After implementation commit |

## Constants

```text
ARCHITECTURE_CONTRACT_VERSION = "b2_dlcm_architecture_v4"
MODEL_CLASS_ID = "rad.phase_b.b2_dlcm_v4.B2DLCMV4"
SMOOTHMAX_TAU = 0.05
SAMPLER_CONTRACT_VERSION = "b2_dlcm_category_balanced_sampler_v1"
TEACHER_ALLOC_WEIGHT = 0.25
GT_SIGNED_WEIGHT = 0.25
TEACHER_SIGNED_WEIGHT = 0.0625
TRAINING_CATEGORIES = ("bottle", "carpet")
BATCH_SIZE = 4
PER_CATEGORY_PER_BATCH = 2
GT_DEPLOYMENT_AGGREGATION = "uniform_relative_smooth_max"
```

`B2DLCMV4` wraps/aliases `B2DLCMV3`/`B2DLCMV2` architecture with identical
weights/init. Category tensors never enter `forward_training`.

Immutable pins:

```text
V3_CONTRACT_TAG = "b2-dlcm-category-robust-contract-v3"
V3_UNQUALIFIED_TAG = "b2-dlcm-category-robust-unqualified-evidence-v1"
V3_UNQUALIFIED_COMMIT = "99c26de94ba7fa5358a7670473876c4a4cf1829d"
ADOPTED_ROSTER_SCIENTIFIC = "267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213"
```

## Error codes

| Code | Meaning |
|------|---------|
| `B2_DLCM_V4_REAL_TRAINING_NOT_ENABLED` | Non-dry-run while C3A disables training |
| `B2_DLCM_V4_CONTRACT_MISMATCH` | Config/schema/identity mismatch |
| `B2_DLCM_CATEGORY_BATCH_INVALID` | Batch not exactly 2+2 bottle/carpet |
| `B2_DLCM_CATEGORY_COVERAGE_INVALID` | Epoch coverage or missing category |
| `B2_DLCM_UNIFORM_BASELINE_INVALID` | Uniform baseline shape/bits invalid |
| `B2_DLCM_RELATIVE_REGRET_INVALID` | Nonfinite regret / mismatched model-uniform targets |
| `B2_DLCM_RELATIVE_SMOOTHMAX_INVALID` | Nonfinite relative Smooth-Max / invalid tau |
| `B2_DLCM_NO_ELIGIBLE_CHECKPOINT` | Diagnostic when no trained eligible exists (Epoch 0 retained) |
| `B2_DLCM_ROSTER_ADOPTION_MISMATCH` | Adoption binds wrong identity/records |
| `B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN` | Final content before unlock |
| `B2_DLCM_DEVELOPMENT_UNQUALIFIED` | Development gates failed |
| `B2_DLCM_FINAL_MATERIALIZATION_MISMATCH` | Materialization A/B inequality |
| `B2_DLCM_FINAL_EVALUATION_MISMATCH` | Evaluation A/B inequality |
| `B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN` | Accepted identity before final pass |

No bypass flags (`force_unlock`, `bypass_gates`, `skip_development`,
`allow_final_without_development`, `ignore_auxiliary_diagnostics`).

## Story map (RED→GREEN)

### Story 1 — Relative regret and V4 total loss

**Tests first (must fail):** `tests/rad/test_b2_dlcm_v4_relative_regret.py`,
`tests/rad/test_b2_dlcm_v4_smoothmax.py`, `tests/rad/test_b2_dlcm_v4_losses.py`

- model/uniform category means use identical batch indices and identical `p_gt`
- `R = K_model - K_uniform` with no slack, no clamp, no abs
- negative regret preserves sign and IEEE float32 bits under direct subtraction
- uniform logits/weights match frozen FP32 softmax baseline
  (`reference_uniform_weights` / zero-logit softmax equality)
- `relative_smooth_max([a,a], tau=0.05) == a`
- matches direct formula within 1e-12 on finite pairs (including negatives)
- numerical stability for large separated regrets
- worse (larger) regret receives larger ∂L/∂R_c than better regret
- tau fixed 0.05; hard-max inequality when regrets differ
- no GroupDRO mutable state
- `total_dlcm_v4_loss` uses relative Smooth-Max only for GT deployment
- teacher alloc / GT signed / teacher signed use sample mean
- coefficients 1 + 0.25 + 0.25 + 0.0625; depths equal weight
- category labels passed only to loss aggregation, not model forward
- aggregation pin `gt_deployment == "uniform_relative_smooth_max"`

**Implement:** `rad/phase_b/b2_dlcm_v4.py`

```python
def frozen_uniform_logits(batch: int, n_players: int, *, device, dtype=torch.float32) -> Tensor: ...
def batch_matched_category_kl_and_uniform(... ) -> dict[str, dict[str, Tensor]]: ...
def relative_regret_from_category_kl(model_kl, uniform_kl) -> dict[str, Tensor]: ...
def relative_smooth_max_normalized(regrets, *, tau=0.05) -> Tensor: ...
def total_dlcm_v4_loss(depth_batch, *, categories, tau=0.05, ...) -> tuple[Tensor, dict]: ...
```

Reuse V3 `per_sample_allocation_kl` / `category_mean_allocation_kl` semantics
for both model logits and uniform logits on the same `p` and category masks.

### Story 2 — Category-balanced sampler (unchanged)

**Tests:** `tests/rad/test_b2_dlcm_v4_sampler.py`

Reuse V3 sampler contract exactly via import or thin re-export under V4 module
identity. Verify:

- each of 4 batches has exactly 2 bottle + 2 carpet
- each of 16 IDs appears once per epoch
- resume exact
- missing category → `B2_DLCM_CATEGORY_COVERAGE_INVALID`
- invalid batch → `B2_DLCM_CATEGORY_BATCH_INVALID`

### Story 3 — Eligibility and worst-relative selection

**Tests first:** `tests/rad/test_b2_dlcm_v4_selection.py`

- eligibility: macro margin and per-category slack vs uniform (absolute KL)
- ineligible never replaces best / never resets patience
- eligible selects by worst relative regret → macro → signed → earlier epoch
- no eligible trained → Epoch 0 best; may surface `B2_DLCM_NO_ELIGIBLE_CHECKPOINT`
- canonical: eligible-first → worst relative → macro → signed → min seed
- all Epoch-0 → seed 17
- teacher/development metrics absent from selector inputs
- selector field name: `worst_relative_regret`

**Implement:** `EligibleWorstRelativeRegretSelector`,
`calibration_metrics_uniform_relative`, `select_canonical_seed_uniform_relative`,
`is_checkpoint_eligible` (absolute gates unchanged).

### Story 4 — Protocol, deployment, evaluation stubs

**Tests:** `tests/rad/test_b2_dlcm_v4_protocol.py`,
`tests/rad/test_b2_dlcm_v4_deployment.py`,
`tests/rad/test_b2_dlcm_v4_evaluation.py`,
`tests/rad/test_b2_dlcm_v4_model.py`

- all V4 error codes present; no bypass flags
- C3A forbids final content access
- deployment exports trunk+GT head only with V4 pins
- gates thresholds identical to V3/V2
- V1/V2/V3 immutable identity helpers

### Story 5 — Training dry-run + CLI

**Tests:** `tests/rad/test_b2_dlcm_v4_training.py`,
`tests/rad/test_b2_dlcm_v4_cli.py`,
`tests/rad/test_b2_dlcm_v4_contract_closure.py`

Dry-run twice must assert:

```text
real_training_started = false
development_evaluation_started = false
final_content_resolved = false
final_materialization_started = false
final_evaluation_started = false
artifact_written = false
run_directory_created = false
teacher_forward_count = 0
```

Contract config: `contract_stage=b2_05c3a`, `real_training_enabled=false`,
`smoothmax_tau=0.05`.

### Story 6 — Roster adoption (after implementation commit)

**Tests:** `tests/rad/test_b2_dlcm_v4_roster_adoption.py`

- scientific identity exact match
- ordered stable IDs identical to C1 roster
- `final_content_resolved=false`
- `paths_present=false`
- no unlock/receipt/accepted
- `selection_reused_without_change=true`
- binds V4 implementation commit + contract identity
- C3A must not open final image/mask/descriptor content

## File-level porting rule

Start from V3 modules under the same relative paths with `v3`→`v4` rename.
Do **not** mutate V3 modules. Core behavioral deltas are confined to:

1. GT deployment loss aggregation (relative regret + relative Smooth-Max)
2. Checkpoint/seed selection key (`worst_relative_regret`)
3. Contract/stage/config/error-code identity pins
4. Roster adoption schema version / commit binding

## Verification gates

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /root/miniconda3/envs/rad-visualad/bin/python -m pytest tests/rad/test_b2_dlcm_v4_*.py -q --tb=short

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /root/miniconda3/envs/rad-visualad/bin/python -m pytest tests/rad -q --tb=short

ruff check rad/phase_b/b2_dlcm_v4*.py tools/*b2_dlcm*v4*.py tests/rad/test_b2_dlcm_v4_*.py
# scoped mypy on new V4 modules
```

Independent review must report Critical=0 and Important=0 before tagging.

## Commit sequence

1. `docs: define uniform-relative B2 DLCM V4 design`
2. `feat: define uniform-relative B2 DLCM V4 contract`  ← freeze `V4_IMPLEMENTATION_COMMIT`
3. `data: adopt untouched B2 DLCM final roster for V4`
4. Local annotated tag `b2-dlcm-uniform-relative-contract-v4` on HEAD

No push.
