# B2-05C2 Category-Robust DLCM V3 Implementation Plan

## Base

- Worktree: `/root/autodl-tmp/AD-phase-b2-dlcm-category-robust-contract-v3`
- Branch: `phase-b2-dlcm-category-robust-contract-v3`
- Base HEAD: `1044b885e86cff4c7f4e1a635f0ebc417105e854` (C1B development-unqualified)
- Base contract tag: `b2-dlcm-decoupled-contract-v2` → `e54f2b44eeb962b05cfb7cf74764e55905f1a8f6`
- Python: `/root/miniconda3/envs/rad-visualad/bin/python` (3.10)
- Architecture: `docs/phase_b/b2_05c2_category_robust_dlcm_v3_architecture.md`

## Deliverables

| Path | Role |
|------|------|
| `rad/phase_b/b2_dlcm_v3.py` | Re-export V2 four-head model under V3 identity; Smooth-Max GT deploy loss; `total_dlcm_v3_loss` |
| `rad/phase_b/b2_dlcm_v3_training.py` | Category-balanced sampler; eligibility; worst-category selector; canonical seed; C2A dry-run |
| `rad/phase_b/b2_dlcm_v3_protocol.py` | V3 error codes; reject bypass; forbid final content; require real-training gate |
| `rad/phase_b/b2_dlcm_v3_evaluation.py` | Development/final gates with unchanged thresholds; C1 comparison diagnostic only |
| `rad/phase_b/b2_dlcm_v3_deployment.py` | Export trunk+GT head with V3 architecture pins |
| `rad/phase_b/b2_dlcm_v3_roster_adoption.py` | Adopt C1 roster; prove no reselection/no paths/no unlock |
| `tools/train_b2_dlcm_v3.py` | C2A CLI (`--config --seed --output-dir --dry-run`) |
| `tools/adopt_b2_dlcm_final_roster_v3.py` | Write adoption manifest + sha256 |
| `tools/materialize_b2_dlcm_final_v3.py` | Fail-closed stub |
| `tools/evaluate_b2_dlcm_final_v3.py` | Fail-closed stub |
| `tools/verify_b2_dlcm_v3_artifacts.py` | Receipt/schema verification |
| `configs/phase_b/b2_dlcm_category_robust_contract_v3.json` | `real_training_enabled=false`, `tau=0.05`, sampler contract version |
| `configs/phase_b/b2_dlcm_category_robust_official_v3.json` | Official stub for C2B; C2A leaves training disabled semantics for dry tools |
| `tests/rad/test_b2_dlcm_v3_*.py` | RED→GREEN suites |
| `tests/rad/b2_dlcm_v3_fixtures.py` | Hermetic bottle/carpet 8+8 training fixtures |
| `docs/phase_b/b2_05c2_final_roster_adoption_manifest.json(+.sha256)` | After implementation commit |

## Constants

```text
ARCHITECTURE_CONTRACT_VERSION = "b2_dlcm_architecture_v3"
MODEL_CLASS_ID = "rad.phase_b.b2_dlcm_v3.B2DLCMV3"
SMOOTHMAX_TAU = 0.05
SAMPLER_CONTRACT_VERSION = "b2_dlcm_category_balanced_sampler_v1"
TEACHER_ALLOC_WEIGHT = 0.25
GT_SIGNED_WEIGHT = 0.25
TEACHER_SIGNED_WEIGHT = 0.0625
TRAINING_CATEGORIES = ("bottle", "carpet")
BATCH_SIZE = 4
PER_CATEGORY_PER_BATCH = 2
```

`B2DLCMV3` wraps/aliases `B2DLCMV2` architecture with identical weights/init.
Category tensors never enter `forward_training`.

## Error codes

| Code | Meaning |
|------|---------|
| `B2_DLCM_V3_REAL_TRAINING_NOT_ENABLED` | Non-dry-run while C2A disables training |
| `B2_DLCM_V3_CONTRACT_MISMATCH` | Config/schema/identity mismatch |
| `B2_DLCM_CATEGORY_BATCH_INVALID` | Batch not exactly 2+2 bottle/carpet |
| `B2_DLCM_CATEGORY_COVERAGE_INVALID` | Epoch coverage or missing category |
| `B2_DLCM_SMOOTHMAX_INVALID` | Nonfinite Smooth-Max / invalid tau |
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

### Story 1 — Smooth-Max and V3 total loss

**Tests first (must fail):** `tests/rad/test_b2_dlcm_v3_smoothmax.py`,
`tests/rad/test_b2_dlcm_v3_losses.py`

- `smooth_max_normalized([a,a], tau=0.05) == a`
- matches direct formula within 1e-12 on finite pairs
- numerical stability for large separated values (e.g. 100 vs 0)
- worse category receives larger ∂L/∂L_c than better category
- tau fixed 0.05; hard-max inequality when losses differ
- no GroupDRO mutable state object
- `total_dlcm_v3_loss` uses Smooth-Max only for GT deployment
- teacher alloc / GT signed / teacher signed use sample mean
- coefficients 1 + 0.25 + 0.25 + 0.0625; depths equal weight
- category labels passed only to loss aggregation, not model forward

**Implement:** `rad/phase_b/b2_dlcm_v3.py`

```python
def smooth_max_normalized(losses: Sequence[float | Tensor], *, tau: float = 0.05) -> Tensor: ...
def category_mean_allocation_kl(p, logits, categories, *, expected=("bottle","carpet")) -> dict[str, Tensor]: ...
def total_dlcm_v3_loss(depth_batch, *, categories_per_depth_batch, tau=0.05, ...) -> tuple[Tensor, dict]: ...
```

Reuse `v1.allocation_kl` per-sample internals: compute per-sample KL then mean
within category mask (do not call batch `.mean()` across mixed categories).

### Story 2 — Category-balanced sampler

**Tests first:** `tests/rad/test_b2_dlcm_v3_sampler.py`

- each of 4 batches has exactly 2 bottle + 2 carpet
- each of 16 IDs appears once per epoch
- no normal/anomalous enforcement
- three independent generator states persist/restore
- resume reproduces identical next-epoch batches
- missing category → `B2_DLCM_CATEGORY_COVERAGE_INVALID`
- invalid batch → `B2_DLCM_CATEGORY_BATCH_INVALID`

**Implement in** `rad/phase_b/b2_dlcm_v3_training.py`:

```python
@dataclass
class CategoryBalancedSamplerState:
    bottle_generator_state: bytes
    carpet_generator_state: bytes
    batch_order_generator_state: bytes
    epoch_index: int
    sampler_contract_version: str  # "b2_dlcm_category_balanced_sampler_v1"

def build_category_balanced_epoch_batches(
    records, *, epoch, bottle_seed, carpet_seed, batch_order_seed
) -> tuple[list[list[str]], CategoryBalancedSamplerState]: ...
```

Hermetic fixture builder produces 8 bottle + 8 carpet training records
(remap V1 hermetic IDs or construct explicitly). Calibration retains mixed
categories filtered to bottle/carpet (4+4) for eligibility tests.

### Story 3 — Eligibility and selection

**Tests first:** `tests/rad/test_b2_dlcm_v3_selection.py`

- eligibility: macro margin and per-category slack vs uniform
- ineligible never replaces best / never resets patience
- eligible selects by worst-category → macro → signed → earlier epoch
- no eligible trained → Epoch 0 best; may surface `B2_DLCM_NO_ELIGIBLE_CHECKPOINT` diagnostic
- canonical: eligible-first → worst → macro → signed → min seed
- all Epoch-0 → seed 17
- teacher/development metrics absent from selector inputs

**Implement:** `EligibleWorstCategorySelector`,
`calibration_metrics_category_robust`, `select_canonical_seed_category_robust`,
`is_checkpoint_eligible`.

Depth-24 GT KL vs uniform computed on calibration bottle/carpet groups.

### Story 4 — Protocol, deployment, evaluation stubs

**Tests first:** `tests/rad/test_b2_dlcm_v3_protocol.py`,
`tests/rad/test_b2_dlcm_v3_deployment.py`,
`tests/rad/test_b2_dlcm_v3_evaluation.py`

- all V3 error codes present
- bypass flags rejected
- `forbid_final_content_access(unlocked=False)` raises
- deployment extract drops aux heads; pins architecture v3
- gates thresholds match C1; no “must beat C1” gate

**Implement:** protocol/deployment/evaluation modules (pattern-copy V2 with V3
names and pins).

### Story 5 — Roster adoption

**Tests first:** `tests/rad/test_b2_dlcm_v3_roster_adoption.py`

- adoption manifest binds roster scientific sha256
  `267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213`
- ordered 16 stable IDs identical to C1 roster
- `selection_reused_without_change=true`
- `final_content_resolved=false`; `paths_present=false`
- proves no unlock/receipt/accepted
- mutating roster IDs → `B2_DLCM_ROSTER_ADOPTION_MISMATCH`
- C2A tools never resolve final content

**Implement:** `b2_dlcm_v3_roster_adoption.py` + `tools/adopt_b2_dlcm_final_roster_v3.py`

### Story 6 — CLI dry-run and contract closure

**Tests first:** `tests/rad/test_b2_dlcm_v3_cli.py`,
`tests/rad/test_b2_dlcm_v3_contract_closure.py`,
`tests/rad/test_b2_dlcm_v3_training.py` (dry-run flags)

Dry-run twice must report:

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

**Implement:** `tools/train_b2_dlcm_v3.py`, dry-run in training module,
fail-closed materialize/evaluate stubs, verify tool, configs.

## Config schema keys (contract v3)

```json
{
  "schema_version": "b2_dlcm_category_robust_contract_v3",
  "contract_stage": "b2_05c2a",
  "real_training_enabled": false,
  "smoothmax_tau": 0.05,
  "sampler_contract_version": "b2_dlcm_category_balanced_sampler_v1",
  "training_categories": ["bottle", "carpet"],
  "per_category_per_batch": 2,
  "authoritative_v2_contract_tag": "b2-dlcm-decoupled-contract-v2",
  "authoritative_v2_contract_commit": "e54f2b44eeb962b05cfb7cf74764e55905f1a8f6",
  "adopted_final_roster_scientific_sha256": "267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213",
  "candidate_layers": [6,12,18,24],
  "prediction_depths": [12,18,24],
  "descriptor_dimension": 18,
  "layer_embedding_dimension": 8,
  "depth_embedding_dimension": 8,
  "hidden_dimension": 64,
  "dropout_probability": 0.1,
  "teacher_allocation_loss_weight": 0.25,
  "gt_signed_loss_weight": 0.25,
  "teacher_signed_loss_weight": 0.0625,
  "seeds": [17,29,43],
  "batch_size": 4,
  "maximum_epochs": 500,
  "patience": 50,
  "min_delta": 1e-5
}
```

Plus unchanged optimizer / LR / dtype / AMP / resume keys from V2 contract.

## Verification commands

Focused:

```bash
cd /root/autodl-tmp/AD-phase-b2-dlcm-category-robust-contract-v3
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/root/miniconda3/envs/rad-visualad/bin/python -m pytest tests/rad/test_b2_dlcm_v3*.py -q --tb=short
```

Full CPU:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/root/miniconda3/envs/rad-visualad/bin/python -m pytest tests/rad -q --tb=short
```

Ruff + scoped mypy:

```bash
/root/miniconda3/envs/rad-visualad/bin/ruff check rad/phase_b/b2_dlcm_v3*.py tools/*v3*.py tests/rad/test_b2_dlcm_v3*.py
/root/miniconda3/envs/rad-visualad/bin/mypy rad/phase_b/b2_dlcm_v3.py rad/phase_b/b2_dlcm_v3_training.py rad/phase_b/b2_dlcm_v3_protocol.py
```

Dry-run twice via `tools/train_b2_dlcm_v3.py --dry-run`.

## Commit sequence

1. `docs: define category-robust B2 DLCM V3 design` (this plan + architecture)
2. `feat: define category-robust B2 DLCM V3 contract` (implementation after review)
3. `data: adopt untouched B2 DLCM final roster for V3` (adoption manifest only)
4. Local annotated tag `b2-dlcm-category-robust-contract-v3` on adoption commit

## Non-goals (enforce in code)

- No `git push` / remote tags / PRs
- No real training in C2A
- No final content resolution
- No mutation of V1/V2 modules, schemas, or roster records
