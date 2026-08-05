# B2-05C4 Uniform-Anchored DLCM V5 Implementation Plan

## Base

- Worktree: `/root/autodl-tmp/AD-phase-b2-dlcm-uniform-anchored-contract-v5`
- Branch: `phase-b2-dlcm-uniform-anchored-contract-v5`
- Base HEAD: `a1447bdabdd7f54eb7883b717dfadc3da906da5b` (C3B development-unqualified)
- Base evidence tag: `b2-dlcm-uniform-relative-unqualified-evidence-v1` → `a1447bdabdd7f54eb7883b717dfadc3da906da5b`
- Python: `/root/miniconda3/envs/rad-visualad/bin/python` (3.10)
- Architecture: `docs/phase_b/b2_05c4_uniform_anchored_dlcm_v5_architecture.md`

## Deliverables

| Path | Role |
|------|------|
| `rad/phase_b/b2_dlcm_v5.py` | Constants, identity pins, FP32 convex weight mix, depth-matched uniform, KL helpers re-export, V5 scientific identity builders |
| `rad/phase_b/b2_dlcm_v5_calibration.py` | Beta grid, eligibility, Depth-24 LOO objective, selection, Calibration A/B runner, canonical manifest |
| `rad/phase_b/b2_dlcm_v5_protocol.py` | V5 error codes; reject bypass; forbid training/final/accepted; dry-run status payload |
| `rad/phase_b/b2_dlcm_v5_evaluation.py` | Development/final gates with thresholds identical to V4/V3/V2; C4 termination helper |
| `rad/phase_b/b2_dlcm_v5_deployment.py` | Load frozen C3 deployment; wrap with frozen scalar beta; prove checkpoint tensors unchanged |
| `rad/phase_b/b2_dlcm_v5_roster_adoption.py` | Adopt C1 roster; prove no reselection/no paths/no unlock |
| `tools/calibrate_b2_dlcm_v5.py` | C4A CLI (`--config --output-dir --dry-run [--process-label A|B]`) |
| `tools/adopt_b2_dlcm_final_roster_v5.py` | Write adoption manifest + sha256 |
| `tools/materialize_b2_dlcm_final_v5.py` | Fail-closed stub |
| `tools/evaluate_b2_dlcm_final_v5.py` | Fail-closed stub |
| `tools/verify_b2_dlcm_v5_artifacts.py` | Receipt/schema verification |
| `configs/phase_b/b2_dlcm_uniform_anchored_contract_v5.json` | `real_training_enabled=false`, `calibration_enabled=false` for C4A dry-run default |
| `configs/phase_b/b2_dlcm_uniform_anchored_official_v5.json` | Official stub for C4B; C4A leaves calibration/development disabled |
| `tests/rad/test_b2_dlcm_v5_*.py` | RED→GREEN suites |
| `tests/rad/b2_dlcm_v5_fixtures.py` | Hermetic Calibration 4+4 records with GT targets + C3-like weights |
| `docs/phase_b/b2_05c4_final_roster_adoption_manifest.json(+.sha256)` | After implementation commit |

## Constants

```text
ARCHITECTURE_CONTRACT_VERSION = "b2_dlcm_architecture_v5"
MODEL_CLASS_ID = "rad.phase_b.b2_dlcm_v5.B2DLCMV5"
CALIBRATION_CONTRACT_VERSION = "b2_dlcm_uniform_anchored_calibration_v5"
BETA_GRID_SIZE = 101
BETA_INDEX_MIN = 0
BETA_INDEX_MAX = 100
LOO_DEPTH = 24
GT_MACRO_MARGIN = 1e-5
GT_PER_CATEGORY_SLACK = 1e-4
LOO_TIE_EPS = 1e-5
TRAINING_CATEGORIES = ("bottle", "carpet")
CALIBRATION_PER_CATEGORY = 4
```

Immutable pins:

```text
V4_CONTRACT_TAG = "b2-dlcm-uniform-relative-contract-v4"
V4_UNQUALIFIED_TAG = "b2-dlcm-uniform-relative-unqualified-evidence-v1"
V4_UNQUALIFIED_COMMIT = "a1447bdabdd7f54eb7883b717dfadc3da906da5b"
V4_ACCEPTED_PLAN_SHA256 = "4979c73a28e0aaffd21f2c6408bb37e90fdc64201bcc326f990543fbbee5650f"
V4_ENVIRONMENT_IDENTITY = "67677c4e9bb83475f7adc03294437bdd104a693e0465e107d3860096a9f03056"
V4_CANONICAL_SEED = 17
ADOPTED_ROSTER_SCIENTIFIC = "267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213"
```

Reuse V4 production KL via import (`b2_dlcm_v4.per_sample_allocation_kl` or shared
helper). Uniform weights must bit-match `reference_uniform_weights` / zero-logit
softmax.

## Error codes

| Code | Meaning |
|------|---------|
| `B2_DLCM_V5_CONTRACT_MISMATCH` | Config/schema/identity mismatch |
| `B2_DLCM_V5_TRAINING_FORBIDDEN` | Any attempt to start real DLCM training |
| `B2_DLCM_V5_BETA_GRID_INVALID` | Grid size/index/decimal string invalid |
| `B2_DLCM_V5_CALIBRATION_INPUT_INVALID` | Calibration records/weights/targets invalid |
| `B2_DLCM_V5_NO_ELIGIBLE_BETA` | No grid candidate passes eligibility |
| `B2_DLCM_V5_CALIBRATION_MISMATCH` | Calibration A/B inequality |
| `B2_DLCM_V5_BETA_SELECTION_INVALID` | Selection violates tie-break / ineligible winner |
| `B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH` | Adoption binds wrong identity/records |
| `B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN` | Final content before unlock |
| `B2_DLCM_DEVELOPMENT_UNQUALIFIED` | Development gates failed |
| `B2_DLCM_FINAL_MATERIALIZATION_MISMATCH` | Materialization A/B inequality |
| `B2_DLCM_FINAL_EVALUATION_MISMATCH` | Evaluation A/B inequality |
| `B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN` | Accepted identity before final pass |

No bypass flags (`force_unlock`, `bypass_gates`, `skip_development`,
`allow_final_without_development`, `ignore_auxiliary_diagnostics`,
`allow_training`).

## Story map (RED→GREEN)

### Story 1 — Convex mix and beta grid

**Tests first (must fail):**
`tests/rad/test_b2_dlcm_v5_weights.py`,
`tests/rad/test_b2_dlcm_v5_beta_grid.py`

- `beta=0` exact uniform (FP32 bits)
- `beta=1` exact dynamic weights (FP32 bits)
- intermediate: `w = (1-β)u + β w̃` in FP32; non-negative; sums to 1
- category tensor never accepted by mix / wrapper APIs
- exact 101 indices; `beta = index/100.0`; decimal strings `"0.00"`…`"1.00"`
- invalid grid → `B2_DLCM_V5_BETA_GRID_INVALID`

**Implement:** `rad/phase_b/b2_dlcm_v5.py` helpers:
`depth_matched_uniform`, `mix_uniform_anchored_weights`, `beta_from_index`,
`beta_decimal_string`, `iter_beta_grid`.

### Story 2 — LOO objective and eligibility/selection

**Tests first:**
`tests/rad/test_b2_dlcm_v5_loo.py`,
`tests/rad/test_b2_dlcm_v5_selection.py`

- Depth-24 only; 4+4 folds → 8 regrets
- negative regret retained (no clamp/abs/slack)
- model/uniform share records/targets
- eligibility: macro margin + per-category slack vs uniform
- no eligible → `B2_DLCM_V5_NO_ELIGIBLE_BETA`
- tie-break: lowest \(M_{LOO}\) → larger beta within `1e-5` → lower macro → smaller index
- beta=0 has no special fallback
- teacher diagnostics ignored by selector

**Implement:** `rad/phase_b/b2_dlcm_v5_calibration.py`

### Story 3 — Deployment wrapper + C3 immutability

**Tests:**
`tests/rad/test_b2_dlcm_v5_deployment.py`

- wrapper stores frozen scalar beta only
- beta=1 reproduces C3 weights from same logits/state
- loading wrapper does not mutate checkpoint state_dict tensors
- category not in checkpoint / forward signature
- `H_deploy,V5` binds V4 deploy hash + beta* + calibration contract identity

### Story 4 — Protocol, evaluation, dry-run CLI

**Tests:**
`tests/rad/test_b2_dlcm_v5_protocol.py`,
`tests/rad/test_b2_dlcm_v5_evaluation.py`,
`tests/rad/test_b2_dlcm_v5_calibration_ab.py`,
`tests/rad/test_b2_dlcm_v5_cli.py`,
`tests/rad/test_b2_dlcm_v5_contract_closure.py`

Dry-run twice must assert:

```text
real_training_started = false
calibration_started = false
development_evaluation_started = false
final_content_resolved = false
final_materialization_started = false
final_evaluation_started = false
artifact_written = false
run_directory_created = false
teacher_forward_count = 0
```

Calibration A/B hermetic: metrics equal, eligible set equal, beta* equal,
canonical JSON byte-equal, scientific identity equal; independent processes
(no shared in-memory model/cache — use separate function invocations with
fresh loads in tests; CLI supports `--process-label`).

Training any non-dry path → `B2_DLCM_V5_TRAINING_FORBIDDEN`.
C4A forbids final content access.
Gates thresholds identical to V4.
C4 termination payload fields present when development unqualified.

### Story 5 — Roster adoption (after implementation commit)

**Tests:** `tests/rad/test_b2_dlcm_v5_roster_adoption.py`

- scientific identity exact match
- ordered stable IDs identical to C1 roster
- `final_content_resolved=false`
- `paths_present=false`
- no unlock/receipt/accepted
- `selection_reused_without_change=true`
- binds V5 implementation commit + contract identity
- C4A must not open final image/mask/descriptor content

## File-level porting rule

- Port protocol/evaluation/roster patterns from V4 with `v4`→`v5` rename and
  V5-specific error codes / pins.
- Do **not** port V4 training/sampler/loss Smooth-Max into V5 as active
  training path; V5 forbids training.
- Deployment loads V4/C3 checkpoint schema and wraps weights; do not rewrite
  checkpoint tensors.

## Config pins (contract)

```json
{
  "schema_version": "b2_dlcm_uniform_anchored_contract_v5",
  "contract_stage": "b2_05c4a",
  "real_training_enabled": false,
  "calibration_enabled": false,
  "development_enabled": false,
  "final_materialization_enabled": false,
  "final_evaluation_enabled": false,
  "beta_grid_size": 101,
  "loo_depth": 24,
  "canonical_seed": 17,
  "adopted_final_roster_scientific_sha256": "267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213",
  "authoritative_v4_unqualified_tag": "b2-dlcm-uniform-relative-unqualified-evidence-v1",
  "authoritative_v4_unqualified_commit": "a1447bdabdd7f54eb7883b717dfadc3da906da5b"
}
```

## Verification commands

```bash
cd /root/autodl-tmp/AD-phase-b2-dlcm-uniform-anchored-contract-v5
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /root/miniconda3/envs/rad-visualad/bin/python -m pytest tests/rad/test_b2_dlcm_v5_*.py -q --tb=short

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /root/miniconda3/envs/rad-visualad/bin/python -m pytest tests/rad -q --tb=short

/root/miniconda3/envs/rad-visualad/bin/ruff check rad/phase_b/b2_dlcm_v5*.py tools/*v5*.py tests/rad/test_b2_dlcm_v5*.py tests/rad/b2_dlcm_v5_fixtures.py
/root/miniconda3/envs/rad-visualad/bin/mypy rad/phase_b/b2_dlcm_v5.py rad/phase_b/b2_dlcm_v5_calibration.py rad/phase_b/b2_dlcm_v5_protocol.py rad/phase_b/b2_dlcm_v5_evaluation.py rad/phase_b/b2_dlcm_v5_deployment.py rad/phase_b/b2_dlcm_v5_roster_adoption.py
```

## Commit sequence

1. `docs: define uniform-anchored B2 DLCM V5 calibration design`
2. `feat: define uniform-anchored B2 DLCM V5 calibration contract`
3. `data: adopt untouched B2 DLCM final roster for V5`
4. Local annotated tag `b2-dlcm-uniform-anchored-contract-v5` (no push)

## Out of scope (C4A)

Real calibration A/B against disk C3 deployment artifacts, Development
evaluation, Final materialization/evaluation, accepted manifest, LSE, push.
