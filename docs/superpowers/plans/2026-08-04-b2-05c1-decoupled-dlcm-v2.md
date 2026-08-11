# B2-05C1 Decoupled DLCM V2 Implementation Plan

## Base

- Worktree: `/root/autodl-tmp/AD-phase-b2-dlcm-decoupled-contract-v2`
- Branch: `phase-b2-dlcm-decoupled-contract-v2`
- Base tag: `b2-dlcm-unqualified-evidence-v1` → `43d856f5ff771957f9f39d0909b1bc87d6b7081b`
- Python: `/root/miniconda3/envs/rad-visualad/bin/python` (3.10)
- Architecture: `docs/phase_b/b2_05c1_decoupled_dlcm_v2_architecture.md`

## Deliverables

| Path | Role |
|------|------|
| `rad/phase_b/b2_dlcm_v2.py` | Four-head model, V2 losses, deployment trunk extract, reuse V1 KL/Huber/rank/fusion/RNG |
| `rad/phase_b/b2_dlcm_v2_training.py` | GT-only selection/patience, hermetic training, dry-run validation |
| `rad/phase_b/b2_dlcm_v2_protocol.py` | Contracts, error codes, identity helpers, unlock receipts, gates schemas |
| `rad/phase_b/b2_dlcm_v2_evaluation.py` | Development/final gates, aux diagnostics, H_decision/H_evidence |
| `rad/phase_b/b2_dlcm_v2_deployment.py` | Deployment export/loader wrapping V1 production path; GT head only |
| `rad/phase_b/b2_dlcm_v2_final_roster.py` | Deterministic 16-record roster builder (identity-only) |
| `tools/train_b2_dlcm_v2.py` | Contract CLI (`--config --seed --output-dir --dry-run`) |
| `tools/build_b2_dlcm_final_roster.py` | Roster freeze CLI |
| `tools/materialize_b2_dlcm_final_v2.py` | Locked until unlock; C1A dry-run fail-closed |
| `tools/evaluate_b2_dlcm_final_v2.py` | Locked until unlock; C1A dry-run fail-closed |
| `tools/verify_b2_dlcm_v2_artifacts.py` | Artifact verification CLI |
| `configs/phase_b/b2_dlcm_decoupled_training_contract_v2.json` | Tracked contract (`real_training_enabled=false`) |
| `configs/phase_b/b2_dlcm_decoupled_training_official_v2.json` | Official stub for C1B (still disabled in C1A) |
| `tests/rad/test_b2_dlcm_v2_*.py` | RED→GREEN suites |
| `tests/rad/b2_dlcm_v2_fixtures.py` | Hermetic fixtures |
| `docs/phase_b/b2_05c1_final_evaluation_roster.json(+.sha256)` | Frozen after implementation commit |

## Error codes

| Code | Meaning |
|------|---------|
| `B2_DLCM_V2_REAL_TRAINING_NOT_ENABLED` | Non-dry-run while contract disables training |
| `B2_DLCM_V2_CONTRACT_MISMATCH` | Config/contract schema or identity mismatch |
| `B2_DLCM_FINAL_ROSTER_INSUFFICIENT` | Group has fewer than 4 candidates |
| `B2_DLCM_FINAL_ROSTER_OVERLAP` | Overlap with original 32 |
| `B2_DLCM_FINAL_ROSTER_SOURCE_INVALID` | Manifest/receipt verification failed |
| `B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN` | Path/content/metrics before unlock |
| `B2_DLCM_DEVELOPMENT_UNQUALIFIED` | Development gates failed |
| `B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_REQUIRED` | Missing materialization unlock |
| `B2_DLCM_FINAL_MATERIALIZATION_UNLOCK_USED` | Unlock already consumed |
| `B2_DLCM_FINAL_MATERIALIZATION_MISMATCH` | Materialization A/B inequality |
| `B2_DLCM_FINAL_EVALUATION_UNLOCK_REQUIRED` | Missing evaluation unlock |
| `B2_DLCM_FINAL_EVALUATION_MISMATCH` | Evaluation A/B inequality |
| `B2_DLCM_AUXILIARY_DIAGNOSTICS_INVALID` | Missing/wrong-source/non-finite diagnostics |
| `B2_DLCM_FINAL_DECISION_INVALID` | H_decision payload invalid |
| `B2_DLCM_FINAL_EVIDENCE_INVALID` | H_evidence payload invalid |
| `B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN` | Accepted identity before final pass |

No bypass flags.

## Story map (RED→GREEN)

### Story 1 — Model and losses

**Tests first (must fail):**

- `tests/rad/test_b2_dlcm_v2_model.py`
  - four heads present; two allocation heads independent and zero-init uniform
  - deployment extract drops teacher allocation + both signed heads
  - production wrapper exposes only deploy weights
  - V1 history constants unchanged (tag target / identities)
- `tests/rad/test_b2_dlcm_v2_losses.py`
  - loss coefficients exactly `1 + 0.25 + 0.25 + 0.0625`
  - GT deploy KL uses GT head; teacher KL uses teacher aux head
  - actual gradient isolation matrix

**Implement:** `rad/phase_b/b2_dlcm_v2.py` reusing `allocation_kl`, `huber_loss`,
`pairwise_ranking_loss`, `signed_loss`, `DeterministicDropout`,
`sum_preserving_fusion`, seed derivation from `b2_dlcm`.

### Story 2 — Training lifecycle and selection

**Tests first:**

- `tests/rad/test_b2_dlcm_v2_training.py`
  - calibration primary/secondary GT-only
  - teacher finite but ignored by selector/patience/canonical
  - epoch-0 primary-only replacement
  - hermetic dry-run flags all false / teacher_forward_count=0

**Implement:** `rad/phase_b/b2_dlcm_v2_training.py` reusing V1 AdamW groups,
`ExplicitLRSchedule`, `TraceHashChain`, `EpochTransaction`, environment
contract helpers.

### Story 3 — Protocol, deployment, evaluation gates, identities

**Tests first:**

- `tests/rad/test_b2_dlcm_v2_protocol.py` — error codes, unlock receipts, no bypass
- `tests/rad/test_b2_dlcm_v2_deployment.py` — GT-only export; aux absent
- `tests/rad/test_b2_dlcm_v2_evaluation.py`
  - development gates; failure blocks final unlock
  - aux diagnostics required, non-blocking
  - development/teacher excluded from `H_decision`
  - final fail forbids `H_accepted`
  - materialization/evaluation unlock/A-B equality (hermetic)

**Implement:** `b2_dlcm_v2_protocol.py`, `b2_dlcm_v2_deployment.py`,
`b2_dlcm_v2_evaluation.py`. Reuse V1 production metrics/fusion/loader
qualification without semantic change.

### Story 4 — Final roster

**Tests first:**

- `tests/rad/test_b2_dlcm_v2_final_roster.py`
  - deterministic 16 records; 4×4 bottle/carpet normal/anomalous
  - zero overlap with original 32
  - no path/filename/URI fields
  - insufficient candidates fail-closed
  - content access forbidden before unlock

**Implement:** `b2_dlcm_v2_final_roster.py` + `tools/build_b2_dlcm_final_roster.py`.
Source: verified Gate-C split source list via `collect_source_records` /
official `split_manifest.json` receipt. Exclusion set = union of training +
calibration + evaluation stable IDs from that manifest.

### Story 5 — CLIs, configs, closure

**Tests first:**

- `tests/rad/test_b2_dlcm_v2_cli.py` — `--config --seed --output-dir --dry-run`
- `tests/rad/test_b2_dlcm_v2_contract_closure.py` — full dry-run twice with
  argument permutation; focused contract assertions

**Implement:** tools + configs. Dry-run must assert:

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

## Commit sequence

1. `docs: define decoupled B2 DLCM V2 design` — architecture + this plan
2. `feat: define decoupled B2 DLCM V2 contract` — all code/tests/configs
   (implementation commit; freeze SHA as `V2_IMPLEMENTATION_COMMIT`)
3. `data: freeze untouched B2 DLCM final evaluation roster` — roster + receipt
   only, bound to implementation commit
4. Local annotated tag `b2-dlcm-decoupled-contract-v2` on roster commit

## Validation commands

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /root/miniconda3/envs/rad-visualad/bin/python -m pytest \
  tests/rad/test_b2_dlcm_v2_*.py -q --tb=short

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /root/miniconda3/envs/rad-visualad/bin/python -m pytest tests/rad -q --tb=short

ruff check rad/phase_b/b2_dlcm_v2*.py tools/*b2_dlcm*v2*.py tests/rad/test_b2_dlcm_v2_*.py
mypy --explicit-package-bases rad/phase_b/b2_dlcm_v2.py \
  rad/phase_b/b2_dlcm_v2_training.py rad/phase_b/b2_dlcm_v2_protocol.py \
  rad/phase_b/b2_dlcm_v2_evaluation.py rad/phase_b/b2_dlcm_v2_deployment.py \
  rad/phase_b/b2_dlcm_v2_final_roster.py
```

Dry-run twice (argument permutation) via `tools/train_b2_dlcm_v2.py`.

Independent review gates: Critical = 0, Important = 0. Focus: V1 immutable,
teacher out of deployment/selection, final content locked, production
metrics/fusion reused, no bypass.

## Explicit non-goals (C1A)

No authoritative seed training, no real development metric reads, no final path
resolution/content, no accepted deployment generation, no LSE/residual-gain,
no push/PR/remote tag.
