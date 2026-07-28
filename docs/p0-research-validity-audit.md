# P0 Research-Validity Audit

Audit date: 2026-07-20  
Verified code SHA: `4d085cb1ec1cdd06628423fedacb505507bbcd92`  
Environment: Python 3.10.20, PyTorch 2.0.0+cu118, `rad-visualad` conda env  
Machine-readable summary: [`docs/p0_manifest.json`](p0_manifest.json)

## Scope

This audit covers **Phase A (P0) CPU implementation contracts** for
Residual-Aware Adaptive-Depth VisualAD. It verifies:

- source-controlled acceptance records for bidirectional official VisualAD baselines;
- dataset-level metric and adapter contracts;
- fail-closed artifact handling;
- selector-signal masking semantics;
- experiment-matrix configuration semantics;
- CI parity (Ruff, mypy, pytest, dry-runs).

This audit **does not** claim CUDA numerical equivalence, full adaptive training
validation, or final paper results.

## Accepted baseline evidence

### MVTec → VisA (`docs/baseline_acceptance/mvtec_to_visa_seed111_official.json`)

| Check | Status | Evidence |
|---|---|---|
| Official protocol epoch=2, batch_size=8 | pass | `protocol` field |
| Accepted checkpoint SHA recorded | pass | `checkpoint.sha256` = `97bd4611…` |
| Category-macro AUPRO ≈ 0.9087 | pass | `accepted_pixel_aupro.value` = 0.9087127714084767 |
| AUPRO provenance max_fpr=0.3, steps=200 | pass | `accepted_pixel_aupro.max_fpr/steps` |
| Original placeholder-zero defect documented | pass | `frozen_original_run.invalid_placeholder_pixel_aupro` = 0.0 |
| Retraining not required | pass | `corrected_evaluation.eval_only` = true, `train_invoked` = false |

### VisA → MVTec (`docs/baseline_acceptance/visa_to_mvtec_seed111_official.json`)

| Check | Status | Evidence |
|---|---|---|
| Official protocol epoch=1, batch_size=8 | pass | `protocol` field |
| Accepted checkpoint SHA recorded | pass | `checkpoint.sha256` = `a4dd80dd…` |
| All predefined tolerances passed | pass | `tolerance_checks` all true |
| Production metric path reports real AUPRO | pass | `pixel_aupro` = 0.8764823107188563, `aupro_provenance` = true |

Local `runs/` directories are **not required** in this worktree; acceptance
is based on source-controlled records only.

## P0 implementation status

| Area | Status | Notes |
|---|---|---|
| Baseline reproduction ordering | validated (CPU contract) | train → checkpoint verify → test |
| Smoke vs real evaluation separation | validated | paper matrix uses `evaluate_adaptive_dataset.py` only |
| Authoritative dataset-level metrics | validated | `compute_paper_metrics` path |
| MVTec/VisA adapters | validated | registry + adapter tests |
| Fail-closed fusion training | validated | gate tests |
| Fair fixed-exit + selector matrix | validated | config + contract tests |
| Residual-gain source | validated | `sample_localization_error` only |
| Selector ablation masking | validated | post-normalization, pre-LSE |
| Legacy test contract sync | validated | Increment 11 shared helpers |
| Artifact hygiene | validated | no tracked generated artifacts |

## Test and CI evidence

### Full CPU suite

```
270 passed, 2 skipped in 45.98s
SKIPPED: test_benchmark_smoke (CUDA), test_environment (CUDA AutoDL gate)
```

### Focused P0 suites (all pass)

| Suite | Passed |
|---|---:|
| `test_dataset_adapter_registry.py` | 25 |
| `test_mvtec_adapter.py` | 7 |
| `test_visa_adapter.py` | 7 |
| `test_preprocess_contract.py` | 9 |
| `test_paper_metrics.py` | 10 |
| `test_dataset_evaluator.py` | 7 |
| `test_adaptive_dataset_cli.py` | 5 |
| `test_baseline_pipeline.py` | 16 |
| `test_fusion_fail_closed.py` | 15 |
| `test_selector_signal_mask.py` | 14 |
| `test_experiment_matrix.py` | 11 |
| `test_zero_shot_transfer_contract.py` | 8 |

### Static CI checks

| Check | Result |
|---|---|
| `ruff check rad tests/rad tools/export_paper_tables.py` | pass |
| `mypy rad/evaluation/paper_tables.py tools/export_paper_tables.py` | pass |
| `export_paper_tables.py --dry-run` | pass |
| `run_experiment_matrix.py --dry-run` | pass |
| `artifacts/` after verification | absent |
| `git ls-files artifacts` | empty |

## Metric-protocol decisions

1. **AUPRO reporting** uses category-macro aggregation with `max_fpr=0.3`,
   `steps=200` (see baseline acceptance records and `utils/metrics.py` fix in
   commit `af06217`).
2. **Deprecated `pro_score_proxy` (non-reporting technical debt):** The helper
   remains defined in `rad/evaluation/zero_shot.py` and is imported by
   non-reporting utilities (`tools/train_joint.py`,
   `tools/export_staged_checkpoint_v1.py`). It is **not** used by
   `PaperMetrics` / `rad/evaluation/paper_metrics.py`, baseline acceptance
   (`tools/reproduce_baseline.py`, `docs/baseline_acceptance/*.json`),
   dataset evaluation (`tools/evaluate_adaptive_dataset.py`,
   `rad/evaluation/dataset_evaluator.py`), or paper-table exports
   (`tools/export_paper_tables.py`, `rad/evaluation/paper_tables.py`). Removal
   is deferred to Phase B cleanup; P0 reporting paths do not call it.
3. **Residual gain** in `rad/evaluation/dataset_evaluator.py` uses
   `sample_localization_error` from `rad.losses.localization` for both adaptive
   and full-depth errors. No AP-derived residual gain exists in the evaluator.

## Dataset-adapter scope

Validated adapters: **MVTec** and **VisA** only (`supported_dataset_names()`).
Both route through `get_adapter()` and shared `evaluate_dataset()` path.

## Fail-closed behavior

Validated fail-closed gates include:

- fusion training without approved gates (`test_fusion_fail_closed.py`);
- baseline output protection and checkpoint integrity (`test_baseline_pipeline.py`);
- zero-shot policy fixture rejection and missing-policy exit 3
  (`test_zero_shot_transfer_contract.py`);
- target tuning rejection (`test_no_target_tuning.py`).

## Experiment-matrix semantics

Verified in `configs/rad/experiments.yaml` and
`tests/rad/contracts/experiment_matrix.py`:

| Row ID | exit_depth | fusion |
|---|---:|---|
| `fixed_exit_12_equal` | 12 | equal |
| `fixed_exit_18_equal` | 18 | equal |
| `fixed_exit_12_dynamic` | 12 | dynamic |
| `fixed_exit_18_dynamic` | 18 | dynamic |

Selector leave-one-out rows present:

- `selector_without_response`
- `selector_without_uncertainty`
- `selector_without_stability`
- `selector_without_complementarity`
- `selector_without_token_separation`

Paper evaluation rows use `tools/evaluate_adaptive_dataset.py`. No matches for
`smoke_adaptive_engine.py` or `tools/evaluate_adaptive.py` in
`configs/rad/experiments.yaml`.

## Selector-signal masking

Implementation contract (`rad/models/selector_signals.py`):

- stage: `post_normalization_pre_lse` (`SELECTOR_MASK_STAGE`);
- mask applied to **LSE descriptor clone only**; DLCM receives full descriptor
  (`rad/inference/adaptive_engine.py`);
- layout and `selector_signal_layout_hash` recorded in provenance.

Tests: `tests/rad/test_selector_signal_mask.py` (14 passed).

## Known limitations

Not validated in this P0 audit:

- staged-backbone CUDA numerical equivalence;
- complete AutoDL Gate C pipeline;
- teacher-cache generation at full scale;
- exact Shapley generation on the full source dataset;
- full DLCM training;
- residual-gain target generation at full scale;
- full LSE training;
- source-policy calibration runs;
- eight-method pilot;
- adaptive latency gains;
- final paper claims or result matrix.

## Phase B entry criteria

Phase B may begin only after:

1. AutoDL CUDA equivalence tests pass for staged execution;
2. full-scale cache, target generation, and staged training gates complete;
3. source-only policy calibration artifacts exist and pass eligibility checks;
4. experiment matrix rows execute (not dry-run only) under GPU gates;
5. latency benchmark meets the batch-1 gate;
6. paper table export uses completed result artifacts, not placeholders.

## Final P0 decision

**P0 CPU contracts: VERIFIED.**

Status: `p0_verified_cpu` (see manifest). This is **not** `paper_ready`,
`fully_validated`, or `experiment_complete`.

**Phase B not started.** GPU training, full-scale target generation, policy
calibration, and experiment-matrix execution remain deferred until Phase B
entry criteria are met and this documentation PR is merged.
