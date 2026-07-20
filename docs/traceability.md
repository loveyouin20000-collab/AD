# RAD-VisualAD Traceability Map

This document maps scientific hypotheses to requirements, modules, tests,
experiments, and paper artifacts. It is the Task 24 release-facing index.

## Paper tables

| Table ID | Purpose | Primary experiment IDs |
|---|---|---|
| `main_comparison` | Main method comparison | `original_visualad`, fixed exits, fusion variants, Pareto profiles |
| `fusion_ablation` | DLCM / Shapley / KD / training ablations | `ablation_dlcm_*`, `ablation_shapley_*`, `ablation_kd_*`, training staged/joint |
| `exit_strategy_comparison` | Exit policy strategies | confidence / stability / residual-gain / random / fixed |
| `selector_ablation` | Selector design ablations | `ablation_selector_*` vs residual-gain and confidence baselines |
| `risk_coverage` | Risk–coverage / operating points | conservative / balanced / aggressive / random / oracle |
| `oracle_gap` | Gap to oracle earliest exit | `oracle_earliest_exit` vs adaptive profiles |
| `zero_shot_transfer` | Source-calibrated zero-shot transfer | `zero_shot_transfer`, `full_balanced`, `original_visualad` |

Export command:

```bash
python tools/export_paper_tables.py \
  --results artifacts/results \
  --output-dir artifacts/paper \
  --seed 111
```

Produces `{table_id}.csv`, `{table_id}.tex`, and `release_manifest.json`.
Recommended tag after a complete release manifest: `cvpr-rad-visualad-v2`.

## Hypothesis → requirement → module → test → experiment → paper artifact

| Hypothesis | Requirement | Module | Test | Experiment | Paper artifact |
|---|---|---|---|---|---|
| Official VisualAD full-depth behavior is reproducible | Baseline gate before adaptive work | `tools/reproduce_baseline.py` | `tests/rad/test_environment.py` | `original_visualad` | `main_comparison` |
| Staged ViT execution is numerically equivalent | Exact block-count / equivalence | `rad/models/checkpoint_maps.py` | `tests/rad/` staged equivalence suites | latency / adaptive eval | GPU AutoDL gate notes in CI |
| Dynamic fusion matches or beats equal fusion at full depth | Fusion gate | `rad/models/dlcm.py`, `rad/trainers/fusion_trainer.py` | `tests/rad/test_fusion_training_step.py` | `dynamic_fusion_only`, `ablation_dlcm_*` | `main_comparison`, `fusion_ablation` |
| Shapley / contribution targets improve fusion | Contribution supervision | `rad/targets/shapley.py` | contribution / fusion tests | `ablation_shapley_*` | `fusion_ablation` |
| Residual-gain LSE reduces false-safe exits vs confidence-only | Selector gate | `rad/models/lse.py`, `rad/trainers/lse_trainer.py` | `tests/rad/test_lse_training_step.py` | `ablation_selector_*`, exit strategies | `selector_ablation`, `exit_strategy_comparison` |
| Source-only calibrated policy transfers zero-shot | Zero-shot gate; no target tuning | `rad/evaluation/zero_shot.py`, `rad/evaluation/export.py` | zero-shot / export tests | `zero_shot_transfer` | `zero_shot_transfer` |
| Batch-1 measured latency improves | Latency gate (not FLOPs alone) | `rad/evaluation/benchmark.py`, `tools/benchmark_latency.py` | latency tests | benchmark matrix rows | release latency artifacts |
| Optional joint DLCM–LSE fine-tuning can help without full-depth regression | Non-primary ablation; staged gates required | `rad/trainers/joint_trainer.py`, `tools/train_joint.py` | `tests/rad/test_joint_training.py` | `ablation_training_joint` | `fusion_ablation` (+ joint summary) |
| Paper tables and provenance are reproducible | CI + release manifest | `rad/evaluation/paper_tables.py`, `tools/export_paper_tables.py` | `tests/rad/test_export_schema.py` | export dry-run / CI | all seven tables + `release_manifest.json` |

## CI vs AutoDL GPU gate

- GitHub Actions CI (`.github/workflows/ci.yml`) runs CPU pytest, Ruff, mypy, and config/CLI dry-runs.
- Numerical equivalence, CUDA latency, and full GPU training remain AutoDL gates and are documented in the CI workflow notice job.

## P0 requirement-to-evidence matrix

Machine-readable audit: [`docs/p0_manifest.json`](p0_manifest.json)
Full audit report: [`docs/p0-research-validity-audit.md`](p0-research-validity-audit.md)

| Requirement | Implementation | Test evidence | Artifact/config evidence | Status |
|---|---|---|---|---|
| **P0-1:** Baseline reproduction ordering (train → checkpoint verify → test) | `tools/reproduce_baseline.py` | `tests/rad/test_baseline_pipeline.py`, `tests/rad/test_baseline_smoke.py`, `tests/rad/contracts/baseline.py` | `configs/rad/baseline_mvtec_to_visa.yaml`, `configs/rad/baseline_visa_to_mvtec_official.yaml` | validated (CPU) |
| **P0-2:** Smoke vs real evaluation separation | `tools/evaluate_adaptive_dataset.py` (paper path); `tools/smoke_adaptive_engine.py` (dev smoke only) | `tests/rad/test_adaptive_dataset_cli.py`, `tests/rad/test_experiment_matrix.py`, `tests/rad/contracts/experiment_matrix.py` | `configs/rad/experiments.yaml` (no smoke/legacy eval CLIs) | validated |
| **P0-3:** Authoritative dataset-level metrics | `rad/evaluation/paper_metrics.py`, `rad/evaluation/dataset_evaluator.py` | `tests/rad/test_paper_metrics.py`, `tests/rad/test_dataset_evaluator.py` | `utils/metrics.py` (AUPRO fix `af06217`) | validated |
| **P0-4:** MVTec and VisA adapters/evaluator | `rad/data/adapters/`, `rad/evaluation/dataset_evaluator.py` | `tests/rad/test_dataset_adapter_registry.py`, `tests/rad/test_mvtec_adapter.py`, `tests/rad/test_visa_adapter.py`, `tests/rad/test_preprocess_contract.py` | `configs/rad/adaptive.yaml` | validated |
| **P0-5:** Fail-closed fusion training | `tools/train_fusion.py`, `rad/trainers/fusion_trainer.py` | `tests/rad/test_fusion_fail_closed.py` | `configs/rad/fusion.yaml` | validated |
| **P0-6:** Fair fixed-exit and selector-ablation matrix | `configs/rad/experiments.yaml`, `configs/rad/matrix/*` | `tests/rad/test_experiment_matrix.py`, `tests/rad/contracts/experiment_matrix.py` | matrix overlays (`fixed_exit_*`, `selector_*`) | validated |
| Residual-gain source: `sample_localization_error` | `rad/evaluation/dataset_evaluator.py` imports `rad.losses.localization` | `tests/rad/test_dataset_evaluator.py` | no AP-derived gain in evaluator | validated |
| Selector ablation: real masked LSE input | `rad/models/selector_signals.py`, `rad/inference/adaptive_engine.py` | `tests/rad/test_selector_signal_mask.py` | `SELECTOR_MASK_STAGE=post_normalization_pre_lse` | validated |
| Legacy synchronization: shared test contracts | `tests/rad/contracts/{baseline,experiment_matrix,zero_shot}.py` | `tests/rad/test_contract_helpers.py`, synced legacy tests | Increment 11 commit `bfe8409` | validated |
| Artifact hygiene: no tracked generated artifacts | `.gitignore`, CI dry-runs to `/tmp` | `tests/rad/test_artifacts_git_hygiene.py` | commit `5ef6251`; `git ls-files artifacts` empty | validated |
| Bidirectional official VisualAD baselines | `tools/reproduce_baseline.py`, `utils/metrics.py` | baseline pipeline + metrics export tests | `docs/baseline_acceptance/*.json` | accepted (records) |

## Release manifest fields

`release_manifest.json` records:

- `git_sha`, `seed`
- `configs`, `checkpoints`, `result_files`
- content `hashes`
- `environment`
- `paper_tables`
- `tag_recommendation` (`cvpr-rad-visualad-v2`)
