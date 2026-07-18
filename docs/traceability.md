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

## Release manifest fields

`release_manifest.json` records:

- `git_sha`, `seed`
- `configs`, `checkpoints`, `result_files`
- content `hashes`
- `environment`
- `paper_tables`
- `tag_recommendation` (`cvpr-rad-visualad-v2`)
