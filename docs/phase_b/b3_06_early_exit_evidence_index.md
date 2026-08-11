# B3-06 Early-Exit Evidence Index

Status: early_exit_phase_closed_negative_result.

This index maps paper-facing early-exit claims to tracked evidence. Ignored
runtime artifacts remain referenced only through identities and SHA256 values.

## Claims

| Claim | Evidence |
|---|---|
| Early-exit was wired behind the accepted DLCM/LSE chain. | `docs/phase_b/b3_01_early_exit_preflight_evidence.json` |
| Exit targets, latency proxy, and calibration trace were materialized before any training. | `docs/phase_b/b3_02_exit_prerequisite_materialization_evidence.json` |
| No positive exit targets were available under the first target materialization. | `docs/phase_b/b3_03_exit_policy_training_contract_evidence.json` |
| The conservative positive-signal redefinition still produced zero legal positives. | `docs/phase_b/b3_04_exit_target_positive_signal_contract_evidence.json` |
| The early-exit line is closed as a negative result with full-depth fallback. | `docs/phase_b/b3_05_early_exit_negative_result_evidence.json`; `docs/phase_b/b3_06_early_exit_phase_closure_manifest.json` |
| Dynamic fusion and LSE are not abandoned by the early-exit negative result. | `docs/phase_b/b3_06_early_exit_paper_results_summary.md` |

## Evidence Files

| File | Purpose |
|---|---|
| `docs/phase_b/b3_01_early_exit_contract_architecture.md` | B3 early-exit evidence. |
| `docs/phase_b/b3_01_early_exit_preflight_evidence.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_02_exit_prerequisite_materialization_manifest.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_02_exit_prerequisite_materialization_evidence.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_03_exit_policy_training_contract.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_03_exit_policy_training_contract_evidence.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_04_exit_target_positive_signal_contract.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_04_exit_target_positive_signal_contract_evidence.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_05_early_exit_line_closure_manifest.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_05_early_exit_negative_result_evidence.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_06_early_exit_phase_closure_manifest.json` | B3 early-exit evidence. |
| `docs/phase_b/b3_06_early_exit_paper_results_summary.md` | B3 early-exit evidence. |
| `docs/phase_b/b3_06_early_exit_evidence_index.md` | B3 early-exit evidence. |

## Boundary

```text
B3-06 does not start training.
B3-06 does not start evaluation.
B3-06 does not read Final content.
B3-06 does not generate model artifacts.
B3-06 does not push or open a PR.
```
