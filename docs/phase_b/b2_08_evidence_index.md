# B2-08 Evidence Index

Status: local paper/evidence index closure.

This index maps each paper-facing claim to tracked evidence. Ignored run
artifacts and checkpoints are referenced only through frozen identities and
SHA256 values.

## Core Claims

| Claim | Evidence |
|---|---|
| DLCM uses layers `[6, 12, 18, 24]` and depths `[12, 18, 24]`. | `docs/phase_b/b2_05a_dlcm_training_architecture.md`; `docs/phase_b/b2_05c4_uniform_anchored_dlcm_v5_architecture.md` |
| V5 uses uniform-anchored beta calibration. | `docs/phase_b/b2_05c4_uniform_anchored_dlcm_v5_architecture.md` |
| `beta*=0.54` is frozen. | `docs/phase_b/b2_06a_lse_accepted_gate_preflight_evidence.json`; `docs/phase_b/b2_06b_accepted_v5_reference_packaging_evidence.json` |
| Accepted DLCM identity is frozen. | `docs/phase_b/b2_06a_lse_accepted_gate_preflight_evidence.json`; `docs/phase_b/b2_07_phase_final_closure_manifest.json` |
| Accepted V5 checkpoint SHA is frozen and not tracked as `.pt`. | `docs/phase_b/b2_06b_accepted_v5_reference_packaging_evidence.json`; `docs/phase_b/b2_07_phase_final_closure_manifest.json` |
| LSE entrypoint is accepted-gate bound. | `docs/phase_b/b2_06a_lse_accepted_gate_preflight_evidence.json` |
| LSE prerequisites were materialized before training. | `docs/phase_b/b2_06c_lse_prerequisite_materialization_evidence.json` |
| LSE first controlled run completed. | `docs/phase_b/b2_06d_lse_first_controlled_run_evidence.json` |
| LSE qualified. | `docs/phase_b/b2_06e_lse_qualification_decision_manifest.json`; `docs/phase_b/b2_06e_lse_evaluation_qualification_evidence.json` |
| Accepted LSE artifact is frozen. | `docs/phase_b/b2_06f_accepted_lse_manifest.json`; `docs/phase_b/b2_06f_accepted_lse_closure_receipt.json`; `docs/phase_b/b2_06f_accepted_lse_closure_evidence.json` |
| B2 phase is locally closed. | `docs/phase_b/b2_07_phase_final_closure_manifest.json`; `docs/phase_b/b2_07_phase_final_closure_report.md` |

## Evidence Files To Cite In Appendix

| File | Purpose |
|---|---|
| `docs/phase_b/b2_05a_dlcm_training_architecture.md` | DLCM model/loss/deployment contract. |
| `docs/phase_b/b2_05c4_uniform_anchored_dlcm_v5_architecture.md` | V5 uniform-anchored calibration contract. |
| `docs/phase_b/b2_06a_lse_accepted_gate_preflight_evidence.json` | Accepted V5 identities and LSE gate wiring evidence. |
| `docs/phase_b/b2_06b_accepted_v5_reference_packaging_evidence.json` | Accepted V5 reference packaging and checkpoint SHA. |
| `docs/phase_b/b2_06c_lse_prerequisite_materialization_evidence.json` | LSE prerequisite materialization evidence. |
| `docs/phase_b/b2_06d_lse_first_controlled_run_evidence.json` | LSE controlled training run evidence. |
| `docs/phase_b/b2_06e_lse_qualification_decision_manifest.json` | LSE qualification decision and metric table. |
| `docs/phase_b/b2_06f_accepted_lse_manifest.json` | Accepted LSE artifact manifest. |
| `docs/phase_b/b2_06f_accepted_lse_closure_evidence.json` | Accepted LSE closure evidence. |
| `docs/phase_b/b2_07_phase_final_closure_manifest.json` | End-to-end B2 phase closure manifest. |
| `docs/phase_b/b2_08_paper_results_summary.md` | Paper-ready result summary. |
| `docs/phase_b/b2_08_paper_results_manifest.json` | Machine-readable B2-08 index manifest. |

## Non-Cite Boundary Notes

The earlier B2-05C4 report records a historical fail-closed point before the
Final tooling repair and accepted V5 closure. For paper-facing final status, use
the accepted identities carried by B2-06A, B2-06B, and B2-07 rather than treating
that historical fail-closed point as the terminal experiment state.

```text
B2-08 does not start training.
B2-08 does not start evaluation.
B2-08 does not read Final content.
B2-08 does not generate model artifacts.
B2-08 does not push or open a PR.
```
