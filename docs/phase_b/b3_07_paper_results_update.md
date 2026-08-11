# B3-07 Paper Results Update

Status: paper_results_update_frozen_locally

## Paper Position

| Component | Status | Position |
|---|---|---|
| DLCM dynamic layer fusion | accepted | primary sample-adaptive fusion mechanism |
| LSE layer-sufficiency validation | qualified | supporting validation |
| Early-exit | negative result | limitation and future work; full-depth fallback retained |

## Result Summary

```text
beta_star_decimal = 0.54
lse_calibration_nll = 0.4768362585455179
lse_required_depths = [12, 18]
adaptive_weight_calibration_records = 8
adaptive_weight_max_sample_linf_delta_from_uniform = 0.15734508633613586
early_exit_candidate_depths = [12, 18]
early_exit_fallback_depth = 24
early_exit_positive_signal_count = 0
```

The accepted system uses DLCM V5 sample-adaptive layer fusion and retains the
qualified LSE validation. The conservative early-exit contract produced no legal
positive exit signal, so it is not an accepted mechanism and the full-depth
fallback remains in force.

## Bound Identities

```text
accepted_dlcm_identity = 0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116
v5_deployment_identity = c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd
accepted_lse_identity = 3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa
b2_phase_final_closure_identity = 2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d
b3_06_phase_closure_identity = a984814c1821dbc6c0b2ee49fbf018be0c8b4f2fe226855f6b3e015eb89e05be
b4_01_weight_evidence_identity = 68bcea45e1fe98ffbee9f9ea51a2b645916b4a623198f787ce8830b1b0f8fe79
b4_02_final_release_identity = 296191577c12aa42e2e4dbad3d34deaef67b04bbd34d3d0f52be20b9e1c99b93
update_identity = a8195a1f553b3a6f4524b119f49b849dee7ae2b134d6b327d55f74fa771a4aa0
```
