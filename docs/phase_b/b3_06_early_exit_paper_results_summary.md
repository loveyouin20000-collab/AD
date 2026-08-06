# B3-06 Early-Exit Paper Results Summary

Status: early_exit_phase_closed_negative_result

## Result Table

| Mechanism | Candidate depths | Positive exit targets | Positive signals | Accepted mechanism | Final behavior |
|---|---:|---:|---:|---|---|
| Early-exit policy | [12, 18] | 0 | 0 | no | full_depth_fallback |

## Paper-Ready Result

Early-exit was evaluated as a downstream efficiency extension after the DLCM
and LSE artifacts had been accepted. Under the conservative positive-signal
contract, neither the materialized exit targets nor the LSE-derived calibration
trace produced a legal positive early-exit signal at depths 12 or 18. Therefore
early-exit policy training was not unlocked, no early-exit checkpoint was
generated, and the accepted system retains full-depth inference.

## Interpretation

This result does not remove the paper's dynamic fusion contribution. DLCM and
LSE remain the accepted mechanisms; early-exit is reported as a negative result
and future-work direction under the current conservative gate.

## Bound Identities

```text
accepted_dlcm_identity = 0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116
accepted_lse_identity = 3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa
b2_phase_final_closure_identity = 2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d
b3_05_line_closure_identity = f281a3bda75d723a45f8942934c7c4d131e3424d63ba65125d7b6d2cb4ad7cb1
b3_06_phase_closure_identity = a984814c1821dbc6c0b2ee49fbf018be0c8b4f2fe226855f6b3e015eb89e05be
```

## Boundary

```text
training_started_in_b3_06 = false
evaluation_started_in_b3_06 = false
final_content_accessed_in_b3_06 = false
model_artifact_generated_in_b3_06 = false
tracked_pt_files = 0
```
