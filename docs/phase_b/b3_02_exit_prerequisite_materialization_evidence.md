# B3-02 Exit Prerequisite Materialization Evidence

Status: exit prerequisites materialized locally.

This step materialized the three B3-01 preflight prerequisites:

```text
exit_target_manifest
latency_profile
calibration_trace
```

It did not train or evaluate an exit policy.

## Bound Chain

```text
accepted_dlcm_identity:
0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116

accepted_lse_identity:
3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa

accepted_lse_checkpoint_sha256:
e6e5a4dbd7471ef9e52430eab9533f8edda57ca76ead2ffbed034044805b1c98

B2 phase final closure identity:
2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d
```

## Materialized Outputs

```text
materialization_identity:
d77c2cd604c7d3eca4cc9f0649bc8e6ef1b84985a31ec470dd6dd7ef1f43e5b8

records:
16

counts_by_depth:
depth 12 = 8
depth 18 = 8

target_exits_by_depth:
depth 12 = 0
depth 18 = 0

exit_target_manifest_sha256:
ba4449d47a1acfc75d1e79720db828f8714eed296461ad4332e1e48ca58e8b3a

latency_profile_sha256:
b976189135f9908de692015cca88e312ebedb11b43d4176e09d337be73ba35f7

calibration_trace_sha256:
0f8ef910981d6081e02b026b17ee2804ad2ffcf0d2a8d13b9b26a0f8100037ce
```

The latency profile is a deterministic layer-count proxy, not a wall-clock
benchmark.

All `target_sufficient` labels in the B2-06D calibration predictions are false
at depths 12 and 18. B3-03 must account for this no-positive-exit target
distribution before training.

## Post-Materialization Preflight

```text
accepted_gate_passed = true
ready = true
missing_prerequisites = []
training_started = false
evaluation_started = false
final_content_accessed = false
artifact_written = false
```

## Boundary

```text
exit_policy_training_started = false
exit_policy_evaluation_started = false
final_content_accessed = false
checkpoint_generated = false
tracked .pt = 0
pushed = false
PR = false
```
