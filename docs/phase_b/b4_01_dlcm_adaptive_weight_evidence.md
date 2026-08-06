# B4-01 DLCM Adaptive Weight Evidence

Status: dlcm_adaptive_weight_evidence_frozen

## Result

Accepted V5 deployment weights were exported on the Calibration split without
training, Final access, or model artifact generation.

```text
sample_adaptive_variation_observed = true
uniform_equivalent_at_tolerance = false
calibration_records = 8
beta_star_decimal = 0.54
```

## Deployment Weight Statistics

```text
layer_means = [0.2541907746344805, 0.30299320816993713, 0.24509595148265362, 0.1977201197296381]
layer_stds = [0.04115799713475293, 0.049581725873028955, 0.03284324573504816, 0.03041907663531247]
max_layer_std = 0.049581725873028955
mean_sample_linf_delta_from_uniform = 0.06814451888203621
max_sample_linf_delta_from_uniform = 0.15734508633613586
rows_non_uniform_at_tolerance = 8
```

## Dynamic Head Statistics

```text
layer_means = [0.2577606812119484, 0.348135557025671, 0.24091841839253902, 0.15318538807332516]
layer_stds = [0.07621850645265565, 0.09181800363318886, 0.06082082869810331, 0.05633162059739318]
max_layer_std = 0.09181800363318886
mean_sample_linf_delta_from_uniform = 0.12619354482740164
max_sample_linf_delta_from_uniform = 0.29137974977493286
rows_non_uniform_at_tolerance = 8
```

## Paper Interpretation

The accepted V5 deployment is not merely a fixed equal-weight fusion at the
selected tolerance. Its deployment weights retain sample-level variation after
the uniform anchor beta is applied. This supports the paper claim that DLCM is
a sample-adaptive layer fusion mechanism, while keeping the claim limited to
the accepted Calibration-split evidence.

## Boundary

```text
training_started = false
evaluation_started = false
final_content_accessed = false
model_artifact_generated = false
tracked_pt_files = 0
```

Weight evidence identity:

```text
68bcea45e1fe98ffbee9f9ea51a2b645916b4a623198f787ce8830b1b0f8fe79
```
