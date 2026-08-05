# B3-01 Early-Exit Contract Architecture

## Scope

B3-01 introduces the early-exit contract and fail-closed preflight gate. It does
not train an exit policy, evaluate an exit policy, read Final content, generate
checkpoints, or publish remote state.

## Accepted Upstream Chain

Early exit is permitted to start only from the frozen B2 chain:

```text
accepted DLCM identity:
0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116

V5 deployment identity:
c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd

accepted LSE identity:
3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa

accepted LSE checkpoint SHA256:
e6e5a4dbd7471ef9e52430eab9533f8edda57ca76ead2ffbed034044805b1c98

B2 phase final closure identity:
2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d
```

The gate rejects any LSE checkpoint path outside the accepted LSE reference
root. A `.pt` path alone is insufficient.

## Exit Points

```text
early_depths = [12, 18]
full_depth = 24
```

Depth 24 is the mandatory fallback. B3-01 only validates this topology; it does
not learn an exit policy.

## Required Future Prerequisites

The dry-run preflight reports not-ready until these B3 inputs exist:

```text
exit_target_manifest
latency_profile
calibration_trace
```

These are future B3 materialization inputs. Missing prerequisites do not start
training and do not write artifacts.

## Failure Boundary

Fail-closed conditions include:

```text
missing accepted LSE manifest
wrong accepted DLCM identity
wrong V5 deployment identity
wrong accepted LSE identity
wrong B2 phase final closure identity
manual LSE checkpoint path outside accepted reference root
invalid early depths
full depth other than 24
```

## B3-01 Terminal State

```text
training_started = false
evaluation_started = false
final_content_accessed = false
artifact_written = false
checkpoint_generated = false
tracked .pt = 0
pushed = false
PR = false
```
