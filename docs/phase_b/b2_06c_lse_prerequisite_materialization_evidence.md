# B2-06C LSE Prerequisite Materialization Evidence

Status: `b2_06c_lse_prerequisite_materialization_closed`

Worktree:

```text
/root/autodl-tmp/AD-phase-b2-06c-lse-prerequisite-materialization
```

Branch:

```text
phase-b2-06c-lse-prerequisite-materialization
```

## Accepted Gate

`tools/train_lse.py --config configs/rad/lse_b2_accepted_v5.yaml --preflight-only --device cuda:0` returned:

```text
ready = true
accepted_gate_passed = true
missing_prerequisites = []
training_started = false
```

Bound identities:

```text
accepted_identity = 0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116
v5_deployment_identity = c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd
H_decision = 6fb60a82d01f987930070aeee75639524512ad481064369b2f06ac99f96ae0a8
H_evidence = bbc3708a8ddcd3b2965ec9e758af1a7bf30a360cdbbc5ff86be911cfbe872e02
```

Accepted V5 checkpoint:

```text
/root/autodl-tmp/AD-phase-b2-dlcm-v5-final-execution/artifacts/phase_b/b2_dlcm_v5_final_evaluation/final-evaluation-20260805-155023/accepted_refs/canonical_deployment_candidate_v5.pt
sha256 = 12b9192643d457eb07745391b68cfa5afe48ec6165b28091bdabde29ec3ece4f
beta* = 0.54
```

## Materialized Prerequisites

Teacher cache:

```text
path = artifacts/cache/mvtec_teacher_b2_06c_all
manifest_sha256 = aad4c9a6c1139e10e7dcfebaa65a34a3ecc96449e54374fbfb64044349183ce9
cache_scientific_sha256 = 66d23807e868696a9c4a68ad83399d82df3d33e743a97d97eeb98ac60c0b1b0a
sample_coverage_sha256 = 6e538b902795c377f9992258e307e58b5c0ba0f99cbbe6c3853a81947ca3d76c
counts = training 16, calibration 8, evaluation 8
```

LSE cache conversion:

```text
receipt = artifacts/cache/b2_06c_lse_cache_conversion_receipt.json
receipt_sha256 = 31a4001e6d1f7d7b78498fa50537a06722273b6fbb4cec8192f521169bbf980a
receipt_identity = 9c28d8a2268cf86f329e471f040c6d91804aba646d685b9e64b27adb23d43050
training_count = 16
calibration_count = 8
evaluation_ignored_count = 8
training_started = false
lse_checkpoint_generated = false
```

Descriptor stats:

```text
path = artifacts/stats/mvtec_seed111.json
sha256 = 95c2e062c9bdb6bf37559787724e5d018e05a69164ffd703adcef629e9d16d17
```

Gain targets:

```text
train = artifacts/targets/gain/mvtec_train.pt
train_sha256 = 69d35e3436fda03098fe5df88d3ea8f948906c8397dc6592b33571cf794d1b0a
train_records = 16

calibration = artifacts/targets/gain/mvtec_calibration.pt
calibration_sha256 = 1fdb3e41227690b27be578c9f94adfddf0e2b41d360da4639354cd2981e8f6b9
calibration_records = 8
```

## Boundaries

```text
LSE training started = false
LSE checkpoint generated = false
accepted V5 artifact changed = false
Final decision changed = false
push = false
PR = false
tracked .pt expected = 0
```
