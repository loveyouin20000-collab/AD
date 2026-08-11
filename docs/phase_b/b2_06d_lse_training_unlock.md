# B2-06D LSE Training Unlock Contract

Status: `training_unlock_created`

This unlock permits exactly the first controlled LSE training run bound to the accepted V5 DLCM artifact and the B2-06C prerequisite materialization closure.

## Bound Inputs

```text
accepted_identity = 0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116
v5_deployment_identity = c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd
H_decision = 6fb60a82d01f987930070aeee75639524512ad481064369b2f06ac99f96ae0a8
H_evidence = bbc3708a8ddcd3b2965ec9e758af1a7bf30a360cdbbc5ff86be911cfbe872e02
beta* = 0.54
```

```text
config = configs/rad/lse_b2_accepted_v5.yaml
config_sha256 = 81d6ea9887226c06c69539de97c89ea27f93af58d0cd55c655e843b5c6ad03e3
06C prerequisite evidence sha256 = e2df49b2b8d6cc9e2f4273ae0381aacd53ca87539d80f342d4525cb88d488271
accepted V5 checkpoint sha256 = 12b9192643d457eb07745391b68cfa5afe48ec6165b28091bdabde29ec3ece4f
train gain targets sha256 = 69d35e3436fda03098fe5df88d3ea8f948906c8397dc6592b33571cf794d1b0a
calibration gain targets sha256 = 1fdb3e41227690b27be578c9f94adfddf0e2b41d360da4639354cd2981e8f6b9
descriptor stats sha256 = 95c2e062c9bdb6bf37559787724e5d018e05a69164ffd703adcef629e9d16d17
```

## Run Bounds

```text
seed = 111
epochs = 30
patience = 10
train_output_dir = /root/autodl-tmp/AD-phase-b2-06d-lse-training-unlock/artifacts/checkpoints/lse/b2_06d_first_controlled_run
```

## Gate Rules

```text
--preflight-only:
  only checks accepted gate readiness
  does not require or consume training unlock
  training_started = false

--dry-run:
  requires this training unlock
  verifies identities, config hash, output dir, seed, epochs, patience
  reports readiness only
  training_started = false

real training:
  requires this training unlock
  fails closed if a prior receipt or LSE checkpoint already exists in the bound output dir
  writes an ignored training receipt after completion
```

## Boundaries

```text
Final decision unchanged = true
accepted V5 artifact unchanged = true
DLCM scientific outputs unchanged = true
push = false
PR = false
```
