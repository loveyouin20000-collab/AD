# B2-06D LSE First Controlled Run Evidence

Status: `b2_06d_first_controlled_lse_training_completed`

Worktree:

```text
/root/autodl-tmp/AD-phase-b2-06d-lse-training-unlock
```

Branch:

```text
phase-b2-06d-lse-training-unlock
```

Training code commit:

```text
df60fd81d33d5186b963e3155d55c9dce3b8fbde
```

## Unlock

```text
unlock_identity = 54295df82e953070a0d4f02d7526da645e6c418d638b5a2ab248b11133b19c99
config_sha256 = 81d6ea9887226c06c69539de97c89ea27f93af58d0cd55c655e843b5c6ad03e3
accepted_identity = 0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116
v5_deployment_identity = c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd
```

Dry-run gate passed before training:

```text
ready = true
accepted_gate_passed = true
missing_prerequisites = []
training_started = false
```

## Training Result

```text
output_dir = artifacts/checkpoints/lse/b2_06d_first_controlled_run
seed = 111
epochs_ran = 30
train_rows = 32
calibration_rows = 16
best_cal_nll = 0.4768362585455179
best_epoch = 22
```

Best checkpoint:

```text
path = artifacts/checkpoints/lse/b2_06d_first_controlled_run/lse_best.pt
sha256 = e6e5a4dbd7471ef9e52430eab9533f8edda57ca76ead2ffbed034044805b1c98
tracked = false
```

Training receipt:

```text
path = artifacts/checkpoints/lse/b2_06d_first_controlled_run/b2_06d_lse_training_receipt.json
sha256 = 07aa2f2ddffc31dab2c242cbba1b233e0acd430f8f0b4cacf81d716ebfbc9634
receipt_identity = d4f569ddd79f72f84c36903abc77b91c1b48a5bafc9dd10b206a830af17b2346
```

Calibration metrics at best:

```text
depth 12: nll = 0.7660192847251892, mae = 0.4048725925385952, ece = 0.17947600036859512
depth 18: nll = 0.18765321373939514, mae = 0.20543629676103592, ece = 0.15546763315796852
combined nll = 0.4768362585455179
```

`auroc` is `NaN` in both depth metrics because the calibration target labels available to this LSE metric path are not class-diverse enough for AUROC.

## Boundaries

```text
Final content accessed = false
accepted V5 artifact changed = false
Final decision changed = false
LSE training started = true
LSE checkpoint generated = true
LSE checkpoint tracked = false
push = false
PR = false
```

Post-run single-use check:

```text
command = tools/train_lse.py --config configs/rad/lse_b2_accepted_v5.yaml --dry-run --device cuda:0
exit_code = 1
fail_closed_code = B2_LSE_TRAINING_UNLOCK_ALREADY_CONSUMED
reason = training receipt already exists
```
