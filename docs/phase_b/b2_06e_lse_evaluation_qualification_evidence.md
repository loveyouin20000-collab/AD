# B2-06E LSE Evaluation / Qualification Evidence

Status: `b2_06e_lse_evaluation_qualification_closed`

Worktree:

```text
/root/autodl-tmp/AD-phase-b2-06e-lse-evaluation-qualification
```

Branch:

```text
phase-b2-06e-lse-evaluation-qualification
```

## Decision

```text
verdict = qualified
H_lse_qualification = 0f08407ade40fb8e447649c80606fbf7d7c39f3030b99307a445f3df27688b14
accepted_artifact_generated = false
calibration_nll = 0.4768362585455179
max_calibration_nll = 0.5
evaluated_rows = 16
```

Bound identities:

```text
accepted_identity = 0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116
v5_deployment_identity = c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd
unlock_identity = 54295df82e953070a0d4f02d7526da645e6c418d638b5a2ab248b11133b19c99
training_receipt_identity = d4f569ddd79f72f84c36903abc77b91c1b48a5bafc9dd10b206a830af17b2346
```

## Artifacts

```text
LSE checkpoint = artifacts/checkpoints/lse/b2_06d_first_controlled_run/lse_best.pt
LSE checkpoint sha256 = e6e5a4dbd7471ef9e52430eab9533f8edda57ca76ead2ffbed034044805b1c98
tracked = false

evaluation report sha256 = c68668b04dcbc1e9e525e514ce9331638c2bd178a4fb02edff4913c697adc330
qualification decision sha256 = 2f02219667ecde0595fb1c483c8d7d95b7cfef5a90312ccfc36dbb62d3198184
```

## Metrics

```text
depth 12:
  n = 8
  nll = 0.7660192847251892
  mae = 0.4048725925385952
  rmse = 0.5041921386533236
  brier = 0.03221412755399544
  ece = 0.17947600036859512

depth 18:
  n = 8
  nll = 0.18765321373939514
  mae = 0.20543629676103592
  rmse = 0.2283385445394299
  brier = 0.024174126336846347
  ece = 0.15546763315796852
```

`auroc` remains undefined in the raw evaluation report because the calibration sufficiency labels for this metric path are single-class.

## Boundaries

```text
evaluation_started = true
accepted LSE artifact generated = false
accepted V5 artifact changed = false
Final decision changed = false
Final content accessed = false
tracked .pt expected = 0
push = false
PR = false
```
