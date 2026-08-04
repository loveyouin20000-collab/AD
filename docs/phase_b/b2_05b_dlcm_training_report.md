# B2-05B Canonical DLCM Training Report

## Status

- Qualification state: `localized_but_target_fidelity_unqualified`
- Deployment qualified: `false`
- Local only: not pushed, not merged, LSE not started

## Contract freeze

- Tag: `b2-dlcm-training-contract-v1`
- Commit: `b580715f3dbfce3e4a03fe4073a57f99e8027f25`
- Accepted plan SHA: `59e20f4cb337ef42384f70bb8b3dad5211d906341b0a2d41f7e6847610635980`

## Environment

- Environment contract SHA-256: `31141df74c1dc8e3d3963dc4f7ccc91714fc903dae7097b0edbcaeb4804641a0`
- Visible GPU count: 1; AMP false; TF32 false; deterministic true

## Seeds (calibration)

| Seed | Best epoch | Primary | Secondary | Best model identity |
|------|------------|---------|-----------|---------------------|
| 17 | 229 | 0.05810313186763475 | 0.4197793699180087 | `2f4a451fa89ee21fc579b8f0f984e2fd4f2950231740bb40332a168a507e473b` |
| 29 | 265 | 0.0638079974645128 | 0.38491790244976676 | `e242f231439cfa068d1f45a57da6eef10df17e222684266c96a83cda0132e251` |
| 43 | 276 | 0.06474217403835307 | 0.37629919545724994 | `318ba58c61b2ae154a79e756a0e9f6911d6774d8ea316ccb72cd0561bbe389a2` |

- Primary mean/std (ddof=0): {'mean': 0.06221776779016688, 'std': 0.0029343759187740763}
- Secondary mean/std (ddof=0): {'mean': 0.39366548927500844, 'std': 0.01879754592552551}
- Seed collection: `94a6a9332a0694889c7a0255814ac13fe8316c601529197063165ce14ec1277f`
- Canonical seed: **17** (lowest primary)
- Selection: `e3bc06dfa02d6109544648020680d907bf0fce5ed7a093372d74009f9e69e142`
- Canonical reproduction: **passed** (nodes equal)

## Deployment

- Deployment scientific SHA-256: `4cbc6fb88f39ed86deacfbbe48580f7682453b94becb046ec6ef1b1302df378a`
- CPU golden: passed; GPU numerical: passed via loader; batch independence B=1,2,4: passed
- Formal loader without accepted manifest: rejected

## Evaluation / qualification

- Unlock: `19dca41e9f647d12afce9877a7340f5af58bf9a23997d7339dded26d89fe73dd`
- Evaluation manifest: `9a782a5589084f4e6059889c7a550dbe0db514463f0fcc994ca00686b74db1c8`
- Production metric invocation proof: {"all_compute_paper_metrics": true, "all_spearman_fidelity": true, "all_top1_overlap": true, "invocation_count": 6}
- Depth-24 gates: localization **passed**; target-learning **failed** (`macro_kl_gate`, category bottle/carpet)
- Qualification: `da51e5fc1302cf507bc844f87e82cb66f7d2fa0a13e61f28a0dba14333201c49`
- Accepted identity: none (unqualified candidate preserved)

## Identity repair

Seed manifests initially bound last-epoch model identities. Repaired to best-checkpoint identities (`identity_repair=best_checkpoint_v1`) and rebound unlock/eval/qual scientific hashes without changing sealed checkpoints or metric tensors.

## Scope exclusions

No residual-gain / LSE / early-exit; no VisA or target-domain access; no teacher/backbone rerun; raw run directory untracked; nothing pushed.

## B2-05C0 follow-up (diagnosis only)

Signed-diagnostic proxy using deployment weights has been removed from the
evaluation code path. Qualification metrics and scientific identities above are
unchanged. See `b2_05c0_signed_diagnostics_boundary.md` and
`b2_05c0_target_conflict_diagnosis.md`.

