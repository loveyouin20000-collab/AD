# B3-01 Early-Exit Contract Implementation Plan

Goal: introduce a fail-closed early-exit contract and dry-run preflight bound to
the accepted DLCM V5 and accepted LSE artifacts.

Scope:

- no early-exit training
- no early-exit evaluation
- no Final content access
- no checkpoint generation
- no push or PR

Outputs:

- `configs/rad/early_exit_b3_accepted_lse.yaml`
- `rad/phase_b/b3_early_exit_gate.py`
- `tools/preflight_early_exit_b3.py`
- `tests/rad/test_b3_early_exit_gate.py`
- B3-01 architecture/evidence docs under `docs/phase_b`

Contract:

- accepted DLCM V5 identity must match B2 frozen identity
- accepted LSE identity must match B2-06F frozen identity
- B2 phase final closure identity must match B2-07
- early depths are `[12, 18]`
- full fallback depth is `24`
- dry-run reports readiness only
- missing policy targets/latency profile/calibration traces fail readiness but
  do not start training
