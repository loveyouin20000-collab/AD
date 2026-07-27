# Phase B1: Staged-Backbone CUDA Numerical Equivalence (B1-05)

**Status:** `passed`
**Detail:** `strict_independent_pass`
**Strict status:** `strict_independent_pass`
**Git SHA:** `fcff2d4419f64bfea43874b48579c6dde60625f7`
**Timestamp (UTC):** `2026-07-27T08:16:51.818944+00:00`
**Execution profile:** `frozen_deterministic_math`
**Profile SHA-256:** `7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d`
**Raw evidence:** `artifacts/phase_b/b1_cuda_equivalence/b1_20260727T081651Z/raw_evidence.json` (sha256=`0d9a3772de411ec9ce79d29c1d412c0b80736fdec3088d5e432bd6c194f70fac`)
**Ten-process:** `True`

## Layer coverage

- Official candidate layers tested: `[6, 12, 18, 24]`
- Synthetic candidate layers tested: `[2, 4, 6, 8]`
- Nonstandard official run validated: `False`

## Why earlier ~1e-4 appeared

Independent CUDA forwards under the default attention backend exhibit a runtime nondeterminism floor near 1e-4. The frozen deterministic math SDP profile (CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic algorithms, TF32 off, flash/mem-efficient SDP off, math SDP on, MHA fastpath off) removes that floor for official self-noise and staged self-noise.

## Deterministic control

- Decision: `A` (strict_independent_pass)
- Deterministic error: `None`

- **A1_vs_A2**: max_feature=0.00e+00, max_map=0.00e+00
- **S1_vs_S2**: max_feature=0.00e+00, max_map=0.00e+00
- **A1_vs_S1**: max_feature=0.00e+00, max_map=9.54e-07
- **A2_vs_S2**: max_feature=0.00e+00, max_map=9.54e-07

## Gate summaries

- Gate 1 same-chain: `passed`
- Gate 2 operational envelope: `passed` (self_p95=0.00e+00, cross_p95=9.54e-07, excess=9.54e-07)
- Task-level: `passed` on 283 samples

- image_auroc_pp: `0.0000` pp
- image_ap_pp: `0.0000` pp
- image_f1_max_pp: `0.0000` pp
- pixel_auroc_pp: `0.0000` pp
- pixel_ap_pp: `0.0000` pp
- pixel_f1_max_pp: `0.0000` pp
- pixel_aupro_pp: `0.0000` pp
- boundary_f_score_pp: `0.0000` pp

## Limitations

- Independent-pass 1e-5 is valid only under frozen_deterministic_math.
- Latency figures are diagnostic sanity checks only, not paper benchmarks.
- Nonstandard layer set [2,4,6,8] validated on tiny synthetic CUDA model only.
- Checkpoint path and SHA-256 are required CLI arguments; no hidden fallback paths.
- mvtec/sample fixture evidence is retained historically but invalid for task gate.
