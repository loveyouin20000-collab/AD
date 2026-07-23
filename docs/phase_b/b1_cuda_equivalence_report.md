# Phase B1: Staged-Backbone CUDA Numerical Equivalence (B1-05)

**Status:** `passed`
**Detail:** `strict_independent_pass`
**Git SHA:** `b61afa98cbfdae9a7c27a1707f5cc033ca5188a7`
**Timestamp (UTC):** `2026-07-20T16:09:42.311440+00:00`
**Execution profile:** `frozen_deterministic_math`
**Raw evidence:** `artifacts/phase_b/b1_cuda_equivalence/b1_20260720T160942Z/raw_evidence.json`

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

## Backend scope (B1-05)

- Profile 1 `frozen_deterministic_math`: official/staged self-noise = 0; cross-path max = 4.77e-07.
- Profile 2 `production_default_attention`: official self max = 4.72e-05; staged self max = 6.96e-05; cross-path max = 6.63e-05 (does not materially exceed self-noise).
- Selected B2 profile: `frozen_deterministic_math` (hashed at `configs/execution/frozen_deterministic_math.json`).
- Ten fresh processes under the selected profile: identical official hashes, identical staged hashes, cross-path max = 9.54e-07.

## Dataset provenance

- Invalid predecessor: `mvtec/sample` (flat `image/` assets; not returned by `MVTecAdapter`).
- Accepted categories: `mvtec/bottle` (83 test) + `visa/candle` (200 test) via production adapters.

