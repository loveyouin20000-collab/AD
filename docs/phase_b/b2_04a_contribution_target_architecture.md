# B2-04A Contribution-Target Architecture

## Status

Contract-only increment. Official materialization is disabled. Real Shapley
targets, DLCM training, residual-gain, LSE, and early-exit policy artifacts are
out of scope.

Base: annotated tag `b2-descriptor-tokenize-v1` at
`bf68a7ba546603535356cfc3222b2bfe9a0b35f8`.

## Dual-target rationale

Every sample receives two independent target families:

- `gt_localization` — localization utility against the production GT mask
- `teacher_fidelity` — fidelity to the verified full-depth teacher reference

Families remain separate in the artifact. DLCM may combine their losses later.

## Depth-local cooperative games

| Depth | Players | Coalitions |
|------:|---------|-----------:|
| 12 | `[6,12]` | 4 |
| 18 | `[6,12,18]` | 8 |
| 24 | `[6,12,18,24]` | 16 |

Do not truncate depth-24 Shapley values to obtain shallow targets. Do not reuse
a candidate-layer map across checkpoint depths.

## Coalition encoding and fusion

Players ordered by ascending layer ID. Local bit `i` maps to player position
`i`. Coalitions ordered by integer bitmask ascending; bitmask `0` is empty.

For nonempty coalition \(S\):

\[
A_S=\frac{1}{|S|}\sum_{l\in S}A_{l|d}
\]

Empty coalition: \(A_\varnothing=0\) with authoritative map shape/dtype.
Coalition fusion uses equal average, never production sum-preserving fusion.

## GT map calibration

Source-training-only robust monotonic calibration, independent per depth:

- depth 12: 16 training × 3 nonempty coalitions
- depth 18: 16 training × 7 nonempty coalitions
- depth 24: 16 training × 15 nonempty coalitions

Nearest-rank ceiling quantiles:

\[
Q(q)=x_{\max(1,\lceil qN\rceil)}
\]

Freeze \(q_{\mathrm{low}}=Q(0.01)\), \(q_{\mathrm{high}}=Q(0.995)\). Map:

\[
\hat A_S=\operatorname{clip}\Big(\frac{A_S-q_{\mathrm{low}}}{q_{\mathrm{high}}-q_{\mathrm{low}}},0,1\Big)
\]

Fail closed if \(q_{\mathrm{high}}\le q_{\mathrm{low}}\). Persist calibration as
`float64`.

## GT mask contract

Masks come from the production adapter/preprocess path. Align to anomaly-map
shape with nearest-neighbor only. Binarize \(M>0.5\). Normal masks must be
all-zero; anomalous masks must contain anomaly and background pixels.

## Frozen GT utilities

### Abnormal

Pixel AP on raw \(A_S\) via production `_binary_ap`. Soft Dice on calibrated
\(\hat A_S\) with \(\varepsilon=10^{-6}\) and linear denominator. Background
penalty mixes Top-1% mean (0.7) and global background mean (0.3) over
\(K=\max(1,\lceil 0.01|\bar M|\rceil)\) pixels with row-major tie-break.

\[
U_{\mathrm{GT}}^{\mathrm{anom}}(S)=0.4\,\mathrm{PixelAP}+0.4\,\mathrm{SoftDice}-0.2\,P_{\mathrm{BG}}
\]

No clipping.

### Normal

\[
U_{\mathrm{GT}}^{\mathrm{norm}}(S)=1-\big(0.7\,\mathrm{Top1\%Mean}+0.3\,\mathrm{GlobalMean}\big)
\]

over \(K=\max(1,\lceil 0.01HW\rceil)\) calibrated pixels. No Pixel AP or Dice.

## Teacher reference

Reconstruct full-depth teacher from \(A_{6|24},A_{12|24},A_{18|24},A_{24|24}\)
via production `sum_preserving_fusion` with equal valid weights. Require
bit-exact equality with cached `full_depth_map` (shape, dtype, layer order,
every element). All depths 12/18/24 share this verified reference.

## Teacher utility

Raw maps only; no GT calibration or masks:

\[
U_T(S)=0.5\frac{\rho_{\mathrm{Spearman}}(A_S,T)+1}{2}+0.5\,\mathrm{Top1\%Overlap}(A_S,T)
\]

Spearman degeneracy: both constant → raw=1; exactly one constant → raw=0;
otherwise average-rank Spearman. Top-1% overlap is intersection-over-K with
stable descending sort and row-major ties.

## Empty-coalition centering and exact Shapley

Natural utilities first. Center \(v(S)=U(S)-U(\varnothing)\) per family.
Require \(v(\varnothing)=0\). Exact enumeration Shapley in `float64` with
efficiency residual \(\le 10^{-12}\).

## Allocation

Positive players (\(\phi_l>\tau\), \(\tau=10^{-12}\)) renormalize to simplex.
All-nonpositive: minimum-harm equal-ties among max \(\phi\) within `1e-12`.
Allocation finite, \(\ge 0\), sum \(=1\) within `1e-12`.

## Leakage boundary

32 samples retain original split membership (16/8/8). GT calibration and
Shapley normalization use training only. Training-access helpers reject
calibration/evaluation records. Evaluation targets only after DLCM freeze.

## Scientific identity graph

```text
GT calibration
→ 32 target records
→ target collection (+ split coverage hashes)
→ Shapley normalization
→ contribution_plan_scientific_sha256
```

Seven layered identities:

1. `gt_map_calibration_scientific_sha256`
2. `contribution_target_sample_coverage_sha256`
3. `contribution_target_collection_scientific_sha256`
4. `shapley_normalization_scientific_sha256`
5. `training_target_coverage_sha256`
6. `calibration_target_coverage_sha256`
7. `evaluation_target_coverage_sha256`

Explicit whitelists exclude paths, timestamps, Git/worktree, runtime
attestation, and file-byte hashes from scientific digests.

## Artifact layout

```text
run_dir/
├── records/<stable_sample_id>.pt × 32
├── gt_map_calibration.pt
├── shapley_normalization.pt
├── final_manifest.json
└── final_manifest.json.sha256
```

Dual-hash `.pt` payloads: scientific record + embedded scientific hash; file
SHA only in the final manifest. Fresh-run only; collisions fail closed.

## Dry-run / official plan binding

Dry-run computes the complete plan in memory, returns all seven identities and
`contribution_plan_scientific_sha256`, writes nothing, `teacher_forward_count=0`.

Official materialization (future B2-04B) requires
`--expected-plan-sha256` and exact recomputed plan match before any write.
B2-04A config sets `official_materialization_enabled: false`.

## Upstream binding

Each target binds accepted teacher cache and descriptor collection on stable
ID, split, layers/depths, and all scientific identities. Targets are computed
from teacher-cache maps; descriptor records are the future DLCM feature
anchor and are not recomputed.

## B2-04B handoff

Future dual runs:

- Target Run A: teacher cache A + descriptor A
- Target Run B: teacher cache A + descriptor B

Both independently fit GT calibration and Shapley normalization. Teacher
cache Run B remains an external equivalence control.

## Scope exclusions

No real contribution-target artifacts, no Shapley from real teacher cache, no
DLCM/LSE/residual-gain/policy training, no teacher/backbone rerun, no VisA /
target-domain access.
