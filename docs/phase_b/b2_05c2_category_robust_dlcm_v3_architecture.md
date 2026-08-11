# B2-05C2 Category-Robust DLCM V3 Architecture

## Scope (C2A)

B2-05C2A freezes the **category-robust V3 training/deployment contract**,
production implementations, hermetic tests, artifact schemas, and **adoption**
of the existing untouched final evaluation roster from B2-05C1.

End state for C2A:

- V3 contract implemented and verified
- existing C1 final roster adopted without reselection
- local annotated tag `b2-dlcm-category-robust-contract-v3`
- real three-seed training not started
- final content not resolved or materialized
- residual-gain / LSE / early-exit not started
- no push, no remote tag, no PR

## Historical immutability

### V1

| Field | Value |
|-------|-------|
| Local tag | `b2-dlcm-unqualified-evidence-v1` |
| Commit | `43d856f5ff771957f9f39d0909b1bc87d6b7081b` |
| Verdict | `localized_but_target_fidelity_unqualified` |

V1 schemas, evidence, checkpoints, manifests, tag, and qualification conclusion
are immutable.

### V2 / C1

| Field | Value |
|-------|-------|
| Contract tag | `b2-dlcm-decoupled-contract-v2` |
| Tag target | `e54f2b44eeb962b05cfb7cf74764e55905f1a8f6` |
| C1B branch HEAD | `1044b885e86cff4c7f4e1a635f0ebc417105e854` |
| Development verdict | `development_unqualified` |
| Failed reason | `gt_category_kl:carpet` |
| Final roster scientific identity | `267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213` |
| Roster size | 16 (4 bottle normal + 4 bottle anomalous + 4 carpet normal + 4 carpet anomalous) |
| Overlap with original 32 | 0 |
| `final_content_resolved` | `false` |

V2 outputs, schemas, identities, roster records, and the C1 negative development
result are immutable. V3 never rewrites them.

## Design thesis

C1 development failed on per-category GT KL for carpet under batch-mean GT
deployment loss. V3 keeps the **identical four-head decoupled architecture**
and does **not** add category as a model input. It only changes how GT
deployment supervision is aggregated and how checkpoints/seeds are selected:

1. **Category-balanced batches** — each batch is exactly `2 bottle + 2 carpet`.
2. **Smooth-Max GT deployment loss** — per-depth category means, then
   normalized Smooth-Max with fixed `τ=0.05`.
3. **Teacher auxiliaries unchanged** — still batch sample mean; never
   category-robust.
4. **Eligibility + worst-category checkpoint selection** — only trained
   checkpoints that beat uniform on macro and every category may become best.
5. **Final roster adoption only** — reuse C1 roster identity; no reselection;
   no path resolution.

## Architecture constants (identical to V2)

```text
candidate layers = [6,12,18,24]
prediction depths = [12,18,24]
descriptor dimension = 18
layer embedding = 8
depth embedding = 8
hidden dimension = 64
DeterministicDropout = 0.1
```

Training heads (unchanged):

1. GT deployment allocation head
2. Teacher auxiliary allocation head
3. GT signed auxiliary head
4. Teacher signed auxiliary head

Formal deployment retains **shared trunk + GT deployment head only**.

### Category must not enter

- descriptor, embeddings, trunk, head
- deployment checkpoint, inference wrapper, golden tests

### Category may enter

- training batch construction
- GT deployment loss aggregation
- calibration checkpoint selection
- canonical seed selection
- diagnostics/reporting

## Category-balanced sampler

Batch size remains 4. Training set is 8 bottle + 8 carpet records.

Each epoch:

1. Independent deterministic permutation of the 8 bottle training IDs.
2. Independent deterministic permutation of the 8 carpet training IDs.
3. Each category is split into 4 contiguous pairs (2 samples each).
4. Assemble 4 batches as `2 bottle + 2 carpet`.
5. Each training record appears exactly once per epoch.
6. Batch order is controlled by a third independent deterministic generator.
7. No normal/anomalous balancing; no sample weights.
8. Missing either category in a batch is fail-closed.

Persisted resume state:

- bottle permutation generator state
- carpet permutation generator state
- batch-order generator state
- epoch index
- sampler contract version

Resume must reproduce the exact next-epoch batch composition and order.

## Smooth-Max GT deployment loss

Per depth and category, sample-mean GT deployment KL:

\[
L_{d,c}^{GT}
=
\frac{1}{N_c}
\sum_{i\in c}
KL(p_{d,i}^{GT}\Vert w_{d,i}^{deploy})
\]

with \(N_{\mathrm{bottle}}=N_{\mathrm{carpet}}=2\) per batch.

Normalized Smooth-Max with \(\tau=0.05\):

\[
L_{d,\mathrm{robust}}^{GT}
=
\tau\log
\left[
\frac{
\exp(L_{d,\mathrm{bottle}}^{GT}/\tau)
+
\exp(L_{d,\mathrm{carpet}}^{GT}/\tau)
}{2}
\right]
\]

Numerically stable form:

\[
m=\max(L_b,L_c),\quad
L_{\mathrm{robust}}
=
m+\tau\log
\left[
\frac{\exp((L_b-m)/\tau)+\exp((L_c-m)/\tau)}{2}
\right]
\]

Properties:

- Equal category losses ⇒ robust loss equals the common value.
- Not hard max; not GroupDRO; no mutable category weights persisted.
- Computed independently per depth; three depths equally averaged.

## V3 total loss

Per depth:

\[
\mathcal L_d
=
L_{d,\mathrm{robust}}^{GT}
+
0.25\mathcal L_{\mathrm{signed},d}^{GT}
+
0.25\mathcal L_{\mathrm{Talloc},d}
+
0.0625\mathcal L_{\mathrm{signed},d}^{T}
\]

GT signed / teacher allocation / teacher signed remain batch sample means.
Only GT deployment allocation uses category Smooth-Max.

\[
\mathcal L_{\mathrm{V3}}
=
\frac{\mathcal L_{12}+\mathcal L_{18}+\mathcal L_{24}}{3}
\]

Loss coefficients are identical to V2.

## Training hyperparameters (unchanged)

```text
seeds = [17,29,43]
batch size = 4
maximum epochs = 500
patience = 50
min_delta = 1e-5
AdamW; max LR 3e-4; min LR 3e-6; weight decay 1e-3
betas = [0.9,0.999]; epsilon = 1e-8
maximum optimizer steps = 2000; warmup = 100; grad clip = 1.0
FP32; AMP=false; TF32=false; single GPU; strict deterministic
```

All seeds initialize from scratch on CPU-derived deterministic seeds.
Loading C1 checkpoints/trunk/optimizer/scheduler/RNG is forbidden.

## Calibration eligibility

Depth-24 calibration GT metrics (macro and per-category) determine eligibility.

Macro:

\[
KL_{\mathrm{macro}}^{GT}
\le
KL_{\mathrm{uniform,macro}}^{GT}
-
10^{-5}
\]

Per category:

\[
KL_c^{GT}
\le
KL_{\mathrm{uniform},c}^{GT}
+
10^{-4}
\]

A checkpoint is eligible only if macro and every category pass.
Epoch 0 is the initial best but is **not** a trained eligible checkpoint.
If no trained checkpoint is eligible, Epoch 0 remains best.

## Checkpoint selection

Among eligible trained checkpoints, choose in order:

1. Lowest worst-category GT KL: \(M_{\mathrm{worst}}=\max_c KL_c^{GT}\)
2. Within `1e-5`, lowest category-macro GT KL
3. Within `1e-5`, lowest GT signed loss
4. Complete ties keep the earlier epoch

Patience resets only when a new eligible best is produced.
Ineligible checkpoints never replace best and never reset patience, but still
record full metrics. Early stop after 50 complete epochs without a new eligible
best.

## Canonical seed selection

Across seeds, using each seed’s best checkpoint:

1. Eligible trained status beats Epoch 0 fallback
2. Lowest worst-category GT KL
3. Lowest category-macro GT KL
4. Lowest GT signed loss
5. Smallest seed

If no seed has a trained eligible checkpoint:

```text
canonical seed = 17
canonical checkpoint = Epoch 0
```

Teacher / development / final never participate in seed selection.

## Data and final roster adoption

Training / calibration / development reuse C1’s accepted 16/8/8 artifacts.
Teacher and backbone are not re-run; the original 32 records are not rematerialized.

The C1 final roster is **adopted**, not rebuilt:

- bind original roster scientific identity
- bind original ordered 16 stable IDs
- bind original roster receipt
- prove `final_content_resolved=false`
- prove absence of unlock / materialization receipt / evaluation unlock /
  accepted manifest
- bind V3 implementation commit and V3 contract identity
- set `selection_reused_without_change=true`

No copy/reorder/add/delete of roster records.

## Development / final gates (unchanged)

Depth-24 development and final gates are identical to C1:

- GT macro margin `1e-5` vs uniform
- GT per-category slack `1e-4`
- Localization macro: \(\Delta\)PixelAP ≥ 0, \(\Delta\)PixelAUROC ≥ −1e−4,
  \(\Delta\)AUPRO ≥ −1e−4
- Per-category localization floor −1e−3
- Teacher diagnostics mandatory but non-blocking

No “must beat C1” qualification gate. C1/C2 comparisons are diagnostics only.

## Module layout

| Path | Role |
|------|------|
| `rad/phase_b/b2_dlcm_v3.py` | Model reuse of V2 architecture + Smooth-Max loss |
| `rad/phase_b/b2_dlcm_v3_training.py` | Category sampler, eligibility, selection, dry-run |
| `rad/phase_b/b2_dlcm_v3_protocol.py` | V3 error codes, forbid final access, bypass rejection |
| `rad/phase_b/b2_dlcm_v3_evaluation.py` | Development/final gate evaluation (thresholds unchanged) |
| `rad/phase_b/b2_dlcm_v3_deployment.py` | Deployment export with V3 architecture pins |
| `rad/phase_b/b2_dlcm_v3_roster_adoption.py` | Adopt C1 roster without reselection |
| `tools/train_b2_dlcm_v3.py` | C2A CLI (`--dry-run` only) |
| `tools/adopt_b2_dlcm_final_roster_v3.py` | Roster adoption CLI |
| `tools/materialize_b2_dlcm_final_v3.py` | Fail-closed stub |
| `tools/evaluate_b2_dlcm_final_v3.py` | Fail-closed stub |
| `tools/verify_b2_dlcm_v3_artifacts.py` | Artifact verification |
| `configs/phase_b/b2_dlcm_category_robust_contract_v3.json` | C2A contract |
| `configs/phase_b/b2_dlcm_category_robust_official_v3.json` | C2B official stub (training disabled in C2A) |

Reuse without mutation: V1/V2 KL/signed/hashing/trace/checkpoint transaction/
deterministic RNG/production metrics/fusion/final protocol primitives.

## Error codes (V3)

```text
B2_DLCM_V3_REAL_TRAINING_NOT_ENABLED
B2_DLCM_V3_CONTRACT_MISMATCH
B2_DLCM_CATEGORY_BATCH_INVALID
B2_DLCM_CATEGORY_COVERAGE_INVALID
B2_DLCM_SMOOTHMAX_INVALID
B2_DLCM_NO_ELIGIBLE_CHECKPOINT
B2_DLCM_ROSTER_ADOPTION_MISMATCH
B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN
B2_DLCM_DEVELOPMENT_UNQUALIFIED
B2_DLCM_FINAL_MATERIALIZATION_MISMATCH
B2_DLCM_FINAL_EVALUATION_MISMATCH
B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN
```

No bypass flags.

## C2A non-goals

- Real three-seed training
- Reading development metrics for selection
- Resolving final paths / images / masks / descriptors / maps / metrics
- Generating deployment checkpoints or accepted manifests
- Residual-gain, LSE, early-exit
- Push / remote tags / PRs
