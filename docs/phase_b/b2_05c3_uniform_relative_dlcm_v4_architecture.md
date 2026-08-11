# B2-05C3 Uniform-Relative DLCM V4 Architecture

## Scope (C3A)

B2-05C3A freezes the **uniform-relative V4 training/deployment contract**,
production implementations, hermetic tests, artifact schemas, and **adoption**
of the existing untouched final evaluation roster from B2-05C1 (already adopted
by V3 without reselection).

End state for C3A:

- V4 contract implemented and verified
- existing untouched final roster adopted without reselection
- local annotated tag `b2-dlcm-uniform-relative-contract-v4`
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
| Development verdict | `development_unqualified` |
| Failed reason | `gt_category_kl:carpet` |
| Final roster scientific identity | `267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213` |
| Roster size | 16 (4 bottle normal + 4 bottle anomalous + 4 carpet normal + 4 carpet anomalous) |
| Overlap with original 32 | 0 |
| `final_content_resolved` | `false` |

### V3 / C2

| Field | Value |
|-------|-------|
| Contract tag | `b2-dlcm-category-robust-contract-v3` |
| Unqualified evidence tag | `b2-dlcm-category-robust-unqualified-evidence-v1` |
| Evidence HEAD | `99c26de94ba7fa5358a7670473876c4a4cf1829d` |
| Accepted V3 plan | `496e35fe8e2d5dcf208235cc58d8386cf7bdf54244d4dad32b86599e2e7104b4` |
| Environment identity | `67677c4e9bb83475f7adc03294437bdd104a693e0465e107d3860096a9f03056` |
| Development verdict | `development_unqualified` |
| Failed reason | `gt_category_kl:carpet` |
| macro GT KL | `0.07320` |
| bottle GT KL | `0.09535` |
| carpet GT KL | `0.05105` |
| carpet uniform KL | `0.04203` |
| Final roster scientific identity | `267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213` |
| `final_content_resolved` | `false` |

V1/V2/V3 outputs, schemas, identities, roster records, and negative development
results are immutable. V4 never rewrites them.

## Design thesis

C2 development failed on per-category GT KL for carpet: absolute category KL
beat uniform on macro and bottle, but carpet model KL (`0.05105`) exceeded
carpet uniform KL (`0.04203`) by more than the `1e-4` slack.

V3 Smooth-Max over absolute category means still optimizes categories that are
already far below their uniform baseline while under-penalizing categories that
lose to uniform. V4 keeps the **identical four-head decoupled architecture**
and does **not** add category as a model input. It only changes how GT
deployment supervision is aggregated and how checkpoints/seeds are selected:

1. **Category-balanced batches** — each batch is exactly `2 bottle + 2 carpet`
   (unchanged from V3).
2. **Batch-matched uniform-relative regret** — per depth/category,
   \(R_{d,c}=K_{d,c}^{model}-K_{d,c}^{uniform}\) on the **same batch and same
   GT targets**.
3. **Normalized Smooth-Max over regrets** — fixed \(\tau=0.05\); negative
   regret retained with IEEE bits (no slack subtraction, no clamp, no abs).
4. **Teacher auxiliaries unchanged** — still batch sample mean; never
   category-robust or relative.
5. **Eligibility unchanged** — macro margin and per-category slack vs
   calibration uniform (absolute KL gates).
6. **Constrained worst-relative-regret selection** — among eligible trained
   checkpoints, minimize \(\max_c[KL_c^{GT}-KL_{c,uniform}^{GT}]\).
7. **Final roster adoption only** — reuse C1 roster identity; no reselection;
   no path resolution.

## Architecture constants (identical to V2/V3)

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
- GT deployment relative-regret aggregation
- calibration checkpoint selection
- canonical seed selection
- diagnostics/reporting

## Category-balanced sampler (identical to V3)

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
- sampler contract version (`b2_dlcm_category_balanced_sampler_v1`)

Resume must reproduce the exact next-epoch batch composition and order.

## Batch-matched uniform-relative regret

For each depth \(d\) and category \(c\) on the **current batch** \(B\):

\[
K_{d,c}^{model}
=
\frac1{|B_c|}
\sum_{i\in B_c}
KL(p_{d,i}^{GT}\Vert w_{d,i}^{deploy})
\]

\[
K_{d,c}^{uniform}
=
\frac1{|B_c|}
\sum_{i\in B_c}
KL(p_{d,i}^{GT}\Vert u_d)
\]

where

\[
u_d=(1/n_d,\ldots,1/n_d)
\]

is the frozen FP32 softmax uniform baseline (identical bit pattern to the
production equal-weight / zero-logit softmax path), and

\[
R_{d,c}
=
K_{d,c}^{model}
-
K_{d,c}^{uniform}
\]

Requirements:

- model and uniform use the **same batch** and **same GT targets**
- no full-category fixed baseline
- no development/final baseline
- do **not** subtract `1e-4` slack inside the loss
- do **not** clamp negative regret
- do **not** take absolute value
- negative regret retains sign and IEEE bits
- KL uses the production exact implementation (`allocation_kl` / per-sample
  equivalent)

## Uniform-relative Smooth-Max

\[
L_{d,relative}^{GT}
=
\tau
\log
\left[
\frac{
e^{R_{d,bottle}/\tau}
+
e^{R_{d,carpet}/\tau}
}{2}
\right]
\]

Fixed:

```text
tau = 0.05
```

Stable form:

\[
m=\max(R_b,R_c)
\]

\[
L_{relative}
=
m+\tau\log
\left[
\frac{
e^{(R_b-m)/\tau}
+
e^{(R_c-m)/\tau}
}{2}
\right]
\]

Properties:

- Equal regrets ⇒ output equals the common value
- Output may be negative
- Not hard max; not hinge; not GroupDRO
- No additional category-macro KL term
- Depths 12/18/24 computed independently and equally averaged

## V4 total loss

Per depth:

\[
\mathcal L_d
=
L_{d,relative}^{GT}
+
0.25\mathcal L_{\mathrm{GTsigned},d}
+
0.25\mathcal L_{\mathrm{Talloc},d}
+
0.0625\mathcal L_{\mathrm{Tsigned},d}
\]

GT signed / teacher allocation / teacher signed remain batch sample means.
Only GT deployment allocation uses relative Smooth-Max.

\[
\mathcal L_{\mathrm{V4}}
=
\frac{\mathcal L_{12}+\mathcal L_{18}+\mathcal L_{24}}{3}
\]

Teacher weights, GT signed weight, and the absence of a macro loss term are
frozen; V4 must not modify them.

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
Loading C2/C1 checkpoints/trunk/optimizer/scheduler/RNG is forbidden.

## Calibration eligibility (unchanged absolute gates)

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

## Checkpoint selection (worst relative regret)

Among eligible trained checkpoints, choose in order:

1. Lowest worst relative regret:
   \[
   M_{\mathrm{worst}}=\max_c[KL_c^{GT}-KL_{c,uniform}^{GT}]
   \]
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
2. Lowest worst relative regret
3. Lowest category-macro GT KL
4. Lowest GT signed loss
5. Smallest seed

If no seed has a trained eligible checkpoint:

```text
canonical seed = 17
canonical checkpoint = Epoch 0
```

and Development fails closed. Teacher / development / final never participate
in seed selection.

## Data and final roster adoption

Training / calibration / development reuse accepted 16/8/8 artifacts.
Teacher and backbone are not re-run; the original 32 records are not rematerialized.

The C1 final roster is **adopted**, not rebuilt:

```text
scientific identity =
267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213
```

- bind original roster scientific identity
- bind original ordered 16 stable IDs
- bind original roster receipt
- prove `final_content_resolved=false`
- prove absence of unlock / materialization receipt / evaluation unlock /
  accepted manifest
- bind V4 implementation commit and V4 contract identity
- set `selection_reused_without_change=true`

No copy/reorder/add/delete of roster records. C3A must not resolve final paths
or read final content.

## Development / final gates (unchanged)

Depth-24 development and final gates are identical to C1/C2:

- GT macro margin `1e-5` vs uniform
- GT per-category slack `1e-4`
- Localization macro: \(\Delta\)PixelAP ≥ 0, \(\Delta\)PixelAUROC ≥ −1e−4,
  \(\Delta\)AUPRO ≥ −1e−4
- Per-category localization floor −1e−3
- Teacher diagnostics mandatory but non-blocking

C1/C2/C3 comparisons are diagnostics only. Final dual-materialization /
dual-evaluation protocols are unchanged. Untouched Final must pass before
accepted/LSE.

## Module layout

| Path | Role |
|------|------|
| `rad/phase_b/b2_dlcm_v4.py` | Model reuse of V3/V2 architecture + relative Smooth-Max loss |
| `rad/phase_b/b2_dlcm_v4_training.py` | Category sampler, eligibility, relative selection, dry-run |
| `rad/phase_b/b2_dlcm_v4_protocol.py` | V4 error codes, forbid final access, bypass rejection |
| `rad/phase_b/b2_dlcm_v4_evaluation.py` | Development/final gate evaluation (thresholds unchanged) |
| `rad/phase_b/b2_dlcm_v4_deployment.py` | Deployment export with V4 architecture pins |
| `rad/phase_b/b2_dlcm_v4_roster_adoption.py` | Adopt C1 roster without reselection |
| `tools/train_b2_dlcm_v4.py` | C3A CLI (`--dry-run` only) |
| `tools/adopt_b2_dlcm_final_roster_v4.py` | Roster adoption CLI |
| `tools/materialize_b2_dlcm_final_v4.py` | Fail-closed stub |
| `tools/evaluate_b2_dlcm_final_v4.py` | Fail-closed stub |
| `tools/verify_b2_dlcm_v4_artifacts.py` | Artifact verification |
| `configs/phase_b/b2_dlcm_uniform_relative_contract_v4.json` | C3A contract |
| `configs/phase_b/b2_dlcm_uniform_relative_official_v4.json` | C3B official stub (training disabled in C3A) |

Reuse without mutation: V1/V2/V3 KL/signed/hashing/trace/checkpoint transaction/
deterministic RNG/production metrics/fusion/final protocol primitives.

## Error codes (V4)

```text
B2_DLCM_V4_REAL_TRAINING_NOT_ENABLED
B2_DLCM_V4_CONTRACT_MISMATCH
B2_DLCM_CATEGORY_BATCH_INVALID
B2_DLCM_UNIFORM_BASELINE_INVALID
B2_DLCM_RELATIVE_REGRET_INVALID
B2_DLCM_RELATIVE_SMOOTHMAX_INVALID
B2_DLCM_NO_ELIGIBLE_CHECKPOINT
B2_DLCM_ROSTER_ADOPTION_MISMATCH
B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN
B2_DLCM_DEVELOPMENT_UNQUALIFIED
B2_DLCM_FINAL_MATERIALIZATION_MISMATCH
B2_DLCM_FINAL_EVALUATION_MISMATCH
B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN
```

Also retain `B2_DLCM_CATEGORY_COVERAGE_INVALID` for sampler/coverage fail-closed
(same semantics as V3). No bypass flags.

## C3A non-goals

- Real three-seed training
- Reading development metrics for selection
- Resolving final paths / images / masks / descriptors / maps / metrics
- Generating deployment checkpoints or accepted manifests
- Residual-gain, LSE, early-exit
- Push / remote tags / PRs
- Mutating V1/V2/V3 history, tags, evidence, or negative results
