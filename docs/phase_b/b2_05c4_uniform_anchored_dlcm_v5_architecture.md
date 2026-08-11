# B2-05C4 Uniform-Anchored DLCM V5 Architecture

## Scope (C4A)

B2-05C4A freezes the **uniform-anchored V5 calibration contract**, production
implementations, hermetic tests, artifact schemas, and **adoption** of the
existing untouched final evaluation roster (scientific identity unchanged from
C1/C2/C3).

End state for C4A:

- V5 calibration contract implemented and verified
- existing untouched final roster adopted without reselection
- local annotated tag `b2-dlcm-uniform-anchored-contract-v5`
- real DLCM training not started / not re-run
- Development metrics not read for beta selection
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

### V2 / C1

| Field | Value |
|-------|-------|
| Contract tag | `b2-dlcm-decoupled-contract-v2` |
| Tag target | `e54f2b44eeb962b05cfb7cf74764e55905f1a8f6` |
| Development verdict | `development_unqualified` |
| Failed reason | `gt_category_kl:carpet` |
| Final roster scientific identity | `267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213` |
| Roster size | 16 (4+4 bottle, 4+4 carpet); overlap with original 32 = 0 |
| `final_content_resolved` | `false` |

### V3 / C2

| Field | Value |
|-------|-------|
| Unqualified evidence tag | `b2-dlcm-category-robust-unqualified-evidence-v1` |
| Evidence HEAD | `99c26de94ba7fa5358a7670473876c4a4cf1829d` |
| Development verdict | `development_unqualified` |
| Failed reason | `gt_category_kl:carpet` |

### V4 / C3

| Field | Value |
|-------|-------|
| Contract tag | `b2-dlcm-uniform-relative-contract-v4` |
| Unqualified evidence tag | `b2-dlcm-uniform-relative-unqualified-evidence-v1` |
| Evidence HEAD | `a1447bdabdd7f54eb7883b717dfadc3da906da5b` |
| Accepted V4 plan | `4979c73a28e0aaffd21f2c6408bb37e90fdc64201bcc326f990543fbbee5650f` |
| Environment identity | `67677c4e9bb83475f7adc03294437bdd104a693e0465e107d3860096a9f03056` |
| Canonical seed | `17` |
| Canonical reproduction | `passed` |
| Development verdict | `development_unqualified` |
| Failed reason | `gt_category_kl:carpet` |
| macro GT KL | `0.08349` |
| bottle GT KL | `0.11900` |
| bottle uniform KL | `0.27789` |
| carpet GT KL | `0.04795` |
| carpet uniform KL | `0.04203` |
| carpet relative regret | `+0.00592` |
| Final roster scientific identity | `267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213` |
| `final_content_resolved` | `false` |

V1/V2/V3/V4 schemas, evidence, checkpoints, manifests, tags, training
identities, and negative development results are **immutable**. V5 never
rewrites them and never re-trains the DLCM.

## Design thesis

C3 development failed on per-category GT KL for carpet: macro and bottle beat
uniform, but carpet model KL exceeded carpet uniform KL by more than the
`1e-4` slack (`+0.00592` relative regret).

V5 does **not** change loss, sampler, selector, trunk, heads, normalization, or
canonical checkpoint. It freezes a **post-hoc global scalar calibration** that
convex-combines depth-matched uniform weights with the frozen C3 dynamic
deployment weights:

\[
w_d^{\mathrm{deploy}}(x;\beta)=(1-\beta)\,u_d+\beta\,\widetilde{w}_d(x)
\]

with one global \(\beta\in[0,1]\) shared across prediction depths 12/18/24.

Beta is selected **only on Calibration** via Depth-24 leave-one-out
worst-category relative regret among eligibility-passing grid candidates.
Development remains a one-shot go/no-go with **unchanged** gates. If Development
fails again, C4 terminates the current DLCM repair line (no C5, no LSE).

## Architecture constants (identical to V4/V3/V2)

```text
candidate layers = [6,12,18,24]
prediction depths = [12,18,24]
descriptor dimension = 18
layer embedding = 8
depth embedding = 8
hidden dimension = 64
DeterministicDropout = 0.1
canonical seed = 17
```

Formal deployment retains **shared trunk + GT deployment head only**, plus a
frozen scalar \(\beta\) in the production wrapper. Category never enters the
model, checkpoint, or wrapper forward.

Teacher diagnostics remain mandatory but non-blocking, sourced from the
canonical best training checkpoint, **not** beta-mixed, and never enter beta
objective or qualification.

## Uniform-anchored deployment weights

Let \(\widetilde{w}_d(x)\) be the C3 dynamic softmax weights at depth \(d\),
and \(u_d=(1/n_d,\ldots,1/n_d)\) the depth-matched uniform.

Requirements:

1. \(\beta=1\) exactly reproduces C3 deployment weights (FP32 bits).
2. \(\beta=0\) exactly equals depth-matched uniform.
3. Convex combination computed in FP32.
4. Output non-negative and sums to 1 (renormalize only if needed for numerical
   safety; preferred path is exact convex combo of already-normalized vectors).
5. Do not modify C3 logits, head, trunk, or checkpoint tensors.
6. Do not absorb \(\beta\) into logits bias.
7. Category does not enter model or wrapper.
8. Production wrapper only adds frozen scalar \(\beta\).

## Beta grid

```text
beta = 0.00, 0.01, ..., 1.00   (101 candidates)
integer grid index i ∈ {0..100}
beta = i / 100.0
```

Canonical JSON stores both `beta_index` and exact decimal string
(`"0.00"` … `"1.00"`). No continuous optimizer. Grid is immutable. Development
and Final never select beta.

## Calibration split and LOO objective

Use the original Calibration 8 records only (4 bottle + 4 carpet). Forbidden:
Development, Final, C1/C2/C3 Development metrics, Final roster content.

For each \(\beta\), category \(c\), and leave-one-out index \(i\in c\):

\[
R_{c,-i}(\beta)=\frac13\sum_{j\in c,\,j\neq i}
\left[
\mathrm{KL}(p_j^{\mathrm{GT}}\Vert w_j(\beta))
-
\mathrm{KL}(p_j^{\mathrm{GT}}\Vert u)
\right]
\]

\[
M_{\mathrm{LOO}}(\beta)=\max_{c,i} R_{c,-i}(\beta)
\]

Requirements:

- Depth 24 only for the LOO objective
- 4 folds per category → 8 regrets
- model/uniform use identical records and GT targets
- production exact KL
- negative regret allowed (no slack subtraction, no clamp, no abs)
- no other weighting; no Training/Development/Final in selection

## Candidate eligibility and selection

On the **full** Calibration set, each beta must pass:

\[
\mathrm{KL}_{\mathrm{macro}}^{\mathrm{GT}}(\beta)
\le
\mathrm{KL}_{\mathrm{uniform,macro}}^{\mathrm{GT}}
-
10^{-5}
\]

and per category:

\[
\mathrm{KL}_{c}^{\mathrm{GT}}(\beta)
\le
\mathrm{KL}_{c,\mathrm{uniform}}^{\mathrm{GT}}
+
10^{-4}
\]

Only eligible candidates compete. \(\beta=0\) has no special fallback. Empty
eligible set → fail-closed (`B2_DLCM_V5_NO_ELIGIBLE_BETA`). Carpet slack and
macro threshold are not relaxed.

Among eligible candidates, select by:

1. lowest \(M_{\mathrm{LOO}}\)
2. within `1e-5` tie → larger beta
3. still tied → lower full-Calibration macro GT KL
4. still tied → smaller grid index

Selection manifest records all 101 candidates' eligibility, LOO objective, and
macro/category metrics.

## Calibration A/B dual process

Two independent processes load the same C3 deployment candidate and Calibration
records from disk, evaluate the 101-grid, apply eligibility + LOO selection, and
write canonical calibration manifests. Required equality:

- candidate metrics
- eligible set
- \(\beta^\star\)
- canonical JSON byte-equality
- scientific identity

No shared cache, model instance, predictions, or in-memory state.

## Development / Final gates (unchanged)

Depth-24 GT macro margin `1e-5`, per-category slack `1e-4`, localization macro
floors (`ΔPixelAP≥0`, `ΔPixelAUROC≥-1e-4`, `ΔAUPRO≥-1e-4`), per-category
localization floor `-1e-3`. Teacher diagnostics mandatory but non-blocking.
No new gates; no relaxations.

Final dual materialization / dual evaluation semantics unchanged from C1–C3.
Untouched Final must pass before accepted/LSE.

## C4 termination strategy

If V5 Development fails (`development_unqualified`):

- Final remains untouched
- do not enter C5
- do not retune loss, sampler, selector, beta grid, LOO, or tie-break
- freeze conclusion: current 18-D descriptor + 16-record training/calibration
  contract cannot stably satisfy the Carpet per-category GT target gate
- next step returns to descriptor sufficiency / training coverage / target
  variance / category-generalization protocol
- LSE not started

## Final roster adoption

Adopt the same roster scientific identity:

```text
267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213
```

Manifest binds original roster identity/receipt, ordered stable IDs,
`final_content_resolved=false`, no unlock/receipt/accepted,
V5 implementation commit, V5 contract identity,
`selection_reused_without_change=true`. No reselection, reorder, add, or delete.

## Identities

\[
H_{\mathrm{deploy,V5}}
=
H(
H_{\mathrm{deploy,V4}},
\beta^\star,
\text{calibration contract},
\text{Calibration A/B identity}
)
\]

- `H_decision` binds untouched Final GT + localization only
- `H_evidence` binds Calibration A/B, beta grid/eligibility/selection,
  Development, teacher diagnostics, Materialization/Evaluation A/B,
  C1–C4 provenance
- `H_accepted` binds deploy/decision/evidence/upstream/V5 contract
- Development never enters decision

## Module map (C4A)

```text
rad/phase_b/b2_dlcm_v5.py
rad/phase_b/b2_dlcm_v5_calibration.py
rad/phase_b/b2_dlcm_v5_protocol.py
rad/phase_b/b2_dlcm_v5_evaluation.py
rad/phase_b/b2_dlcm_v5_deployment.py
rad/phase_b/b2_dlcm_v5_roster_adoption.py

tools/calibrate_b2_dlcm_v5.py
tools/adopt_b2_dlcm_final_roster_v5.py
tools/materialize_b2_dlcm_final_v5.py
tools/evaluate_b2_dlcm_final_v5.py
tools/verify_b2_dlcm_v5_artifacts.py

configs/phase_b/b2_dlcm_uniform_anchored_contract_v5.json
configs/phase_b/b2_dlcm_uniform_anchored_official_v5.json
```

## Error codes

```text
B2_DLCM_V5_CONTRACT_MISMATCH
B2_DLCM_V5_TRAINING_FORBIDDEN
B2_DLCM_V5_BETA_GRID_INVALID
B2_DLCM_V5_CALIBRATION_INPUT_INVALID
B2_DLCM_V5_NO_ELIGIBLE_BETA
B2_DLCM_V5_CALIBRATION_MISMATCH
B2_DLCM_V5_BETA_SELECTION_INVALID
B2_DLCM_V5_ROSTER_ADOPTION_MISMATCH
B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN
B2_DLCM_DEVELOPMENT_UNQUALIFIED
B2_DLCM_FINAL_MATERIALIZATION_MISMATCH
B2_DLCM_FINAL_EVALUATION_MISMATCH
B2_DLCM_ACCEPTED_MANIFEST_FORBIDDEN
```

No bypass flags.

## C4A non-goals

- Re-training DLCM
- Reading Development metrics for beta selection
- Resolving/materializing Final content
- Generating accepted manifests
- Starting residual-gain / LSE / early-exit
- Mutating C3 canonical seed, checkpoint, trunk, heads, normalization, or
  training identities
- Push / remote tags / PRs
