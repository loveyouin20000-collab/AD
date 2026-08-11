# B2-05C1 Decoupled DLCM V2 Architecture

## Scope (C1A)

B2-05C1A freezes the **decoupled V2 training/deployment contract**, production
implementations, hermetic tests, artifact schemas, and an **untouched public
final evaluation roster**.

End state for C1A:

- V2 contract implemented and verified
- untouched 16-record final roster frozen locally
- local annotated tag `b2-dlcm-decoupled-contract-v2`
- real three-seed training not started
- final content not resolved or materialized
- residual-gain / LSE / early-exit not started
- no push, no remote tag, no PR

## V1 immutability

V1 unqualified evidence remains frozen and must never be rewritten:

| Field | Value |
|-------|-------|
| Historical worktree | `/root/autodl-tmp/AD-phase-b2-dlcm-canonical-training` |
| Historical branch | `phase-b2-dlcm-canonical-training` |
| Historical HEAD / tag target | `43d856f5ff771957f9f39d0909b1bc87d6b7081b` |
| Local tag | `b2-dlcm-unqualified-evidence-v1` |
| Accepted training plan | `59e20f4cb337ef42384f70bb8b3dad5211d906341b0a2d41f7e6847610635980` |
| Seed collection | `94a6a9332a0694889c7a0255814ac13fe8316c601529197063165ce14ec1277f` |
| Canonical seed | `17` |
| Deployment scientific identity | `4cbc6fb88f39ed86deacfbbe48580f7682453b94becb046ec6ef1b1302df378a` |
| Evaluation unlock | `19dca41e9f647d12afce9877a7340f5af58bf9a23997d7339dded26d89fe73dd` |
| Qualification scientific identity | `da51e5fc1302cf507bc844f87e82cb66f7d2fa0a13e61f28a0dba14333201c49` |
| V1 verdict | `localized_but_target_fidelity_unqualified` |

V1 schemas, evidence, checkpoints, manifests, tag, and qualification conclusion
are immutable. V2 lives on an isolated branch/worktree and never mutates V1
artifacts.

## Design thesis

V1 mixed GT and teacher positive allocations into one shared deployment head.
That coupling produced localization-capable but target-fidelity-unqualified
deployment weights. V2 **decouples**:

1. **GT-only deployment supervision** — the only allocation head that survives
   into the deployment artifact.
2. **Teacher allocation as training-only auxiliary** — separate head, separate
   KL, never exported.
3. **GT/teacher signed heads as training-only auxiliaries** — deleted from
   deployment; diagnostics require the canonical best training checkpoint.
4. **Development as go/no-go only** — never enters `H_decision`.
5. **Untouched final evaluation** — sole qualification source; roster contains
   identities only (no paths) until an explicit later unlock.

## Architecture constants

Configuration-driven (never hard-coded as magic fours inside reusable APIs
beyond reading config):

```text
candidate layers = [6,12,18,24]
prediction depths = [12,18,24]
depth 12 players = [6,12]
depth 18 players = [6,12,18]
depth 24 players = [6,12,18,24]
descriptor dimension = 18
layer embedding = 8
depth embedding = 8
hidden dimension = 64
DeterministicDropout = 0.1
layer encoder = 34 → 64 → 64
context encoder = 192 → 64 → 64
```

Hidden block: `Linear → LayerNorm → GELU → DeterministicDropout`.

Mean/Max aggregate **only along the player dimension**. No batch communication.

## Four training outputs (per depth)

From shared trunk features \(h_{d,l}\):

1. **GT deployment allocation head** \(W_{GT}, b_{GT}\)
2. **Teacher auxiliary allocation head** \(W_T, b_T\)
3. **GT signed auxiliary head**
4. **Teacher signed auxiliary head**

\[
w_{d,l}^{deploy}=\operatorname{softmax}_l(W_{GT}h_{d,l}+b_{GT})
\]

\[
w_{d,l}^{T,aux}=\operatorname{softmax}_l(W_Th_{d,l}+b_T)
\]

plus \(\hat\phi_{d,l}^{GT}\) and \(\hat\phi_{d,l}^{T}\).

Formal deployment artifact retains **shared trunk + GT deployment head only**.
Teacher allocation, GT signed, and teacher signed heads are deleted. Production
wrappers must not expose auxiliary outputs.

## Initialization

Reuse V1 CPU initialization, SHA-256 seed derivation, 63-bit generators, four
independent Dropout streams, CPU→GPU bit-exact move, and RNG persistence.

Both allocation heads are zero-initialized (`weight=0`, `bias=0`), so Epoch 0
outputs are exactly uniform for both GT deploy and teacher aux. Embeddings,
shared trunk, and signed heads follow V1 Xavier/normal rules. The two
allocation heads are independent parameters — never shared, bound, or copied
during training.

## Losses

GT deployment:

\[
\mathcal L_{\mathrm{GTdeploy},d}=KL(p_d^{GT}\Vert w_d^{deploy})
\]

Teacher allocation acts only on the teacher auxiliary head:

\[
\mathcal L_{\mathrm{Talloc},d}=KL(p_d^T\Vert w_d^{T,aux})
\]

KL is V1-exact: target-weighted, `log_softmax`, natural log, zero-target terms
exactly 0, no epsilon, no smoothing.

Signed:

\[
\mathcal L_{\mathrm{signed},d}^{GT}=L_{\mathrm{Huber},d}^{GT}+0.25L_{\mathrm{rank},d}^{GT}
\]

\[
\mathcal L_{\mathrm{signed},d}^{T}=L_{\mathrm{Huber},d}^{T}+0.25L_{\mathrm{rank},d}^{T}
\]

Fixed: Huber \(\delta=1.0\), ranking tie tolerance \(10^{-6}\), ranking margin 0.

Per-depth total:

\[
\mathcal L_d=
\mathcal L_{\mathrm{GTdeploy},d}
+0.25\mathcal L_{\mathrm{signed},d}^{GT}
+0.25\mathcal L_{\mathrm{Talloc},d}
+0.0625\mathcal L_{\mathrm{signed},d}^{T}
\]

Total loss equals the equal-weight mean over depths `{12,18,24}`. Teacher
auxiliary weight `0.25` is frozen (no search).

## Gradient boundaries

Verified with actual `.grad` probes:

| Loss | Updates |
|------|---------|
| GT deployment KL | shared trunk + GT deployment head |
| Teacher allocation KL | shared trunk + teacher allocation head |
| GT signed | shared trunk + GT signed head |
| Teacher signed | shared trunk + teacher signed head |

Teacher losses must not directly update the GT deployment head. GT deployment
must not update the three auxiliary heads. Shared trunk is never detached.

## Training hyperparameters (unchanged from V1)

```text
seeds = [17,29,43]
batch size = 4
max epochs = 500
patience = 50
min_delta = 1e-5
AdamW
max LR = 3e-4
min LR = 3e-6
weight decay = 1e-3
betas = [0.9,0.999]
epsilon = 1e-8
max optimizer steps = 2000
warmup steps = 100
gradient clip = 1.0
FP32; AMP=false; TF32=false
single GPU; strict deterministic
OMP/MKL/torch threads as V1; PYTHONHASHSEED=0; CUBLAS=:4096:8
```

No capacity, Dropout, optimizer, LR, stage, or V1 warm-start changes.

## Checkpoint / canonical selection

Calibration primary (GT only):

\[
\frac18\sum_i\frac13\sum_d KL(p_{d,i}^{GT}\Vert w_{d,i}^{deploy})
\]

Secondary (GT signed only):

\[
\frac18\sum_i\frac13\sum_d\mathcal L_{\mathrm{signed},d,i}^{GT}
\]

Teacher metrics must be recorded and finite but must never affect best,
patience, or canonical seed.

Epoch 0 may be replaced only by GT primary improvement beyond `1e-5`. After
training, compare GT primary first, then GT signed secondary; exact ties keep
the earlier checkpoint. Cross-seed: GT primary/secondary only; exact ties take
smallest seed; all-epoch-0 fallback is seed 17.

## Data lifecycle

Reuse the original 32-sample lifecycle without rematerialization:

```text
training = original 16
calibration = original 8
development = original evaluation 8
```

Reuse descriptors, normalization, GT calibration, dual-family targets, Shapley
normalization, maps, and masks. Do not re-invoke teacher/backbone for these 32.

Original evaluation is permanently labeled:

```text
development_evaluation
used_for_B2_05B_qualification_and_postmortem
```

## Final roster (untouched)

Allowed categories: `bottle`, `carpet`. Per category: `4 normal + 4 anomalous`
(16 total). Zero overlap with the original 32 stable IDs.

Deterministic selection from a verified source master manifest only:

1. Verify manifest/receipt
2. Exclude original 32 stable IDs
3. Group by category + label
4. Sort each group by ascending stable ID
5. Take the first 4 per group
6. Fail-closed if any group has fewer than 4

Forbidden: directory scans for selection, filename inference, shrinking the
roster, adding other categories, reusing old samples, or viewing images to pick.

Public roster fields only:

```text
stable_sample_id
category
normal_or_anomalous
source_record_scientific_sha256
source_manifest_scientific_sha256
selection_rank
```

No paths, filenames, URIs, directories, data values, or results.

Tracked files:

```text
docs/phase_b/b2_05c1_final_evaluation_roster.json
docs/phase_b/b2_05c1_final_evaluation_roster.json.sha256
```

Roster identity binds implementation commit, source manifest, 32-ID exclusion
coverage, selection rule, and ordered 16 records. C1A must not write
`final_evaluation_resolution.json`.

## Development / final gates

Development runs only after three-seed training, collection, selection,
reproduction, and deployment candidate export.

Depth-24 blocking gates (identical for development and final):

\[
KL_{\mathrm{DLCM}}^{GT,macro}\le KL_{\mathrm{uniform}}^{GT,macro}-10^{-5}
\]

Per category:

\[
KL_{\mathrm{DLCM},c}^{GT}\le KL_{\mathrm{uniform},c}^{GT}+10^{-4}
\]

Localization macro:

\[
\Delta PixelAP\ge0,\quad
\Delta PixelAUROC\ge-10^{-4},\quad
\Delta AUPRO\ge-10^{-4}
\]

Per-category localization deltas must not fall below `-1e-3` on any of the three
metrics. Teacher diagnostics are mandatory but non-blocking.

Development failure blocks final unlock, path resolution, materialization,
accepted identity, and LSE.

## Auxiliary diagnostics

Per split/depth report teacher allocation KL/JSD/Top-1/Spearman and GT/teacher
signed Huber/ranking/Top-1/Spearman. Sole source:

```text
canonical_best_training_checkpoint
```

Marked:

```text
diagnostic_source = canonical_best_training_checkpoint
not_available_from_deployment_artifact = true
qualification_blocking = false
```

Deployment weights must never proxy signed outputs. Independent artifacts:

```text
auxiliary_diagnostics_manifest.json
auxiliary_diagnostics_manifest.json.sha256
```

Missing/wrong-source/non-finite diagnostics fail-close the evaluation artifact;
numeric quality does not affect qualification.

## Identity layering

- `H_deploy` — V2 deployment architecture/state, normalization, golden, upstream
- `H_decision` — untouched-final GT target-learning, localization, thresholds,
  verdict only (no development, no teacher)
- `H_evidence` — decision + development go/no-go + aux diagnostics +
  materialization/evaluation A/B + production metric proof + coverage/provenance
- \(H_{\mathrm{accepted}}=H(H_{\mathrm{deploy}},H_{\mathrm{decision}},H_{\mathrm{evidence}},
  H_{\mathrm{selection}},\text{upstream},\text{V2 contract})\)

Accepted identity is forbidden until final passes.

## Final materialization / evaluation dual-run

After development go and explicit unlocks (later stages):

- Materialization A/B must be bit-equal; failure cleans partials; partials are
  never reused
- Evaluation A/B must agree on decision and evidence hashes
- Unlock receipts are single-use and verified from disk (no CLI bypass)

## Reuse contract

V2 reuses V1 canonical hashing, trace, optimizer/scheduler, atomic persistence,
production verification/metrics/fusion, environment, and RNG semantics. V2 must
not fork or alter production metrics/fusion semantics. V1 modules remain
available for historical equivalence; V2 modules are versioned (`*_v2`).

## Scope exclusions (C1A)

- Real three-seed training
- Reading real development metrics
- Resolving final paths or reading final images/masks
- Generating final descriptors, teacher maps, contribution targets, anomaly maps,
  or metrics
- Generating real deployment checkpoints or accepted manifests
- Residual-gain / LSE / early-exit
- Push / remote tags / PRs
