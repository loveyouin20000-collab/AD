# B2-05A DLCM Training Architecture

## Scope

B2-05A freezes the **DLCM training and deployment contract** only.

End state for this increment:

- B2-05A DLCM training and deployment contract implemented
- real DLCM training not started
- evaluation not unlocked
- residual-gain / LSE / early-exit not started
- B2-05B not started

Authoritative base: tag `b2-contribution-target-artifacts-v1` at commit
`97a4f497f6f2b096dd4a339555f81e7296ec3035`.

## Model

Candidate layers `[6,12,18,24]` and prediction depths `[12,18,24]` are
configuration-driven. Players at depth `d` are layers `≤ d`.

Shared trunk:

1. Concatenate standardized 18-D descriptors with 8-D layer and 8-D depth embeddings (34-D).
2. Layer encoder `34→64→64` with `Linear→LayerNorm→GELU→DeterministicDropout(0.1)`.
3. Set context `g_d = [Mean; Max]` over players only → 128-D; concat with player features → 192-D.
4. Context encoder `192→64→64` with the same block structure.
5. Heads from shared 64-D features:
   - deployment `64→1` (softmax over players, temperature 1, no entropy regularizer)
   - GT signed `64→1` (training only)
   - teacher signed `64→1` (training only)

Auxiliary heads are deleted from the deployment artifact.

## Dual-target rationale

GT and teacher positive allocations supervise the shared deployment weights with
equal family weight `1/2`. Separate signed heads learn standardized Shapley
values without mixing families into one target artifact.

## Losses

Allocation KL (no ε, no smoothing; zero targets contribute 0):

\[
D_{\mathrm{KL}}(p\|w)=\sum_{l:p_l>0}p_l[\log p_l-\log w_l]
\]

Per depth: \(\mathcal L_{\mathrm{alloc},d}=\tfrac12 D_{\mathrm{KL}}(p^{GT}\|w)+\tfrac12 D_{\mathrm{KL}}(p^{T}\|w)\).

Signed: Huber(\(\delta=1\)) + \(0.25\) zero-margin softplus ranking (tie tolerance \(10^{-6}\)).

Total:

\[
\mathcal L_d=\mathcal L_{\mathrm{alloc},d}+0.25\cdot\tfrac{\mathcal L_{\mathrm{signed},d}^{GT}+\mathcal L_{\mathrm{signed},d}^{T}}{2},\quad
\mathcal L=\tfrac{\mathcal L_{12}+\mathcal L_{18}+\mathcal L_{24}}{3}
\]

## Gradient boundaries

- Allocation → shared trunk + deployment head
- GT signed → shared trunk + GT head
- Teacher signed → shared trunk + teacher head
- Signed losses never update the deployment head; allocation never updates signed heads

## Initialization

CPU-only init from a dedicated generator derived from the model seed. Explicit
module order; deployment head zeroed last so epoch-0 weights are exact uniform
softmax. CPU→CUDA move requires bit-exact round-trip identity.

## Deterministic RNG

Versioned SHA-256 → 63-bit seeds for `model_initialization`, `sampler`,
`dropout`, `dataloader`. Four independent DeterministicDropout site generators.
No `nn.Dropout`; no default RNG consumption by dropout.

## Environment contract

Immutable `environment_contract.json` before RNG/init/data. Binds library /
CUDA / determinism / thread / TF32 / AMP flags. Operational fields (GPU UUID,
hostname, paths, times) stay in runtime attestation only.

## Training lifecycle

- 16/8/8 split isolation; evaluation content locked until unlock artifact
- Epoch 0 calibration baseline protected (`min_delta=1e-5` primary-only replace)
- AdamW decay/no-decay groups; explicit warmup+cosine schedule (not PyTorch scheduler state)
- Global L2 clip 1.0; any nonfinite fails the seed closed
- Trace exact IEEE bits + scientific hash chain (tail is final identity)
- Epoch staging transaction; resume only from verified last at epoch boundaries
- Seeds `[17,29,43]`; collection → canonical selection → reproduction → deploy export → evaluation unlock

## Deployment

Deployment checkpoint embeds B2-03B normalization, golden CPU cases, and
deployment-only state. Loader: CPU golden bit-exact → GPU \(10^{-6}\)
qualification → immutable inference wrapper. Process-local positive cache only.

## Fusion

Extend production sum-preserving formula \(A_d=n_d\sum_i w_i A_i\) with fixed
layer-order FP32 accumulation and **exact uniform bit-pattern** fast path
(`uniform_baseline` adds maps; near-uniform does not trigger).

## Evaluation and gates

Depth-24 category-macro gates for target learning (KL vs uniform) and
localization (Pixel AP / AUROC / AUPRO deltas). Qualification states:
`deployment_qualified`, `trained_but_not_deployment_qualified`,
`localized_but_target_fidelity_unqualified`, or both-fail. Accepted manifest
binds deploy / qualification / accepted scientific identities; `.pt` alone is
insufficient.

## Failure states

Nonfinite / identity / persistence / environment failures write failure
attestations, forbid in-place resume, and stop the seed collection without a
success final manifest.

## Scope exclusions

No teacher/backbone invocation, no VisA/target-domain access, no residual-gain
supervision, no LSE, no early-exit policy, no real DLCM checkpoints in B2-05A,
no evaluation unlock, no accepted deployment artifact generation from real runs.
