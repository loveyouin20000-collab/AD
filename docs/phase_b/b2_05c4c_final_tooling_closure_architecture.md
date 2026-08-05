# B2-05C4C Final Tooling Closure Architecture

## Scope

B2-05C4C belongs to C4. It does not open C5 and does not change any scientific result from C3, C4A, or C4B. The stage only implements and freezes the V5 Final execution tooling contract so a later authorized stage can materialize and evaluate the already adopted Final roster.

The existing fail-closed evidence commit `8b2dc932727f3d7ee85fe6734c1d030a90191845` is preserved as history. Its conclusion remains valid for that point in time: Final execution was blocked by unimplemented materialization tooling. C4C replaces the stub tooling with guarded tooling, but it does not run real Final materialization or real Final evaluation.

## Frozen Scientific Inputs

The tooling must preserve these identities without reinterpretation:

- C3 canonical seed: `17`
- beta*: `0.54`
- Calibration A/B identity: `cae406c91ec392ffd7cc6d48ec2f0c94ab78d78f905cbfe904287842a7a7278a`
- V5 deployment identity: `c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd`
- Final roster identity: `267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213`
- final_content_resolved: `false`
- accepted manifest: none
- LSE/residual-gain/early-exit: not started

The tooling does not modify the C3 canonical checkpoint, seed, trunk, heads, normalization, beta grid, beta*, Calibration artifacts, Development-qualified identity/result, Final roster identity/order/content, thresholds, or teacher policy.

## Contract And Unlocks

`b2_dlcm_v5_final_execution_contract_v1` binds the frozen V5 deployment, beta, Calibration A/B identity, Development-qualified identity, Final roster identity, source master manifest identity, normalization/calibration identities, Final A/B protocols, gate thresholds, and decision/evidence/accepted schemas.

No legal unlock means all Final content access fails with `B2_DLCM_FINAL_CONTENT_ACCESS_FORBIDDEN`. There is no `--force-final-access`, environment-variable bypass, or ordinary CLI path input for Final images or masks. A valid unlock must verify schema/version, receipt SHA, V5 deployment, beta, Calibration, Development, Final roster, tooling tag/commit, accepted execution plan, single-use state, HEAD, worktree status, and config identity.

## Stable ID Resolution

Stable ID resolution is allowed only after a valid materialization unlock. Resolution verifies the source master manifest and receipt, performs exact stable ID lookup, verifies category, label, and source-record identity, and writes a temporary operational resolution. It forbids directory scans, file-name inference, fuzzy matching, fallback records, and record replacement.

Resolution is untracked, excluded from scientific identity and evidence, unavailable to ordinary CLI paths, and removed after success or failure. Receipts never record paths.

## Materialization

Materialization is a transaction:

```text
staging -> production verify -> A/B compare -> atomic authoritative commit
```

Materialization A and B run as independent processes. Each process starts from disk inputs and loads the source manifest, roster, training-only stats, teacher/backbone inputs, and upstream identities independently. The two processes do not share Python process state, model instances, GPU cache, tensors, scientific cache, or partial artifacts.

Each Final record produces descriptor, mask, causal/full-depth maps, teacher maps, GT/teacher coalition utilities, GT/teacher Shapley, GT/teacher allocation targets, and record identity. A/B comparison requires scientific payload equality and canonical scientific file byte equality. Operational attestation may differ only outside scientific identity.

Failures never write success receipts, delete staging/resolution/partial artifacts, leave the unlock unconsumed, and forbid partial reuse. Success makes A authoritative, records B as reproduction evidence, writes atomic receipts, and consumes the unlock. Reusing the same unlock fails.

## Evaluation

Evaluation A and B are independent processes. Each loads the authoritative Final materialization, the V5 deployment candidate, the C3 canonical best training checkpoint, the Final evaluation unlock, and production metrics from disk. They do not share model/cache/predictions/metric state.

Evaluation computes GT allocation metrics, Pixel AUROC/AP/AUPRO, teacher-map Spearman and Top-1% overlap, teacher allocation diagnostics, GT/teacher signed diagnostics, per-sample/per-category/macro/pooled summaries, blocking gates, production invocation proof, `H_decision`, and `H_evidence`.

The Final gates are unchanged from Development. `H_decision` binds only untouched Final GT target-learning, localization, thresholds, and verdict. `H_evidence` binds `H_decision`, Calibration, Development, teacher diagnostics, Materialization/Evaluation A/B, tooling identity, production proof, and C1-C4 provenance.

## Accepted Loader

The accepted writer is strictly ordered:

1. `final_decision_manifest`
2. `final_evidence_manifest`
3. `accepted_deployment_manifest`

The formal loader accepts only the accepted V5 path. It verifies accepted manifest/receipt, V5 deployment identity, beta*=0.54, Calibration identity, decision/evidence identities, upstream identities, and CPU/GPU/batch proofs. Without an accepted manifest it continues to reject Development-qualified candidates.

## C4C Dry-Run Boundary

C4C dry-runs must prove:

```text
real_final_content_accessed = false
stable_ids_resolved = false
materialization_started = false
evaluation_started = false
accepted_written = false
run_directory_created = false
artifact_written = false
```

Two independent dry-runs must produce the same `accepted_v5_final_execution_plan_scientific_sha256`. The plan excludes paths, timestamps, runtime labels, GPU UUIDs, and mode.

## Testing

All tests in C4C use hermetic fixtures. Real Final content remains untouched until tooling is frozen, Critical=0, Important=0, focused/full/GPU/Ruff/mypy pass, the Final execution plan is pinned, and a valid unlock is generated.
