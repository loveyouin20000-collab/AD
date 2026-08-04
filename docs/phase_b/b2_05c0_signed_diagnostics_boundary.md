# B2-05C0 Signed-Diagnostics Boundary Correction

## Finding corrected

Previous minor note claimed missing aux heads at deployment evaluation could use
deployment allocation weights as a signed-Shapley diagnostic proxy.

That substitution is **mathematically invalid**:

| Object | Geometry |
|--------|----------|
| deployment weights | probability simplex; nonnegative; sum = 1 |
| standardized signed Shapley φ | unbounded signed reals; may be negative; does not sum to 1 |

## Correct reporting boundary

- **Full training checkpoint**: signed Huber / signed Spearman / signed pairwise
  ranking may use `gt_signed_head` and `teacher_signed_head` independently.
  Label `diagnostic_source = canonical_best_training_checkpoint` and
  `not_part_of_deployment_artifact = true`.
- **Stripped deployment checkpoint**: allocation KL/JSD, allocation Top-1, and
  localization remain valid. Signed metrics are recorded as:

```text
status = not_available_in_deployment_artifact
reason = training_only_auxiliary_heads_removed
```

Never zero, NaN, or a weight-derived proxy.

## Qualification impact

Signed diagnostics are **not** a hard V1 target gate. After correction:

```text
qualification_status = localized_but_target_fidelity_unqualified
deployment_qualified = false
accepted_deployment_manifest_created = false
```

All scientific training / deployment / qualification identities are unchanged
(see `b2_05b_dlcm_training_manifest.json` and sealed run artifacts).

## Evidence hashes

| Artifact | Role |
|----------|------|
| sealed `qualification_result.json` | unchanged scientific qualification identity |
| `b2_05c0_target_conflict_diagnosis.json` | new diagnosis-only evidence (may change hash when regenerated) |
