# B2-04B Contribution-Target Materialization Architecture

## Status and authority

B2-04B enables the B2-04A production materializer, creates two independent
accepted-input contribution-target collections, and qualifies their scientific
equivalence. It does not change the frozen B2-04A mathematics.

Authoritative base:

- tag: `b2-contribution-target-contract-v1`
- commit: `29591668c3228f6cebd7fd923ae1c39c6dad49bc`

## Two distinct plan identities

`fixture_contract_plan_sha256` and `expected_accepted_input_plan_sha256` are
different identities and are never interchangeable.

| Identity | Inputs | Proves | Authorizes official materialization |
|---|---|---|---|
| `fixture_contract_plan_sha256` | hermetic B2-04A synthetic fixture | frozen mathematics, deterministic input ordering, no-write dry run | never |
| `expected_accepted_input_plan_sha256` | accepted teacher cache A plus descriptor collection A/B | the real accepted-input plan | yes, and only after two independent real dry-runs agree |

The hermetic regression pin is
`a072b67b9154b193dccd99e1123c2d5ef09583114e6b2840061cdfcf92ac93d5`. It
supersedes `fa3d2435d684a310c81c151c48717afc9455401adce41fbe4d8a96f5c776a84e`,
which additionally hashed the run-control field
`official_materialization_enabled`; the superseded value was never an
accepted-input identity.

The scientific plan identity covers upstream scientific hashes, the planned
sample identities and their record hashes, GT calibration, Shapley
normalization, coverage counts, the depth lattice, the contract versions, and
the teacher forward count. Run control, repository gating, the observed HEAD
and worktree state, the CLI mode, and every output location are operational
attestation only and never enter the identity.

## Frozen scientific contract

The following B2-04A values remain unchanged:

- target families: `gt_localization`, `teacher_fidelity`
- depth 12: players `[6,12]`, 4 coalitions
- depth 18: players `[6,12,18]`, 8 coalitions
- depth 24: players `[6,12,18,24]`, 16 coalitions
- coalition fusion: equal average
- Shapley: exact enumeration, `float64`, efficiency tolerance `1e-12`
- split coverage: 16 training, 8 calibration, 8 evaluation
- GT calibration membership: 16/0/0
- Shapley normalization membership: 16/0/0
- teacher forward count: zero

The accepted execution-profile, split, and checkpoint identities remain:

- execution profile:
  `7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d`
- split V2:
  `91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0`
- checkpoint:
  `97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4`

No utility, Shapley, allocation, calibration, normalization, hashing, or
coverage formula is duplicated or rewritten in B2-04B.

## Configuration boundary

`configs/phase_b/b2_contribution_targets_official_v1.json` is a new,
independent official configuration. The historical
`b2_contribution_targets_gate_c.json` remains byte-for-byte unchanged.

The official configuration preserves the existing long field names:

```text
expected_contribution_contract_tag
expected_contribution_contract_commit
```

It does not introduce or accept `expected_contract_tag` or
`expected_contract_commit`. Its B2-04B-specific values are:

```text
configuration_id = b2_contribution_targets_official_v1
contract_stage = b2_04b
official_materialization_enabled = true
repository_identity_gate_enabled = true
resume_enabled = false
expected_plan_sha_required_for_official = true
expected_contribution_contract_tag = b2-contribution-target-contract-v1
expected_contribution_contract_commit = 29591668c3228f6cebd7fd923ae1c39c6dad49bc
```

All scientific pins are copied unchanged from the reviewed Gate-C
configuration. No machine-local path is tracked.

## Production repository-identity gate

The gate is enforced inside `materialize_contribution_target_collection`, not
only by the CLI. A caller cannot reach persistence under the official
configuration without passing it.

At official API entry the gate:

1. resolves the configured contract tag as a commit;
2. fails if the tag is missing or resolves to any commit other than the
   configured contract commit;
3. fails if the configured commit differs from the frozen B2-04B commit;
4. reads the observed HEAD and proves the contract commit is its ancestor;
5. records that HEAD as the generation execution identity;
6. requires `git status --porcelain --untracked-files=all` to be empty.

After the complete scientific plan is recomputed and matched to the caller's
expected plan SHA, but before creating the run directory or writing any file,
the gate repeats the tag, ancestry, HEAD, and clean-worktree checks. The second
observation must equal the entry snapshot. A HEAD transition or worktree
mutation during calculation therefore fails without creating a passed run.

The verified outer identity is persisted in the final manifest as provenance,
but is excluded from scientific hashes. Hermetic tests use temporary Git
repositories; no identity override, skip-tag flag, or force-clean harness is
added.

Dry-run still performs the complete scientific calculation and writes nothing.
The clean-worktree persistence gate applies only to official materialization.

## Materialization flow

The existing B2-04A path remains authoritative:

```text
production upstream validators
  → ContributionInputBundle
  → run_contribution_target_collection
  → exact accepted plan comparison
  → second repository-identity observation
  → fresh run-directory claim
  → existing atomic record/statistics writers
  → final manifest and receipt
  → verify_contribution_target_collection from disk
```

Output collision is checked before persistence and resume remains unsupported.
The expected layout is exactly 36 files:

```text
records/*.pt × 32
gt_map_calibration.pt
shapley_normalization.pt
final_manifest.json
final_manifest.json.sha256
```

No teacher or checkpoint inference argument is introduced. The materializer
consumes accepted teacher-cache maps and descriptor records only.

## Verified-collection comparison

`verify_contribution_target_collection` remains the sole constructor used for
qualification inputs. It independently checks receipt, directory integrity,
file hashes, embedded scientific hashes, record schemas, calibration,
normalization, replayed plan, and manifest consistency from disk.

`compare_contribution_target_collections` accepts only two
`VerifiedContributionTargetCollection` values. It returns a structured
comparison with explicit predicates and mismatch reasons. Scientific
equivalence requires exact equality of:

- the contribution plan and all seven layered scientific identities;
- the ordered 32 stable IDs, split membership, candidate layers, and depths;
- every record scientific hash and source teacher/descriptor identity;
- every depth-local coalition bitmask table and utility component;
- raw and centered GT and teacher utility values;
- signed Shapley values, allocation targets, and efficiency residuals;
- all three GT calibration statistics and training/coalition coverage;
- every target-family × depth × layer normalization count, mean, standard
  deviation, minimum, maximum, and zero-variance marker;
- training/calibration/evaluation coverage and 16/8/8 membership;
- `teacher_forward_count == 0` for both collections.

Comparison uses exact tensor equality and exact scalar/mapping equality because
the accepted computation is deterministic `float64`. File-byte equality is
reported separately and never defines scientific equivalence.

## Qualification writer

`tools/qualify_b2_contribution_target_reproduction.py` is a narrow,
deterministic evidence writer. It:

1. loads the official configuration;
2. verifies Run A and Run B from their run directories using the production
   verification API;
3. compares the resulting verified objects using the production comparison
   API;
4. validates required semantic-check, negative-control, test, Ruff, and mypy
   result summaries supplied as deterministic tool-generated qualification
   inputs;
5. rejects any absent decision-critical field or non-passing result;
6. atomically writes concise JSON and Markdown outputs.

The writer records source contract identity, generation commit, accepted plan,
all layered identities, an ordered 32-record hash summary, coverage hashes,
calibration and normalization identities, comparison predicates,
teacher-forward count, source-only audit, negative controls, validation
results, and explicit scope exclusions.

Absolute paths, raw tensors, raw utility dumps, timestamps, and temporary
command output are excluded. Run labels are diagnostic only. Output status is:

```text
deterministic_dual_contribution_target_reproduction
```

`--dry-run` performs verification and rendering without writing. `--config`,
`--seed`, and `--output-dir` remain available according to the repository CLI
contract.

## Accepted upstream boundary

Run A uses accepted teacher cache A plus descriptor collection A. Run B uses
the same teacher cache A plus independent descriptor collection B. Paths alone
never establish identity: the production validators must prove status,
receipt, every record hash, 32-sample and 16/8/8 coverage, split/checkpoint/
profile pins, source-only provenance, and absence of VisA or target-domain
records.

Descriptor A and B must be scientifically equivalent before target generation.
The canonical MVTec source is identified through stable IDs and accepted source
provenance rather than an absolute pathname. A missing or invalid accepted
source stops execution; no source is regenerated or substituted.

## Reproduction protocol

On the clean committed implementation:

1. Dry-run A computes teacher A + descriptor A and must reproduce the accepted
   plan SHA without writing.
2. Dry-run B independently computes teacher A + descriptor B and must reproduce
   the same accepted plan SHA without writing.
3. Official Run A independently fits all calibration/normalization statistics,
   writes 36 artifacts, and reload-verifies them.
4. Official Run B repeats the complete process in a separate process and fresh
   directory without reusing any Run A target artifact or in-memory plan.
5. Production verification and comparison establish scientific equivalence.
6. Six source-only semantic spot checks recompute selected normal/anomalous
   training, calibration, and evaluation records without running the backbone.
7. Negative controls operate only on temporary copies and must fail closed.
8. The qualification writer emits the two concise tracked evidence files.

If either real dry-run plan differs from the accepted SHA, execution stops and
the differing scientific payload fields are reported for specification
adjudication. A new SHA is never silently accepted.

## Evidence and provenance layers

The report distinguishes:

- scientific identity: canonical scientific content and layered identities;
- file-byte integrity: per-file SHA-256 and receipt validation;
- outer provenance: contract tag/commit, generation commit, clean execution,
  accepted upstream identities, seed, split manifest, and source-only audit.

Raw `.pt` files and run directories remain ignored and untracked.

## Scope exclusions

B2-04B does not train DLCM, create residual-gain targets, train LSE, calibrate
an early-exit policy, rerun the VisualAD teacher/backbone, access target-domain
data, publish a branch, open a PR, create a release tag, or start B2-05.
