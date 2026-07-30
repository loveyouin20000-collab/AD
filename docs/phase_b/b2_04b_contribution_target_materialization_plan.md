# B2-04B Contribution-Target Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans` to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Enable the reviewed production contribution-target materializer,
produce two independent accepted-input collections, and prove deterministic
scientific equivalence without running the teacher or training downstream
models.

**Architecture:** Add one pinned B2-04B configuration and enforce its
repository identity inside the production materialization API. Reuse the
existing computation and verification path, add a strict comparison API over
disk-verified collections, and generate concise evidence through an atomic
qualification writer.

**Tech Stack:** Python 3.10.20, PyTorch CPU, pytest, Ruff, mypy, Git.

## Global constraints

- Worktree:
  `/root/autodl-tmp/AD-phase-b2-contribution-target-materialization`
- Branch: `phase-b2-contribution-target-materialization`
- Base tag: `b2-contribution-target-contract-v1`
- Base commit: `29591668c3228f6cebd7fd923ae1c39c6dad49bc`
- Python: `/root/miniconda3/envs/rad-visualad/bin/python`
- Accepted plan:
  `fa3d2435d684a310c81c151c48717afc9455401adce41fbe4d8a96f5c776a84e`
- Use `CUDA_VISIBLE_DEVICES=""`; teacher forward count remains exactly zero.
- Preserve `expected_contribution_contract_tag` and
  `expected_contribution_contract_commit`; do not add short aliases.
- Do not modify the B2-04A schema or historical Gate-C configuration.
- Do not rewrite utility, Shapley, allocation, calibration, normalization, or
  scientific-hash mathematics.
- Candidate layers remain configuration-driven.
- No resume, bypass, force-clean, fixture, identity override, hash override, or
  skip-tag CLI option.
- No target-domain data, teacher/backbone rerun, DLCM, residual-gain, LSE,
  early-exit policy, B2-05, push, PR, or release tag.
- Raw `.pt` artifacts and run directories remain ignored and untracked.

## File responsibility map

| Path | Responsibility |
|---|---|
| `configs/phase_b/b2_contribution_targets_official_v1.json` | Pinned, portable B2-04B official configuration |
| `rad/phase_b/b2_contribution_targets.py` | Config profile validation, API repository gate, disk verification, strict comparison |
| `tools/create_b2_contribution_targets.py` | CLI argument wiring only; production gate remains in domain API |
| `tools/qualify_b2_contribution_target_reproduction.py` | Disk verify, compare, deterministic render, atomic evidence writes |
| `tests/rad/b2_contribution_target_fixtures.py` | Hermetic official-config and independent-run fixtures |
| `tests/rad/test_b2_contribution_targets.py` | Domain, identity, materialization, comparison, and negative tests |
| `tests/rad/test_b2_contribution_targets_cli.py` | Official CLI and qualification-writer tests |
| `tests/rad/test_b2_contribution_targets_portability.py` | No local paths, no short aliases, no target-domain/teacher imports |
| `docs/phase_b/b2_04b_contribution_targets_manifest.json` | Generated concise machine-readable evidence |
| `docs/phase_b/b2_04b_contribution_targets_report.md` | Generated concise human-readable evidence |

---

### Task 1: Commit the approved architecture and plan

**Files:**

- Create:
  `docs/phase_b/b2_04b_contribution_target_materialization_architecture.md`
- Create:
  `docs/phase_b/b2_04b_contribution_target_materialization_plan.md`

**Interfaces:**

- Consumes: approved B2-04B narrow-extension design.
- Produces: immutable implementation and execution protocol for later commits.

- [ ] **Step 1: Check both documents for required identities and exclusions**

Run:

```bash
rg -n \
  'b2-contribution-target-contract-v1|29591668c3228f6cebd7fd923ae1c39c6dad49bc|fa3d2435d684a310c81c151c48717afc9455401adce41fbe4d8a96f5c776a84e|expected_contribution_contract_' \
  docs/phase_b/b2_04b_contribution_target_materialization_*.md
```

Expected: both documents contain the frozen authority; the architecture
contains both long configuration fields.

- [ ] **Step 2: Reject placeholders and confirm short aliases are prohibited**

Run:

```bash
! rg -n 'T[B]D|T[O]DO|implement[ ]later|fill[ ]in[ ]details' \
  docs/phase_b/b2_04b_contribution_target_materialization_*.md
rg -n 'does not introduce or accept.*expected_contract|short alias' \
  docs/phase_b/b2_04b_contribution_target_materialization_*.md
```

Expected: the placeholder scan has no matches and the documents explicitly
prohibit short aliases.

- [ ] **Step 3: Review the exact diff**

Run:

```bash
git diff --check &&
git diff -- \
  docs/phase_b/b2_04b_contribution_target_materialization_architecture.md \
  docs/phase_b/b2_04b_contribution_target_materialization_plan.md
```

Expected: no whitespace errors; only the two approved documents are present.

- [ ] **Step 4: Commit architecture and plan**

Run:

```bash
git add \
  docs/phase_b/b2_04b_contribution_target_materialization_architecture.md \
  docs/phase_b/b2_04b_contribution_target_materialization_plan.md
git commit -m "docs: define B2 contribution target materialization"
git status --short
```

Expected: commit succeeds and status is empty.

---

### Task 2: RED—pin the independent official configuration

**Files:**

- Create: `configs/phase_b/b2_contribution_targets_official_v1.json`
- Modify: `tests/rad/b2_contribution_target_fixtures.py`
- Modify: `tests/rad/test_b2_contribution_targets.py`
- Modify: `tests/rad/test_b2_contribution_targets_portability.py`

**Interfaces:**

- Consumes: `load_contribution_targets_config(path)`.
- Produces: a B2-04B `ContributionTargetsConfig` with the existing long
  contract fields and unchanged scientific pins.

- [ ] **Step 1: Add failing official-config identity tests**

Add tests asserting:

```python
official.configuration_id == "b2_contribution_targets_official_v1"
official.contract_stage == "b2_04b"
official.official_materialization_enabled is True
official.repository_identity_gate_enabled is True
official.resume_enabled is False
official.expected_plan_sha_required_for_official is True
official.expected_contribution_contract_tag == "b2-contribution-target-contract-v1"
official.expected_contribution_contract_commit == (
    "29591668c3228f6cebd7fd923ae1c39c6dad49bc"
)
```

Also assert every mathematical/scientific field equals the loaded B2-04A
Gate-C value, the historical file bytes are unchanged, no machine-local path is
present, and neither short alias occurs in the JSON or loader source.

- [ ] **Step 2: Add failing drift cases**

Parameterize the official payload with:

```python
{"contract_stage": "b2_04a"}
{"official_materialization_enabled": False}
{"repository_identity_gate_enabled": False}
{"resume_enabled": True}
{"expected_contribution_contract_tag": None}
{"expected_contribution_contract_tag": "moved-tag"}
{"expected_contribution_contract_commit": None}
{"expected_contribution_contract_commit": "0" * 40}
```

Every case must raise `B2_CONTRIBUTION_CONFIG_DRIFT`.

- [ ] **Step 3: Observe RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest \
  tests/rad/test_b2_contribution_targets.py \
  tests/rad/test_b2_contribution_targets_portability.py \
  -q --tb=short
```

Expected: failures because the official file/profile does not exist and the
loader only accepts the B2-04A profile.

- [ ] **Step 4: Add the official JSON and narrow profile validation**

Copy the Gate-C scientific values exactly. Change only the B2-04B identity and
gate fields. In `_validate_pinned_contribution_config`, retain the existing
B2-04A branch and add one explicit branch for
`b2_contribution_targets_official_v1`; reject every other combination. Do not
add aliases or a generic policy framework.

- [ ] **Step 5: Observe GREEN**

Run the command from Step 3.

Expected: all selected tests pass.

---

### Task 3: RED→GREEN—enforce repository identity at the production API

**Files:**

- Modify: `rad/phase_b/b2_contribution_targets.py`
- Modify: `tools/create_b2_contribution_targets.py`
- Modify: `tests/rad/test_b2_contribution_targets.py`
- Modify: `tests/rad/test_b2_contribution_targets_cli.py`

**Interfaces:**

- Add immutable `ContributionRepositoryIdentity` carrying:
  `contract_tag`, `contract_commit`, `generation_commit`,
  `head_is_descendant`, and `worktree_clean`.
- Add:

```python
def verify_contribution_repository_identity(
    *,
    config: ContributionTargetsConfig,
    repository_root: Path,
    expected_generation_commit: str | None = None,
) -> ContributionRepositoryIdentity:
    ...
```

- Extend:

```python
def materialize_contribution_target_collection(
    *,
    config: ContributionTargetsConfig,
    inputs: Any,
    output_run_dir: Any,
    expected_plan_sha256: Any,
    repository_root: Any | None = None,
    candidate_layers: Sequence[int] | None = None,
    prediction_depths: Sequence[int] | None = None,
) -> ContributionTargetMaterializationResult:
    ...
```

`repository_root` is a location required by the enabled gate, not an identity
override. Configurations with the gate disabled preserve hermetic B2-04A tests.

- [ ] **Step 1: Add temporary-repository RED tests**

Create real temporary Git histories and test:

- missing contract tag;
- moved tag;
- wrong configured commit;
- sibling and unrelated HEAD;
- clean descendant HEAD passes;
- dirty official worktree;
- expected observed HEAD mismatch.

Assert documented `B2_CONTRIBUTION_*` errors and no output directory.

- [ ] **Step 2: Add materialization-boundary RED tests**

Call `materialize_contribution_target_collection` directly, bypassing the CLI.
Prove the official config cannot persist without `repository_root`, with a
dirty repository, or from a non-descendant HEAD.

Patch only the production collection call in a hermetic test so it commits a
new HEAD or creates an untracked file after plan computation. Assert the second
identity check rejects the run before directory creation.

- [ ] **Step 3: Add CLI RED tests**

Prove official non-dry-run:

- requires `--expected-plan-sha256`;
- rejects malformed and mismatched values;
- rejects an existing output directory;
- routes through the API gate;
- exposes no resume, tag-skip, identity, fixture, hash-override, or force-clean
  argument.

- [ ] **Step 4: Observe RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest \
  tests/rad/test_b2_contribution_targets.py \
  tests/rad/test_b2_contribution_targets_cli.py \
  -q --tb=short
```

Expected: new repository and API-boundary tests fail because no gate exists.

- [ ] **Step 5: Implement the minimum gate**

Use `git -C <repo>` with captured output and no shell. Resolve
`<configured-tag>^{commit}`, compare it to the pinned commit, run
`merge-base --is-ancestor`, read HEAD, and check porcelain status including
untracked files.

At materialization entry, snapshot identity. Refuse output collisions, compute
the full collection, match the accepted plan, then call the same verifier with
`expected_generation_commit=<snapshot HEAD>` immediately before `mkdir`.
Attach the verified outer provenance to the final manifest without adding it to
scientific payloads.

- [ ] **Step 6: Observe GREEN**

Run the command from Step 4.

Expected: all selected tests pass and all rejected runs leave no output.

---

### Task 4: RED→GREEN—add strict verified-collection comparison

**Files:**

- Modify: `rad/phase_b/b2_contribution_targets.py`
- Modify: `tests/rad/test_b2_contribution_targets.py`

**Interfaces:**

- Add immutable `ContributionTargetCollectionComparison` with:

```python
scientifically_equivalent: bool
reasons: tuple[str, ...]
layered_identities_equal: bool
record_scientific_hashes_equal: bool
utility_tables_equal: bool
signed_shapley_equal: bool
allocations_equal: bool
gt_calibration_equal: bool
shapley_normalization_equal: bool
coverage_equal: bool
teacher_forward_count_equal: bool
file_byte_equal: bool
```

- Add:

```python
def compare_contribution_target_collections(
    *,
    first: VerifiedContributionTargetCollection,
    second: VerifiedContributionTargetCollection,
) -> ContributionTargetCollectionComparison:
    ...
```

- `verify_contribution_target_collection` applies a module-private verification
  seal. Comparison rejects mappings, materialization results, and manually
  constructed/unsealed collection objects.

- [ ] **Step 1: Add identical independent-run RED test**

Materialize two fresh fixture runs, reload both through
`verify_contribution_target_collection`, compare them, and require every
scientific predicate true. File-byte equality is asserted only as a diagnostic
field.

- [ ] **Step 2: Add categorized mismatch RED tests**

Create independently verified fixture collections whose valid scientific
content differs in one controlled category at a time. Require
`scientifically_equivalent is False` for:

- record scientific hash;
- coalition utility component;
- raw utility;
- centered value;
- signed Shapley;
- allocation;
- GT calibration statistic;
- Shapley normalization statistic;
- split/coverage;
- nonzero teacher-forward count.

Assert each difference sets its corresponding predicate false and emits a
stable reason containing the sample/depth/family where applicable.

- [ ] **Step 3: Add unverified-input RED tests**

Pass a manifest mapping, a materialization result, and an unsealed collection
to comparison. Each must raise
`B2_CONTRIBUTION_COLLECTION_NOT_VERIFIED`.

- [ ] **Step 4: Observe RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest tests/rad/test_b2_contribution_targets.py \
  -k 'comparison or unverified' -q --tb=short
```

Expected: failures because the comparison API does not exist.

- [ ] **Step 5: Implement exact categorized comparison**

Compare the plan and seven identity keys first, then ordered IDs and metadata.
For every sample and depth, compare the complete coalition table, separate
family raw/centered values, exact signed Shapley, allocations, and residuals.
Compare calibration and normalization scientific content independently of
their declared hashes. Use recursive exact equality with `torch.equal` for
tensors. Compute file-byte maps separately from all files beneath each verified
run directory.

- [ ] **Step 6: Observe GREEN**

Run the command from Step 4.

Expected: all comparison and unverified-input tests pass.

---

### Task 5: RED→GREEN—add deterministic qualification writer

**Files:**

- Create: `tools/qualify_b2_contribution_target_reproduction.py`
- Modify: `tests/rad/test_b2_contribution_targets_cli.py`
- Modify: `tests/rad/test_b2_contribution_targets_portability.py`

**Interfaces:**

CLI arguments:

```text
--config
--run-a
--run-b
--qualification-results
--output-dir
--seed
--dry-run
```

The deterministic qualification-results JSON has:

```json
{
  "schema_version": 1,
  "semantic_spot_checks": {
    "status": "passed",
    "sample_count": 6,
    "depths": [12, 18, 24],
    "run_a_equals_run_b": true
  },
  "source_only_audit": {
    "status": "passed",
    "target_domain_record_count": 0,
    "teacher_forward_count": 0
  },
  "negative_controls": {
    "status": "passed",
    "required": 34,
    "passed": 34,
    "case_ids": []
  },
  "validation": {
    "focused_pytest": {"status": "passed", "exit_code": 0, "summary": ""},
    "full_cpu_pytest": {"status": "passed", "exit_code": 0, "summary": ""},
    "ruff": {"status": "passed", "exit_code": 0, "summary": ""},
    "mypy": {"status": "passed", "exit_code": 0, "summary": ""}
  }
}
```

`case_ids` must equal the frozen ordered negative-control list from Task 6;
summaries are concise command-produced text with absolute paths removed.

- [ ] **Step 1: Add qualification RED tests**

Materialize two fixture collections and assert the writer:

- calls production verification for both directories;
- rejects an unverified or failed collection;
- rejects every false scientific comparison predicate;
- rejects nonzero teacher-forward count;
- rejects missing semantic, source-only, negative, test, Ruff, or mypy results;
- rejects an incomplete negative-control list;
- excludes absolute paths and tensors;
- emits ordered 32-record hashes and all layered identities;
- produces byte-identical JSON and Markdown for repeated inputs;
- performs no write in `--dry-run`;
- refuses output collisions;
- removes temporary files after success or failure.

- [ ] **Step 2: Observe RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest tests/rad/test_b2_contribution_targets_cli.py \
  -k qualification -q --tb=short
```

Expected: failures because the qualification module does not exist.

- [ ] **Step 3: Implement verification, rendering, and atomic writes**

Load config, call `verify_contribution_target_collection` twice, then call
`compare_contribution_target_collections`. Build evidence only from verified
objects and validated qualification results. Render canonical JSON with sorted
keys and a trailing newline; derive Markdown from the same in-memory evidence.
Write each destination through a same-directory temporary file, flush, fsync,
and `os.replace`. Refuse pre-existing output files.

- [ ] **Step 4: Observe GREEN**

Run the command from Step 2.

Expected: all qualification tests pass.

---

### Task 6: Complete the fail-closed matrix and implementation validation

**Files:**

- Modify: `tests/rad/test_b2_contribution_targets.py`
- Modify: `tests/rad/test_b2_contribution_targets_cli.py`
- Modify: `tests/rad/test_b2_contribution_targets_portability.py`

**Interfaces:**

- Consumes: production materialization, verification, comparison, and
  qualification APIs.
- Produces: 34 named hermetic negative controls with no passed manifest left by
  a rejected case.

- [ ] **Step 1: Map or add every negative case**

Freeze these IDs:

```text
record_file_byte_drift
record_scientific_hash_drift
coalition_utility_component_drift
raw_utility_drift
centered_value_drift
signed_shapley_drift
allocation_drift
efficiency_residual_above_tolerance
changed_split_membership
training_record_moved_to_calibration
calibration_record_in_gt_fitting
evaluation_record_in_normalization
gt_calibration_statistic_drift
shapley_normalization_statistic_drift
teacher_cache_identity_drift
descriptor_collection_identity_drift
descriptor_record_identity_drift
wrong_split_checkpoint_profile
target_domain_or_visa_source
missing_record
extra_record
orphan_pt
path_traversal
symlink_escape
missing_receipt
receipt_mismatch
output_directory_collision
completed_run_reuse
resume_attempt
wrong_expected_plan_sha
dirty_official_worktree
non_descendant_official_head
moved_or_missing_contract_tag
nonzero_teacher_forward_count
```

Reuse existing verifier tests where they already exercise the production API;
add only missing cases. Every case works on a fixture run or temporary copy and
asserts no passed final manifest is created by the rejected operation.

- [ ] **Step 2: Run focused tests**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest \
  tests/rad/test_b2_contribution_targets.py \
  tests/rad/test_b2_contribution_targets_cli.py \
  tests/rad/test_b2_contribution_targets_portability.py \
  -q --tb=short
```

Expected: zero failures.

- [ ] **Step 3: Run the full CPU suite**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest tests/rad -q --tb=short
```

Expected: zero failures.

- [ ] **Step 4: Run Ruff and scoped mypy**

Run:

```bash
/root/miniconda3/envs/rad-visualad/bin/python -m ruff check \
  rad/phase_b/b2_contribution_targets.py \
  tools/create_b2_contribution_targets.py \
  tools/qualify_b2_contribution_target_reproduction.py \
  tests/rad/b2_contribution_target_fixtures.py \
  tests/rad/test_b2_contribution_targets.py \
  tests/rad/test_b2_contribution_targets_cli.py \
  tests/rad/test_b2_contribution_targets_portability.py

/root/miniconda3/envs/rad-visualad/bin/python -m mypy \
  rad/phase_b/b2_contribution_targets.py \
  tools/create_b2_contribution_targets.py \
  tools/qualify_b2_contribution_target_reproduction.py
```

Expected: both commands exit zero.

- [ ] **Step 5: Commit official enablement and qualification implementation**

Run:

```bash
git add \
  configs/phase_b/b2_contribution_targets_official_v1.json \
  rad/phase_b/b2_contribution_targets.py \
  tools/create_b2_contribution_targets.py \
  tools/qualify_b2_contribution_target_reproduction.py \
  tests/rad/b2_contribution_target_fixtures.py \
  tests/rad/test_b2_contribution_targets.py \
  tests/rad/test_b2_contribution_targets_cli.py \
  tests/rad/test_b2_contribution_targets_portability.py
git commit -m "feat: enable qualified B2 contribution target materialization"
git status --short
```

Expected: commit succeeds and the worktree is clean.

---

### Task 7: Resolve and verify accepted upstream artifacts

**Files:**

- No tracked file changes.

**Interfaces:**

- Consumes: accepted teacher and descriptor manifests and production validators.
- Produces: verified source paths and identities for real execution.

- [ ] **Step 1: Locate candidate manifests without substituting sources**

Inspect the known candidate directories beneath existing Phase B artifact
roots:

```text
artifacts/phase_b/b2_teacher_cache/authoritative-run-a-20260723-155404
artifacts/phase_b/b2_descriptor_artifacts/authoritative-run-a-20260729-013956
artifacts/phase_b/b2_descriptor_artifacts/authoritative-run-b-20260729-014404
```

Select only manifests whose production validators reproduce the accepted
scientific and coverage identities. If any source is absent, stop.

- [ ] **Step 2: Verify teacher cache A**

Require:

```text
teacher_cache_scientific_sha256 =
66d23807e868696a9c4a68ad83399d82df3d33e743a97d97eeb98ac60c0b1b0a
teacher_cache_sample_coverage_sha256 =
6e538b902795c377f9992258e307e58b5c0ba0f99cbbe6c3853a81947ca3d76c
```

The production validator must prove passed status, receipt, all file and
scientific hashes, 32 records, 16/8/8, and the frozen split/checkpoint/profile.

- [ ] **Step 3: Verify descriptor A and B**

For both require:

```text
descriptor_collection_scientific_sha256 =
eb967822725e730ee2eb8afa3a5c8e28b4657141aa920d6a688ab370c70c6dd9
descriptor_sample_coverage_sha256 =
27d064db21b5c699503be32e414d579bd1aa7158f1d9b141de26555fc79bc6df
```

Call `compare_descriptor_artifact_collections` on the two verified values and
require `scientifically_equivalent = true`.

- [ ] **Step 4: Resolve canonical MVTec source by provenance**

Use stable IDs and accepted source provenance in the upstream manifests to
validate the production dataset root. Require zero VisA/target-domain records.
Do not accept a root solely because its absolute path resembles an expected
path.

---

### Task 8: Execute accepted-input dry-runs and bind the plan

**Files:**

- No tracked file changes.

**Interfaces:**

- Consumes: verified teacher A, descriptor A/B, official config.
- Produces: two complete no-write dry-run result payloads.

- [ ] **Step 1: Snapshot both prospective output parents**

Record directory contents before each dry-run. Choose nonexistent output
directories under ignored
`artifacts/phase_b/b2_contribution_targets/`.

- [ ] **Step 2: Run Dry-run A in a fresh CPU process**

Invoke `tools/create_b2_contribution_targets.py` with teacher A, descriptor A,
the canonical MVTec root, official config, seed 0, `--dry-run`, and
`--expected-plan-sha256` set to the accepted SHA.

Require status passed, no path change, 32 planned samples, 16/8/8 targets,
16/0/0 calibration and normalization membership, coalition counts 4/8/16,
and teacher-forward count zero.

- [ ] **Step 3: Run Dry-run B in a separate CPU process**

Repeat Step 2 with descriptor B. Do not reuse the A process or in-memory plan.

- [ ] **Step 4: Compare dry-run payloads**

Require both plan hashes equal:

```text
fa3d2435d684a310c81c151c48717afc9455401adce41fbe4d8a96f5c776a84e
```

If either differs, stop before official generation and report the differing
scientific payload fields.

---

### Task 9: Execute independent official Run A and Run B

**Files:**

- Raw ignored run directories only.

**Interfaces:**

- Consumes: verified accepted inputs and frozen plan.
- Produces: two fresh 36-file contribution-target collections.

- [ ] **Step 1: Prove Commit 2 worktree cleanliness**

Run:

```bash
git status --porcelain --untracked-files=all
git rev-parse HEAD
```

Expected: no status output; HEAD is the official-enablement commit.

- [ ] **Step 2: Run official A with CPU resource measurement**

Create a timestamped name but not the directory. Execute in a fresh process
under `/usr/bin/time -v` with `CUDA_VISIBLE_DEVICES=""`, teacher A, descriptor
A, and the accepted expected plan SHA.

Require exactly 32 record files plus calibration, normalization, manifest, and
receipt; total 36. Record elapsed wall time and maximum resident set size.
Require zero CUDA allocated/reserved memory.

- [ ] **Step 3: Reload-verify Run A**

Call `verify_contribution_target_collection` from a separate process. Require
all record/file hashes, receipt, source-only audit, passed status, no orphan,
temporary, or lock file, and teacher-forward count zero.

- [ ] **Step 4: Run official B independently**

Repeat Steps 2–3 in another fresh process and fresh timestamped directory using
descriptor B. Do not read or copy Run A records, statistics, manifest, or
in-memory plan.

- [ ] **Step 5: Compare verified official collections**

Call `compare_contribution_target_collections`. Require every scientific
predicate true and record file-byte equality only as a diagnostic.

---

### Task 10: Semantic checks, negative controls, and evidence generation

**Files:**

- Create ignored temporary qualification-results JSON.
- Generate:
  `docs/phase_b/b2_04b_contribution_targets_manifest.json`
- Generate:
  `docs/phase_b/b2_04b_contribution_targets_report.md`

**Interfaces:**

- Consumes: accepted upstream inputs, verified official A/B, validation command
  results.
- Produces: deterministic qualification evidence.

- [ ] **Step 1: Perform six semantic spot checks**

Select one normal and one anomalous sample from each split. Recompute their
depth 12/18/24 targets from the accepted upstream maps through the existing
production scientific functions, never the backbone. Check depth-local maps,
equal-average fusion, GT and teacher components, exact Shapley, allocation, and
equality with both persisted runs.

- [ ] **Step 2: Execute the 34-case negative matrix**

Run the focused tests carrying the frozen case IDs from Task 6. Require every
case to raise `ContributionTargetError` or exit nonzero and leave no passed
manifest.

- [ ] **Step 3: Re-run final validation commands**

Run focused pytest, full CPU pytest, Ruff, and scoped mypy exactly as in Task 6.
Capture exit codes and concise summaries into the ignored deterministic
qualification-results JSON. Require all exit codes zero.

- [ ] **Step 4: Run the qualification writer**

Pass official config, Run A, Run B, qualification results, seed 0, and
`docs/phase_b` as output directory. Require:

```text
status = deterministic_dual_contribution_target_reproduction
scientifically_equivalent = true
teacher_forward_count = 0
```

Run once with `--dry-run` first and confirm no write, then run without it.

- [ ] **Step 5: Validate evidence hygiene**

Run:

```bash
rg -n '/root/|/home/|autodl|\.pt|VisA|visa' \
  docs/phase_b/b2_04b_contribution_targets_manifest.json \
  docs/phase_b/b2_04b_contribution_targets_report.md
git status --short --ignored
```

Expected: no absolute path or target-domain claim in evidence; only the two
evidence documents are untracked/nonignored. Raw runs appear ignored.

---

### Task 11: Commit evidence, review, and stop before publication

**Files:**

- Add:
  `docs/phase_b/b2_04b_contribution_targets_manifest.json`
- Add:
  `docs/phase_b/b2_04b_contribution_targets_report.md`

**Interfaces:**

- Consumes: completed deterministic qualification.
- Produces: evidence commit and focused review handoff.

- [ ] **Step 1: Hash and commit only evidence**

Run:

```bash
sha256sum \
  docs/phase_b/b2_04b_contribution_targets_manifest.json \
  docs/phase_b/b2_04b_contribution_targets_report.md
git add \
  docs/phase_b/b2_04b_contribution_targets_manifest.json \
  docs/phase_b/b2_04b_contribution_targets_report.md
git diff --cached --name-only
git commit -m "docs: record B2 contribution target reproduction"
```

Expected: exactly two staged files before commit.

- [ ] **Step 2: Prove Git hygiene**

Run:

```bash
git status --short
git ls-files 'artifacts/**' '*.pt'
git diff --stat b2-contribution-target-contract-v1..HEAD
```

Expected: status empty; no B2-04B raw run or `.pt` is tracked.

- [ ] **Step 3: Perform focused independent review**

Review `b2-contribution-target-contract-v1..HEAD` for accepted source
identities, plan binding, clean committed generation, independent runs, all
layered identities and 32 records, leakage isolation, zero teacher forward,
source-only access, Git hygiene, and production verification usage.

Require:

```text
Critical = 0
Important = 0
```

- [ ] **Step 4: Stop and return the review handoff**

Report the three commit SHAs, run locations and resource measurements,
identities, comparison predicates, semantic and negative results, validation
commands, evidence hashes, review verdict, diff stat, clean status, and all
scope exclusions.

Do not push, open a PR, tag, merge, train DLCM, or start B2-05.
