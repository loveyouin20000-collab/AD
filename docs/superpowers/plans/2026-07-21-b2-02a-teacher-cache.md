# B2-02A Teacher-Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:test-driven-development to implement this plan task-by-task.
> Do not commit before human review.

**Goal:** Build a fail-closed, CPU-tested teacher-cache planner, tensor
contract, dual-hash persistence path, resume validator, and no-write dry-run
CLI without executing a real GPU teacher.

**Architecture:** Pure cache-domain logic lives in
`rad/phase_b/b2_teacher_cache.py`; the CLI owns repository/input checks and
filesystem coordination. Existing production functions construct causal
outputs, sum-preserving maps, descriptors, and image scores. Option A stores
the scientific record plus scientific hash in `.pt`, then records the
completed file-byte hash in partial/final manifest entries.

**Tech Stack:** Python 3.10.20, PyTorch 2.0.0, pytest, Ruff, mypy, existing B2
execution-profile/split infrastructure, and existing VisualAD production
interfaces.

## Global Constraints

- Base tag: `b2-tiny-split-v1`.
- Base commit: `18bac047227754c975b23b46842458a5b41d5e2a`.
- Branch: `phase-b2-teacher-cache`.
- Worktree: `/root/autodl-tmp/AD-phase-b2-teacher-cache`.
- B1 tag/commit:
  `b1-strict-independent-v1` /
  `3a751b2784a50eb0a08ed49e1db2df0b53608ccc`.
- Split acceptance identity: V2 /
  `91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0`.
- V1 hash is migration provenance only.
- Execution profile SHA-256:
  `7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d`.
- Checkpoint SHA-256:
  `97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4`.
- Candidate layers are configuration-driven; primary layers are
  `[6, 12, 18, 24]`.
- Primary scientific tensors are float32.
- Every production behavior starts with an observed failing pytest.
- Do not execute a real GPU teacher or generate downstream Phase B artifacts.
- Do not commit before B2-02A review.

---

### Task 1: Configuration, provenance, and exact generation plan

**Files:**
- Create: `configs/phase_b/b2_teacher_cache_gate_c.json`
- Create: `tests/rad/test_b2_teacher_cache.py`
- Create: `rad/phase_b/b2_teacher_cache.py`

**Interfaces:**
- `load_teacher_cache_config(path: Path) -> TeacherCacheConfig`
- `validate_split_manifest(manifest: Mapping[str, Any], config: TeacherCacheConfig) -> None`
- `build_generation_plan(manifest: Mapping[str, Any], config: TeacherCacheConfig) -> tuple[PlannedSample, ...]`
- `validate_outer_provenance(...) -> OuterProvenance`

- [ ] **Step 1: Add the tracked fixed configuration**

Write schema/version fields, candidate layers, prediction depths, V2 split
identity, checkpoint/profile identities, B1/B2 identities, 16/8/8 counts,
tensor/descriptor/record-hash contract versions, resume policy, and explicit
fail-closed requirements. Do not place selected sample IDs in configuration.

- [ ] **Step 2: Write failing provenance and plan tests**

Tests must cover missing bootstrap, profile hash drift, missing runtime
attestation, missing split, non-V2/current V1 identity, wrong V2 hash, missing
checkpoint, wrong checkpoint hash, unresolved/moved B2 tag, non-descendant
HEAD, dirty official worktree, selected-ID drift, exact 32 IDs, and exact
16/8/8 membership.

The desired API starts with:

```python
config = load_teacher_cache_config(CONFIG)
plan = build_generation_plan(accepted_split_manifest, config)
assert len(plan) == 32
assert Counter(row.membership for row in plan) == {
    "training": 16,
    "calibration": 8,
    "evaluation": 8,
}
```

- [ ] **Step 3: Observe RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest tests/rad/test_b2_teacher_cache.py -q
```

Expected: collection fails because `rad.phase_b.b2_teacher_cache` does not
exist.

- [ ] **Step 4: Implement minimal immutable config/plan types and validators**

Use frozen dataclasses. Preserve split sample order only after validating all
IDs and memberships; reject duplicates, missing masks for anomalous samples,
extra/missing IDs, wrong counts, target access, and scientific hash drift.
Keep runtime/Git/environment values in `OuterProvenance`, not per-record
scientific content.

- [ ] **Step 5: Run focused tests to GREEN**

Run the Task 1 tests and require zero failures.

### Task 2: Exact nine-map tensor and descriptor contract

**Files:**
- Modify: `tests/rad/test_b2_teacher_cache.py`
- Modify: `rad/phase_b/b2_teacher_cache.py`

**Interfaces:**
- `expected_lattice(candidate_layers, prediction_depths) -> frozenset[MapIdentity]`
- `validate_teacher_output(output: TeacherOutput, contract: CacheContract) -> ValidatedTeacherOutput`
- `build_cumulative_maps(validated: ValidatedTeacherOutput) -> Mapping[int, torch.Tensor]`
- `reconstruct_descriptors(validated: ValidatedTeacherOutput) -> Mapping[int, torch.Tensor]`
- `compute_final_image_score(full_depth_map: torch.Tensor) -> torch.Tensor`

- [ ] **Step 1: Write failing lattice and tensor tests**

Require exact set equality:

```python
assert expected_lattice((6, 12, 18, 24), (12, 18, 24)) == {
    MapIdentity(12, 6), MapIdentity(12, 12),
    MapIdentity(18, 6), MapIdentity(18, 12), MapIdentity(18, 18),
    MapIdentity(24, 6), MapIdentity(24, 12),
    MapIdentity(24, 18), MapIdentity(24, 24),
}
```

Cover duplicate/unsorted layers, layer beyond backbone depth, missing and
extra map identities, missing descriptor source, wrong shape/dimensions,
non-float32 tensors, NaN/Inf maps, nonfinite cumulative map/image score,
missing anomalous mask, unexpected/missing sample, and explicit dimension
semantics.

- [ ] **Step 2: Write failing production-function reuse tests**

Monkeypatch only the imported production call boundary and assert:

- cumulative maps call `rad.models.dlcm.sum_preserving_fusion`;
- descriptor reconstruction calls
  `rad.models.descriptors.LayerDescriptorExtractor`;
- image scoring calls
  `rad.inference.adaptive_engine.compute_exit_signals`;
- the image-score argument is the exact sum-preserving cumulative map and not
  the extractor's equal-average reference.

- [ ] **Step 3: Write the CPU fake-teacher parity RED test**

Define the deterministic fake entirely in the test module. It emits
configurable `MapIdentity -> float32 Tensor` values and supports missing,
NaN, and Inf variants. Compute live descriptors with the existing extractor,
round-trip the future payload writer/loader, reconstruct with the same
extractor, and require `torch.equal` at depths 12, 18, and 24.

Also prove production mode rejects `artifact_kind="test_fixture"` with no CLI
flag or factory path that can select the fake.

- [ ] **Step 4: Observe RED**

Run the new lattice/reuse/parity tests and confirm failures are due to absent
contract functions.

- [ ] **Step 5: Implement minimal contract adapters**

Represent every map by an immutable `(checkpoint_depth,
candidate_layer_id)` key. Validate the exact set
`{(d, l) | d in prediction_depths, l in candidate_layers, l <= d}`.
Stack maps in explicit ascending layer order only when calling existing
production functions. Never copy fusion, descriptor, or image-score formulas.

- [ ] **Step 6: Run Task 2 tests to GREEN**

Require exact fake live/cache descriptor equality and all negative cases to
pass.

### Task 3: Versioned descriptor and canonical scientific hashing

**Files:**
- Modify: `tests/rad/test_b2_teacher_cache.py`
- Modify: `rad/phase_b/b2_teacher_cache.py`

**Interfaces:**
- `descriptor_implementation_sha256(repo_root: Path) -> str`
- `descriptor_contract(config: TeacherCacheConfig, repo_root: Path) -> Mapping[str, Any]`
- `canonical_tensor_digest(name, tensor, dimension_semantics) -> str`
- `scientific_record_content(record: Mapping[str, Any]) -> Mapping[str, Any]`
- `record_scientific_sha256(record: Mapping[str, Any]) -> str`

- [ ] **Step 1: Write failing descriptor-contract tests**

Require contract version, exact feature order, source kind
`causal_anomaly_maps`, extractor configuration digest, tracked
`rad/models/descriptors.py` byte digest, and fail-closed loading on any drift.

- [ ] **Step 2: Write failing canonical tensor tests**

Cover deterministic hashing, path/time/run independence, value sensitivity,
shape sensitivity, dtype sensitivity in the generic encoder, explicit
little-endian behavior, sorted logical names, noncontiguous input, and
rejection of sparse/quantized/unsupported/nonfinite tensors.

- [ ] **Step 3: Write failing record-whitelist tests**

Require the versioned whitelist to include sample/scientific tensor,
descriptor, V2 split, checkpoint, and execution-profile identities. Verify
that changing any listed field changes the digest. Verify runtime attestation,
commit/branch/worktree, machine/environment fields, run ID, output path,
timestamp, and file hash do not change it while remaining mandatory in outer
provenance.

- [ ] **Step 4: Observe RED**

Run only Task 3 tests and confirm missing hashing functions cause the failure.

- [ ] **Step 5: Implement canonical encoding**

Use length-delimited canonical metadata, sorted logical keys, explicit dtype,
shape and dimension semantics, and little-endian contiguous CPU bytes. Build
scientific content by naming every accepted field; do not hash arbitrary
record dictionaries and subtract excluded keys.

- [ ] **Step 6: Run Task 3 tests to GREEN**

Require all hash invariants and descriptor drift checks to pass.

### Task 4: Option A atomic persistence, resume, and coverage

**Files:**
- Modify: `tests/rad/test_b2_teacher_cache.py`
- Modify: `rad/phase_b/b2_teacher_cache.py`

**Interfaces:**
- `write_sample_atomic(path: Path, scientific_record: Mapping[str, Any]) -> PersistedSampleEntry`
- `validate_resume_state(run_dir: Path, partial_manifest: Mapping[str, Any], ...) -> tuple[PersistedSampleEntry, ...]`
- `audit_complete_coverage(run_dir: Path, plan, entries) -> None`
- `build_final_manifest(...) -> Mapping[str, Any]`

- [ ] **Step 1: Write failing Option A tests**

Assert `.pt` contains only `scientific_record` and
`record_scientific_sha256`. Assert `record_file_sha256` appears only in the
partial/final manifest entry after atomic persistence.

- [ ] **Step 2: Write failing collision/atomic tests**

Cover sample overwrite refusal, existing run refusal, interrupted temp write,
partial manifest forbidden from `status=passed`, and impossibility of a
passed manifest after interrupted persistence.

- [ ] **Step 3: Write failing resume tests**

Cover missing/invalid partial manifest, wrong scientific hash, wrong file
hash, descriptor/provenance drift, and successful explicit reuse of verified
immutable records. Resume must recompute both hashes before reuse.

- [ ] **Step 4: Write failing coverage/orphan tests**

Require exact plan/entry/file identity sets. Reject missing sample, unexpected
sample, unknown `.pt`, sidecars, temporary/lock files, duplicate path mapping,
and any other samples-directory entry.

- [ ] **Step 5: Observe RED**

Run Task 4 tests and verify failures originate from absent persistence APIs.

- [ ] **Step 6: Implement narrow persistence flow**

Use exclusive destination checks, a same-directory temporary file,
`torch.save`, flush/fsync, atomic replace, completed-file SHA-256, immediate
reload, scientific rehash, and atomic partial JSON update. Reuse
`rad.artifacts.atomic_write_json` for JSON publication; do not create a generic
artifact framework.

- [ ] **Step 7: Run Task 4 tests to GREEN**

Require successful resume and all interruption/orphan cases to pass.

### Task 5: Bootstrap-only dry-run CLI and fail-closed matrix

**Files:**
- Create: `tests/rad/test_b2_teacher_cache_cli.py`
- Create: `tools/create_b2_teacher_cache.py`
- Modify: `rad/phase_b/b2_teacher_cache.py`

**Interfaces:**
- CLI arguments: `--config`, `--seed`, `--output-dir`, `--dry-run`,
  `--split-manifest`, `--checkpoint`, `--expected-checkpoint-sha256`,
  `--output-root`, and `--resume`.
- CLI must run only under `tools/run_with_execution_profile.py`.

- [ ] **Step 1: Write failing valid dry-run test**

Execute through the launcher and require structured output containing:

```text
mode = dry_run
status = passed
artifact_written = false
run_directory_created = false
planned_samples = 32
split_scientific_hash_version = 2
split_scientific_sha256 = 91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0
checkpoint_sha256 = 97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4
candidate_layers = [6,12,18,24]
```

Snapshot the output-root tree before and after and require exact equality.
Patch the production loader to raise if called, proving dry-run does not load
VisualAD.

- [ ] **Step 2: Write failing negative CLI tests**

Cover altered V2 hash, V1 supplied as current identity, invalid checkpoint
hash, missing launcher bootstrap, dirty worktree, non-descendant branch,
output collision plan, and production selection/injection of a test teacher.
Every failure must be nonzero and must create no passed manifest.

- [ ] **Step 3: Observe RED**

Run:

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest tests/rad/test_b2_teacher_cache_cli.py -q
```

Expected: failure because the CLI does not exist.

- [ ] **Step 4: Implement stdlib-first CLI coordination**

Parse arguments without importing the production teacher. Apply controlled
runtime attestation, validate config/split/checkpoint/Git identity, construct
the exact plan and intended manifest metadata, check collision intent, and
print the summary. Return before creating directories or importing/loading
the teacher in dry-run.

Production execution keeps the same interface but is not invoked in B2-02A.
No test-teacher selector is exposed.

- [ ] **Step 5: Run CLI tests to GREEN**

Require valid no-write dry-run and all fail-closed cases to pass.

### Task 6: Complete validation and review evidence

**Files:**
- Modify only if a new failing test demonstrates a defect.
- Review:
  `docs/phase_b/b2_02a_teacher_cache_architecture.md`

**Interfaces:**
- Produces review evidence only; no real teacher/cache artifacts and no
  commit.

- [ ] **Step 1: Run focused tests**

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest tests/rad/test_b2_teacher_cache.py \
  tests/rad/test_b2_teacher_cache_cli.py -q
```

- [ ] **Step 2: Run the complete Python 3.10.20 CPU suite**

```bash
CUDA_VISIBLE_DEVICES="" /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest tests/rad -q
```

Require zero failures.

- [ ] **Step 3: Run Ruff**

```bash
/root/miniconda3/envs/rad-visualad/bin/python -m ruff check \
  rad/phase_b/b2_teacher_cache.py \
  tools/create_b2_teacher_cache.py \
  tests/rad/test_b2_teacher_cache.py \
  tests/rad/test_b2_teacher_cache_cli.py
```

- [ ] **Step 4: Run mypy with explicit package bases**

```bash
MYPYPATH=. /root/miniconda3/envs/rad-visualad/bin/python -m mypy \
  --explicit-package-bases \
  rad/phase_b/b2_teacher_cache.py \
  tools/create_b2_teacher_cache.py
```

- [ ] **Step 5: Run the required CLI matrix**

Run valid dry-run and each specified negative case through
`tools/run_with_execution_profile.py`. Record exit code, structured output or
error code, and no-write/no-passed-manifest evidence.

- [ ] **Step 6: Audit forbidden artifact classes and Git state**

Enumerate the output test locations and prove no real teacher cache,
descriptor/statistics, Shapley, DLCM, residual-gain, LSE, or policy artifact
was produced. Capture:

```bash
git diff --stat
git status --short
```

- [ ] **Step 7: Produce the review handoff**

Report worktree/branch, foundation identity, changed files, architecture
reuse, exact descriptor source and schema contracts, RED→GREEN evidence,
exact 32-sample plan, accepted hashes, dry-run output/no-write proof,
resume/atomic/negative tests, CPU/Ruff/mypy results, Git diff/status, and
recommended commit grouping. Do not commit.
