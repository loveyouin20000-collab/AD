# B3-07 Paper Results Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a versioned B3-07 paper-results update that fail-closed binds the accepted DLCM/LSE evidence and the B3-06 early-exit negative result, without changing any frozen scientific artifact.

**Architecture:** A pure Python builder will load four frozen JSON manifests, validate schemas, identities, claims, boundaries, and the fixed B3-06 negative-result state, then produce a self-identifying B3-07 manifest. A small CLI will expose a read-only dry-run and an explicit B3-07-only materialization path, writing JSON/Markdown evidence plus SHA-256 sidecars; it will never invoke training, evaluation, Final resolution, or model serialization.

**Tech Stack:** Python standard library, pytest, JSON, Markdown, SHA-256, existing `rad.phase_b` evidence helpers.

## Global Constraints

- Read only `docs/phase_b/b2_08_paper_results_manifest.json`, `docs/phase_b/b3_06_early_exit_phase_closure_manifest.json`, `docs/phase_b/b4_01_dlcm_adaptive_weight_evidence_manifest.json`, and `docs/phase_b/b4_02_final_local_paper_release_manifest.json` as scientific source manifests.
- Preserve the accepted DLCM identity `0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116`, V5 deployment identity `c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd`, accepted LSE identity `3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa`, B3-06 closure identity `a984814c1821dbc6c0b2ee49fbf018be0c8b4f2fe226855f6b3e015eb89e05be`, B4-01 weight-evidence identity `68bcea45e1fe98ffbee9f9ea51a2b645916b4a623198f787ce8830b1b0f8fe79`, and B4-02 final-release identity `296191577c12aa42e2e4dbad3d34deaef67b04bbd34d3d0f52be20b9e1c99b93`.
- Preserve `beta_star_decimal = "0.54"`, LSE qualification `verdict = "qualified"`, B3-06 `early_exit_accepted_mechanism = false`, B3-06 `full_depth_fallback_retained = true`, target depths `[12, 18]`, fallback depth `24`, and zero positive exit signals.
- Do not modify B2, B3-06, B4-01, or B4-02 evidence files or SHA-256 sidecars.
- Dry-run and materialization must report `training_started = false`, `evaluation_started = false`, `final_content_accessed = false`, `model_artifact_generated = false`, `pushed = false`, `pr_opened = false`, and `tracked_pt_files = 0`.
- The workflow does not push, open a PR, run LSE, evaluate a model, resolve Final records, or create a `.pt` file.

---

### Task 1: Define the B3-07 fail-closed contract in tests

**Files:**
- Create: `tests/rad/test_b3_paper_results_update.py`
- Create: `rad/phase_b/b3_paper_results_update.py`

**Interfaces:**
- Consumes: four mapping fixtures that model the B2-08, B3-06, B4-01, and B4-02 frozen manifests.
- Produces: `B3PaperResultsUpdateError(code: str, detail: str)` and `build_b3_paper_results_update_manifest(*, b2_manifest: Mapping[str, Any], b3_manifest: Mapping[str, Any], b4_weight_manifest: Mapping[str, Any], b4_release_manifest: Mapping[str, Any], tracked_pt_count: int) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing test**

```python
from rad.phase_b import b3_paper_results_update as update


def test_builds_a_versioned_update_from_frozen_evidence() -> None:
    result = update.build_b3_paper_results_update_manifest(
        b2_manifest=_b2(), b3_manifest=_b3(), b4_weight_manifest=_b4_weight(),
        b4_release_manifest=_b4_release(), tracked_pt_count=0,
    )

    assert result["schema_version"] == "b3_07_paper_results_update_manifest_v1"
    assert result["status"] == "paper_results_update_frozen_locally"
    assert result["paper_claims"]["dlcm_sample_adaptive_fusion_supported"] is True
    assert result["paper_claims"]["lse_qualified"] is True
    assert result["paper_claims"]["early_exit_accepted_mechanism"] is False
    assert result["boundary"]["training_started"] is False
```

Add fixtures that reproduce the accepted B2/B3/B4 source schemas and identities. Add three independent negative tests: a wrong B4-02 final-release identity must raise `B3_PAPER_RESULTS_UPDATE_IDENTITY_MISMATCH`; a B3 manifest accepting early-exit must raise `B3_PAPER_RESULTS_UPDATE_EARLY_EXIT_CLAIM_INVALID`; and `tracked_pt_count=1` must raise `B3_PAPER_RESULTS_UPDATE_TRACKED_PT`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rad/test_b3_paper_results_update.py -q`

Expected: FAIL during collection because `rad.phase_b.b3_paper_results_update` does not exist.

- [ ] **Step 3: Commit the red test**

```bash
git add tests/rad/test_b3_paper_results_update.py
git commit -m "test: define B3-07 paper results update contract"
```

### Task 2: Implement the pure manifest builder

**Files:**
- Modify: `rad/phase_b/b3_paper_results_update.py`
- Test: `tests/rad/test_b3_paper_results_update.py`

**Interfaces:**
- Consumes: the four source mappings and `tracked_pt_count` from Task 1.
- Produces: canonical JSON SHA-256 helpers, `load_json`, and a manifest with `update_identity`, `bound_identities`, `paper_claims`, `result_summary`, `source_documents`, and `boundary`.

- [ ] **Step 1: Write the failing test**

```python
def test_rejects_missing_lse_qualification() -> None:
    b2 = _b2()
    b2["lse_qualification"] = {"verdict": "unqualified"}

    with pytest.raises(update.B3PaperResultsUpdateError) as exc:
        update.build_b3_paper_results_update_manifest(
            b2_manifest=b2, b3_manifest=_b3(), b4_weight_manifest=_b4_weight(),
            b4_release_manifest=_b4_release(), tracked_pt_count=0,
        )

    assert exc.value.code == "B3_PAPER_RESULTS_UPDATE_LSE_INVALID"
```

Add tests that reject a schema mismatch, a mismatch between the B2 and B4-02 accepted DLCM identity, adaptive evidence with `sample_adaptive_variation_observed = false`, and any source boundary flag that is not `false`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rad/test_b3_paper_results_update.py -q`

Expected: FAIL because the builder does not yet enforce `B3_PAPER_RESULTS_UPDATE_LSE_INVALID`.

- [ ] **Step 3: Write minimal implementation**

Implement `B3PaperResultsUpdateError`, `_fail`, `_require_schema`, `_nested`, `_require_equal`, `_require_false`, `canonical_json_sha256`, `sha256_file`, `load_json`, and `build_b3_paper_results_update_manifest`. Require all six fixed source identities to agree through their official source fields. Require B2 LSE `verdict == "qualified"`; require B4-01 adaptive variation true and uniform-equivalence false; require B3-06 negative early-exit/full-depth claims; require B4-02 primary claims to retain accepted DLCM and qualified LSE while rejecting an accepted early-exit mechanism. Populate `update_identity` from the canonical payload before the identity field is inserted.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/rad/test_b3_paper_results_update.py -q`

Expected: PASS with all positive and fail-closed cases passing.

- [ ] **Step 5: Commit**

```bash
git add rad/phase_b/b3_paper_results_update.py tests/rad/test_b3_paper_results_update.py
git commit -m "feat: build B3-07 paper results update"
```

### Task 3: Add the B3-07 dry-run and evidence materialization CLI

**Files:**
- Create: `tools/close_paper_results_update_b3_07.py`
- Modify: `tests/rad/test_b3_paper_results_update.py`

**Interfaces:**
- Consumes: `build_b3_paper_results_update_manifest` and four CLI manifest paths, defaulting to the four frozen documents listed in Global Constraints.
- Produces: `b3_07_paper_results_update_manifest.json`, `b3_07_paper_results_update.md`, `b3_07_paper_evidence_index.md`, and a `.sha256` sidecar for each file only when `--dry-run` is absent.

- [ ] **Step 1: Write the failing test**

```python
def test_cli_dry_run_is_read_only_and_prints_update_identity(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "tools/close_paper_results_update_b3_07.py", "--dry-run", "--output-dir", str(tmp_path)],
        check=True, capture_output=True, text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["boundary"]["training_started"] is False
    assert payload["boundary"]["evaluation_started"] is False
    assert not list(tmp_path.iterdir())
```

Add a materialization test using fixture manifests and a temporary output directory. Assert exactly three named content files and their sidecars are created, each sidecar equals `sha256_file(content) + "  " + content.name + "\n"`, and none of the B2/B3-06/B4 source paths is modified.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rad/test_b3_paper_results_update.py -q`

Expected: FAIL because `tools/close_paper_results_update_b3_07.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement `parse_args`, `_resolve`, `_tracked_pt_count`, `_write_json`, `_write_text`, `_paper_results_summary`, `_evidence_index`, and `main`. In dry-run print only the deterministic JSON payload. In materialization, allow output only at the requested `docs/phase_b`-style B3-07 destination and write atomically through a sibling temporary file followed by `Path.replace`; then write sidecars after each content file. The Markdown must distinguish: accepted DLCM sample-adaptive fusion, qualified supporting LSE validation, and early-exit as a negative result/future work with full-depth fallback.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/rad/test_b3_paper_results_update.py -q`

Expected: PASS, including read-only dry-run and source-preservation assertions.

- [ ] **Step 5: Commit**

```bash
git add tools/close_paper_results_update_b3_07.py tests/rad/test_b3_paper_results_update.py
git commit -m "feat: materialize B3-07 paper results evidence"
```

### Task 4: Materialize, hash, and freeze the official B3-07 evidence

**Files:**
- Create: `docs/phase_b/b3_07_paper_results_update_manifest.json`
- Create: `docs/phase_b/b3_07_paper_results_update_manifest.json.sha256`
- Create: `docs/phase_b/b3_07_paper_results_update.md`
- Create: `docs/phase_b/b3_07_paper_results_update.md.sha256`
- Create: `docs/phase_b/b3_07_paper_evidence_index.md`
- Create: `docs/phase_b/b3_07_paper_evidence_index.md.sha256`

**Interfaces:**
- Consumes: the Task 3 CLI and the four frozen source manifests.
- Produces: immutable B3-07 paper-facing evidence with a deterministic `update_identity` and verified SHA-256 sidecars.

- [ ] **Step 1: Write the failing test**

```python
def test_repository_b3_07_evidence_is_hash_valid_and_boundary_preserving() -> None:
    manifest = update.load_json("docs/phase_b/b3_07_paper_results_update_manifest.json")
    assert manifest["status"] == "paper_results_update_frozen_locally"
    assert manifest["boundary"]["final_content_accessed"] is False
    for name in (
        "b3_07_paper_results_update_manifest.json",
        "b3_07_paper_results_update.md",
        "b3_07_paper_evidence_index.md",
    ):
        assert _sidecar_matches(Path("docs/phase_b") / name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/rad/test_b3_paper_results_update.py -q`

Expected: FAIL because the official B3-07 evidence files do not yet exist.

- [ ] **Step 3: Materialize the official evidence**

Run:

```bash
python tools/close_paper_results_update_b3_07.py
python -m pytest tests/rad/test_b3_paper_results_update.py -q
```

Expected: the CLI emits the B3-07 manifest with all boundary flags false; the focused suite passes and all three SHA-256 sidecars match.

- [ ] **Step 4: Run final verification**

Run:

```bash
python tools/close_paper_results_update_b3_07.py --dry-run
python -m pytest tests/rad/test_b3_paper_results_update.py -q
python -m pytest tests/rad -q
ruff check rad/phase_b/b3_paper_results_update.py tools/close_paper_results_update_b3_07.py tests/rad/test_b3_paper_results_update.py
mypy rad/phase_b/b3_paper_results_update.py tools/close_paper_results_update_b3_07.py
git ls-files '*.pt'
git status --short
```

Expected: dry-run identifies the same update, all tests/lint/type checks pass, `git ls-files '*.pt'` emits no paths, no LSE/training/evaluation process is started, and the only intended changes are B3-07 implementation, tests, plan/spec, and evidence files.

- [ ] **Step 5: Commit**

```bash
git add docs/phase_b/b3_07_paper_results_update_manifest.json docs/phase_b/b3_07_paper_results_update_manifest.json.sha256 docs/phase_b/b3_07_paper_results_update.md docs/phase_b/b3_07_paper_results_update.md.sha256 docs/phase_b/b3_07_paper_evidence_index.md docs/phase_b/b3_07_paper_evidence_index.md.sha256 tests/rad/test_b3_paper_results_update.py
git commit -m "docs: freeze B3-07 paper results update"
```

## Self-Review

1. **Spec coverage:** Task 1 pins the accepted positive/negative paper claims and base boundary. Task 2 verifies every source schema, identity, qualification verdict, adaptive-weight condition, and negative early-exit state. Task 3 makes dry-run read-only and restricts writes to new B3-07 evidence. Task 4 materializes and checks SHA sidecars, focused/full tests, lint/type checks, untracked model files, and worktree state. No task modifies B2, B3-06, B4-01, or B4-02.
2. **Placeholder scan:** The plan contains no unresolved markers and names each production API, output file, verification command, expected failure, and commit boundary.
3. **Type consistency:** The test, builder, and CLI all use `build_b3_paper_results_update_manifest`, `B3PaperResultsUpdateError`, `load_json`, and `sha256_file`; the generated identity is consistently named `update_identity`.
