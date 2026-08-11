# B2-07 Phase Final Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a local, fail-closed B2 phase final closure manifest and handoff report that binds the accepted DLCM V5 and accepted LSE evidence chain without starting new training, evaluation, push, or PR work.

**Architecture:** Add a small Phase-B closure helper that reads tracked evidence documents from B2-06A through B2-06F, validates identity continuity, verifies boundary flags, and emits a deterministic final manifest. A CLI writes the manifest/report and SHA sidecars only after validation passes.

**Tech Stack:** Python 3.10, pytest, Ruff, mypy, existing `docs/phase_b` evidence layout.

## Global Constraints

- Do not start LSE training.
- Do not run a new LSE evaluation.
- Do not read Final content.
- Do not modify DLCM or LSE scientific outputs.
- Do not track `.pt` files.
- Do not push, open a PR, or create remote tags.
- Treat tracked docs as the authority for B2-07; ignored run artifacts are referenced only by frozen SHA.

---

### Task 1: Closure Contract Tests

**Files:**
- Create: `tests/rad/test_b2_phase_final_closure.py`

**Interfaces:**
- Consumes: `rad.phase_b.b2_phase_final_closure.build_phase_final_closure_manifest`
- Produces: tests that prove B2-07 fails closed on unqualified LSE, identity mismatch, and boundary violations.

- [ ] **Step 1: Write the failing tests**

Create tests that pass minimal dictionaries into `build_phase_final_closure_manifest` and assert:

```python
manifest["schema_version"] == "b2_07_phase_final_closure_manifest_v1"
manifest["status"] == "b2_phase_completed_locally"
manifest["accepted_lse_identity"] == "lse-id"
```

Add failure tests for:

```text
06E verdict != qualified
06F accepted_lse_identity mismatch
06F boundary says pushed=true
tracked_pt_count != 0
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/rad/test_b2_phase_final_closure.py -q
```

Expected: import failure for missing module or missing function.

### Task 2: Closure Helper And CLI

**Files:**
- Create: `rad/phase_b/b2_phase_final_closure.py`
- Create: `tools/close_b2_phase_final.py`

**Interfaces:**
- Produces: `build_phase_final_closure_manifest(...) -> dict[str, Any]`
- Produces: CLI flags `--output-dir` and `--dry-run`

- [ ] **Step 1: Implement helper**

The helper must validate:

```text
B2-06A accepted_gate_passed = true
B2-06B accepted_identity == B2-06F accepted_dlcm_identity
B2-06C accepted_gate.ready = true
B2-06D training receipt identity == B2-06E training_receipt_identity == B2-06F training_receipt_identity
B2-06E verdict = qualified
B2-06F accepted_artifact_generated = true
B2-06F accepted_lse_identity matches receipt and evidence
B2-06F checkpoint SHA matches 06D/06E/06F
tracked_pt_count = 0
push/pr flags remain false
```

- [ ] **Step 2: Implement CLI**

The CLI loads:

```text
docs/phase_b/b2_06a_lse_accepted_gate_preflight_evidence.json
docs/phase_b/b2_06b_accepted_v5_reference_packaging_evidence.json
docs/phase_b/b2_06c_lse_prerequisite_materialization_evidence.json
docs/phase_b/b2_06d_lse_first_controlled_run_evidence.json
docs/phase_b/b2_06e_lse_qualification_decision_manifest.json
docs/phase_b/b2_06f_accepted_lse_manifest.json
docs/phase_b/b2_06f_accepted_lse_closure_receipt.json
docs/phase_b/b2_06f_accepted_lse_closure_evidence.json
```

It writes:

```text
docs/phase_b/b2_07_phase_final_closure_manifest.json
docs/phase_b/b2_07_phase_final_closure_manifest.json.sha256
docs/phase_b/b2_07_phase_final_closure_report.md
docs/phase_b/b2_07_phase_final_closure_report.md.sha256
```

### Task 3: Evidence Freeze And Verification

**Files:**
- Create: `docs/phase_b/b2_07_phase_final_closure_manifest.json`
- Create: `docs/phase_b/b2_07_phase_final_closure_manifest.json.sha256`
- Create: `docs/phase_b/b2_07_phase_final_closure_report.md`
- Create: `docs/phase_b/b2_07_phase_final_closure_report.md.sha256`

- [ ] **Step 1: Run dry-run**

Run:

```bash
python tools/close_b2_phase_final.py --dry-run
```

Expected:

```text
ready = true
training_started = false
evaluation_started = false
push_performed = false
tracked_pt_count = 0
```

- [ ] **Step 2: Generate docs evidence**

Run:

```bash
python tools/close_b2_phase_final.py
```

- [ ] **Step 3: Verify**

Run:

```bash
python -m pytest tests/rad/test_b2_phase_final_closure.py -q
ruff check rad/phase_b/b2_phase_final_closure.py tools/close_b2_phase_final.py tests/rad/test_b2_phase_final_closure.py
mypy rad/phase_b/b2_phase_final_closure.py --follow-imports=skip
sha256sum -c docs/phase_b/b2_07_phase_final_closure_manifest.json.sha256
sha256sum -c docs/phase_b/b2_07_phase_final_closure_report.md.sha256
git ls-files '*.pt' | wc -l
```

- [ ] **Step 4: Commit**

Commit as:

```bash
test: specify B2 phase final closure
feat: implement B2 phase final closure
docs: freeze B2 phase final closure evidence
```
