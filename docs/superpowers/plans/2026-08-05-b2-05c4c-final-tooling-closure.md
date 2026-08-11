# B2-05C4C Final Tooling Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze V5 Final execution tooling contract, tests, dry-run plan, and local tooling tag without reading real Final content.

**Architecture:** Add small versioned Final modules for unlock validation, stable-ID resolution, materialization/evaluation A/B protocol comparison, manifests, and loader acceptance. Keep real Final access behind unlock validation and expose only hermetic fixture execution plus dry-run plan hashing in C4C.

**Tech Stack:** Python 3.10, pytest, torch, existing `rad.phase_b` V5 modules, project canonical JSON helpers, Git local tags.

## Global Constraints

- C4C belongs to C4, not C5.
- Preserve fail-closed commit `8b2dc932727f3d7ee85fe6734c1d030a90191845`.
- Preserve beta* `0.54`, Calibration A/B identity `cae406c91ec392ffd7cc6d48ec2f0c94ab78d78f905cbfe904287842a7a7278a`, V5 deployment identity `c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd`, Final roster identity `267b7b527f13f84f76f69576d01b1532005d0bb7eda792d558ce5dcce1278213`.
- Do not modify C3 checkpoint, seed, trunk, heads, normalization, beta grid, beta*, Calibration artifacts, Development result, Final roster, thresholds, or teacher policy.
- Do not read real Final content in this stage.
- Do not push, open PRs, create remote tags, start LSE, residual-gain, or early-exit.

---

### Task 1: Contract And Dry-Run Plan

**Files:**
- Create: `configs/phase_b/b2_dlcm_v5_final_execution_contract_v1.json`
- Create: `configs/phase_b/b2_dlcm_v5_final_execution_official_v1.json`
- Create: `rad/phase_b/b2_dlcm_v5_final_unlock.py`
- Modify: `tools/materialize_b2_dlcm_final_v5.py`
- Test: `tests/rad/test_b2_dlcm_v5_final_tooling_contract.py`

**Interfaces:**
- Produces: `build_final_execution_plan(config: Mapping[str, Any], repo_identity: Mapping[str, str]) -> dict[str, Any]`
- Produces: `final_execution_plan_sha256(plan: Mapping[str, Any]) -> str`
- Produces: `dry_run_status(plan_sha256: str) -> dict[str, Any]`
- Produces: `validate_materialization_unlock(unlock: Mapping[str, Any], *, expected: Mapping[str, Any]) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests** for no unlock forbidden, malformed unlock forbidden, and identical dry-run A/B plan SHA.
- [ ] **Step 2: Run tests** with `/root/miniconda3/envs/rad-visualad/bin/python -m pytest tests/rad/test_b2_dlcm_v5_final_tooling_contract.py -q --tb=short`; expected failure because module/config behavior is missing.
- [ ] **Step 3: Implement contract config, plan hashing, dry-run CLI, and unlock validator.**
- [ ] **Step 4: Re-run tests** and verify pass.
- [ ] **Step 5: Commit** with `test: specify V5 final materialization and evaluation closure`.

### Task 2: Stable ID Resolution And Materialization Transaction

**Files:**
- Create: `rad/phase_b/b2_dlcm_v5_final_resolution.py`
- Create: `rad/phase_b/b2_dlcm_v5_final_materialization.py`
- Modify: `tools/materialize_b2_dlcm_final_v5.py`
- Test: `tests/rad/test_b2_dlcm_v5_final_materialization.py`

**Interfaces:**
- Produces: `resolve_stable_ids(source_manifest: Mapping[str, Any], roster: Mapping[str, Any], *, authorized: bool) -> dict[str, Any]`
- Produces: `run_hermetic_materialization(process_label: str, records: Sequence[Mapping[str, Any]], unlock: Mapping[str, Any]) -> dict[str, Any]`
- Produces: `compare_materialization_ab(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests** for exact lookup only, no scan/fuzzy/fallback, resolution cleanup, independent A/B labels, canonical byte equality, mismatch error, and success-only unlock consumption.
- [ ] **Step 2: Run tests** and verify expected missing-feature failures.
- [ ] **Step 3: Implement resolution and hermetic materialization transaction using only fixture payloads in tests.**
- [ ] **Step 4: Re-run tests** and verify pass.

### Task 3: Evaluation, Decision, Evidence, Accepted Loader

**Files:**
- Create: `rad/phase_b/b2_dlcm_v5_final_evaluation.py`
- Create: `rad/phase_b/b2_dlcm_v5_final_manifests.py`
- Create: `rad/phase_b/b2_dlcm_v5_final_loader.py`
- Modify: `tools/evaluate_b2_dlcm_final_v5.py`
- Test: `tests/rad/test_b2_dlcm_v5_final_evaluation.py`
- Test: `tests/rad/test_b2_dlcm_v5_final_loader.py`

**Interfaces:**
- Produces: `run_hermetic_final_evaluation(process_label: str, materialization: Mapping[str, Any], unlock: Mapping[str, Any]) -> dict[str, Any]`
- Produces: `compare_evaluation_ab(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]`
- Produces: `build_final_decision_manifest(...) -> dict[str, Any]`
- Produces: `build_final_evidence_manifest(...) -> dict[str, Any]`
- Produces: `build_accepted_deployment_manifest(...) -> dict[str, Any]`
- Produces: `verify_accepted_v5_final_manifest(manifest: Mapping[str, Any], *, expected: Mapping[str, Any]) -> None`

- [ ] **Step 1: Write failing tests** for independent evaluation A/B, decision/evidence equality, no accepted loader reject, valid accepted loader accept, and forbidden Development-only candidate acceptance.
- [ ] **Step 2: Run tests** and verify expected failures.
- [ ] **Step 3: Implement evaluation comparison, manifest builders, accepted writer model, and loader verifier.**
- [ ] **Step 4: Re-run tests** and verify pass.
- [ ] **Step 5: Commit** with `feat: implement V5 final materialization and evaluation tooling`.

### Task 4: Freeze Evidence And Tooling Tag

**Files:**
- Create: `docs/phase_b/b2_05c4c_final_tooling_closure_evidence.json`
- Create: `docs/phase_b/b2_05c4c_final_tooling_closure_report.md`

**Interfaces:**
- Consumes: dry-run SHA, hermetic A/B comparison identities, verification output, Git audit.
- Produces: local tag `b2-dlcm-uniform-anchored-final-tooling-v1`.

- [ ] **Step 1: Run dry-run A/B** and confirm `real_final_content_accessed=false`, `stable_ids_resolved=false`, `materialization_started=false`, `evaluation_started=false`, `accepted_written=false`, `run_directory_created=false`, `artifact_written=false`, and equal plan SHA.
- [ ] **Step 2: Run hermetic Materialization A/B and Evaluation A/B tests.**
- [ ] **Step 3: Run focused Final tooling tests, full CPU `tests/rad`, GPU qualification where available, Ruff, scoped mypy, and Git hygiene checks.**
- [ ] **Step 4: Write evidence/report and receipts.**
- [ ] **Step 5: Commit** with `docs: freeze V5 final tooling closure evidence`.
- [ ] **Step 6: Create local annotated tag** `b2-dlcm-uniform-anchored-final-tooling-v1`.
