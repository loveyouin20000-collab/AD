# PR #39 CI Portability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the paper-release CI portable to GitHub-hosted runners without changing DLCM/LSE scientific artifacts, acceptance evidence, or training behavior.

**Architecture:** Supply the deterministic environment contract in the CPU workflow. Tests invoke their active interpreter and checkout, historical roster assertions use a minimal historical fixture, and LSE validates its unlock before inspecting a checkpoint.

**Tech Stack:** GitHub Actions, Python 3.10, pytest, PyYAML.

## Global Constraints

- Do not modify models, checkpoints, Final qualifications, accepted identities, or evidence payloads.
- Do not start training, evaluation, or artifact materialization.
- Keep tracked `.pt` files at zero.
- Preserve fail-closed behavior when accepted references or prerequisites are absent.

---

### Task 1: Supply the CPU Environment Contract

**Files:** `.github/workflows/ci.yml`, `tests/rad/test_b2_dlcm_best_checkpoint_identity.py`, `tests/rad/test_b2_dlcm_v3_official.py`

- [x] Write/retain the RED reproduction: both tests fail without `OMP_NUM_THREADS=4`.
- [x] Add `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `PYTHONHASHSEED=0` to the CPU pytest environment.
- [x] Run both tests with the complete environment and verify GREEN.

### Task 2: Make Test Entrypoints Host-Portable

**Files:** `tests/rad/test_b2_dlcm_cli.py`, `tests/rad/test_b2_dlcm_v2_cli.py`, `tests/rad/test_b2_dlcm_v2_contract_closure.py`, `tests/rad/test_b2_dlcm_v3_cli.py`, `tests/rad/test_b2_dlcm_v3_roster_adoption.py`, `tests/rad/test_b2_dlcm_v4_cli.py`, `tests/rad/test_b2_dlcm_v5_cli.py`

- [x] Preserve the Actions RED evidence showing hard-coded interpreter and legacy worktree paths fail on GitHub runners.
- [x] Replace hard-coded interpreter paths with `sys.executable`, derive test roots from `__file__`, and catch `OSError` in optional upstream-artifact probes.
- [x] Run the affected tests; they must pass or skip only because accepted upstream artifacts are absent.

### Task 3: Scope Historical Roster Tests Correctly

**Files:** `tests/rad/test_b2_dlcm_v3_roster_adoption.py`, `tests/rad/test_b2_dlcm_v4_roster_adoption.py`, `tests/rad/test_b2_dlcm_v5_roster_adoption.py`

- [x] Run the current tests and observe their RED failure against later B2-06F manifests.
- [x] Copy only the C1 roster and receipt to `tmp_path/docs/phase_b` before invoking the existing adoption builders.
- [x] Re-run the tests; production detection remains fail-closed when that root contains final-access artifacts.

### Task 4: Enforce Dry-Run No-Artifact-Touch Order

**Files:** `tests/rad/test_lse_train_cli_preflight.py`, `tools/train_lse.py`

- [x] Add a RED test with a present checkpoint whose hash function raises if touched before the required unlock exits.
- [x] Move checkpoint hashing after training-unlock validation and the dry-run return path.
- [x] Verify no-unlock dry runs exit `2` without reading checkpoint bytes.

### Task 5: Preserve Frozen Artifact-Bound Configs

**Files:** `tests/rad/test_integration_main_portability.py`, `tests/rad/test_b2_lse_training_unlock.py`

- [x] Preserve the portability test RED condition for mutable default configs with `/root/autodl-tmp/` values.
- [x] Exclude only the two accepted-artifact-bound configs from that mutable-default test and retain their exact bytes.
- [x] Add a regression test proving the accepted LSE config hash matches its frozen B2-06D training unlock.
- [x] Verify portability, frozen-config, and LSE preflight tests are GREEN.

### Task 6: Verify and Publish

- [x] Run all affected tests plus `ruff` and scoped `mypy`.
- [x] Confirm no tracked `.pt`, training/evaluation process, or unintended scientific-artifact change.
- [ ] Commit as `fix: make paper release CI checks portable`, push the existing branch, and inspect PR #39 checks.
