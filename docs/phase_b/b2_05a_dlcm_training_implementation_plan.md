# B2-05A DLCM Training Implementation Plan

## Base

- Worktree: `/root/autodl-tmp/AD-phase-b2-dlcm-training-contract`
- Branch: `phase-b2-dlcm-training-contract`
- Tag: `b2-contribution-target-artifacts-v1` → `97a4f497f6f2b096dd4a339555f81e7296ec3035`
- Python: `/root/miniconda3/envs/rad-visualad/bin/python` (3.10.20)

## Deliverables

| Path | Role |
|------|------|
| `rad/phase_b/b2_dlcm.py` | Architecture, dropout, losses, fusion |
| `rad/phase_b/b2_dlcm_training.py` | Lifecycle, env, scheduler, trace, transactions |
| `rad/phase_b/b2_dlcm_deployment.py` | Export, loader, gates, metrics helpers |
| `tools/train_b2_dlcm.py` | Contract-only CLI |
| `tools/verify_b2_dlcm_artifacts.py` | Artifact verification CLI |
| `configs/phase_b/b2_dlcm_training_contract_v1.json` | Tracked contract |
| `tests/rad/b2_dlcm_fixtures.py` | Hermetic fixtures |
| `tests/rad/test_b2_dlcm_*.py` | RED→GREEN suites |
| `docs/phase_b/b2_05a_dlcm_training_architecture.md` | Architecture |
| `docs/phase_b/b2_05a_dlcm_training_implementation_plan.md` | This plan |

## Story map

### Story 1 — Architecture and losses

Sections 3–10, 17, 44. Code: `b2_dlcm.py`. Tests: `test_b2_dlcm_model.py`,
`test_b2_dlcm_losses.py`. Commit: `feat: define B2 DLCM architecture and losses`.

### Story 2 — Deterministic training state

Sections 11–16, 18–32. Code: `b2_dlcm_training.py`. Tests:
`test_b2_dlcm_training.py`, fixtures. Commit:
`feat: add deterministic B2 DLCM training lifecycle`.

### Story 3 — Qualification and deployment contracts

Sections 33–55, CLIs, docs. Code: `b2_dlcm_deployment.py`, tools, config.
Tests: deployment / artifacts / CLI / portability. Commit:
`feat: add B2 DLCM qualification and deployment contracts`.

## Error codes (selected)

| Code | Meaning |
|------|---------|
| `B2_DLCM_REAL_TRAINING_NOT_ENABLED` | Non-dry-run in B2-05A |
| `B2_DLCM_SEED_COLLISION` | Derived seed collision |
| `B2_DLCM_PLAYER_VOCABULARY_MISMATCH` | Invalid players for depth |
| `B2_DLCM_NONFINITE_LOSS` | Seed failure on nonfinite loss |
| `B2_DLCM_RESUME_FORBIDDEN` | Passed/failed seed resume |
| `B2_DLCM_NOT_ACCEPTED` | Formal loader without accepted manifest |
| `B2_DLCM_GOLDEN_*` | CPU golden self-test failures |
| `B2_DLCM_GPU_QUAL_FAIL` | GPU vs CPU drift |

## Validation

Focused pytest per story; full `tests/rad`; Ruff; scoped mypy; dry-run twice
with argument permutation proving zero artifact writes and
`teacher_forward_count = 0`.

## Explicit non-goals

No authoritative seed training, no real DLCM checkpoints, no evaluation unlock,
no accepted deployment from real artifacts, no residual-gain/LSE/policy, no
teacher forward, no push/PR before review.
