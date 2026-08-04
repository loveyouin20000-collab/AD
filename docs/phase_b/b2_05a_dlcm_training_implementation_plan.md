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
| `rad/phase_b/b2_dlcm_deployment.py` | Export, loader, unlock, eval, gates, accepted |
| `tools/train_b2_dlcm.py` | Contract-only CLI |
| `tools/verify_b2_dlcm_artifacts.py` | Artifact verification CLI |
| `configs/phase_b/b2_dlcm_training_contract_v1.json` | Tracked contract |
| `tests/rad/b2_dlcm_fixtures.py` | Hermetic fixtures |
| `tests/rad/test_b2_dlcm_*.py` | RED→GREEN suites |
| `tests/rad/test_b2_dlcm_gpu_qualification.py` | GPU entrypoint (skip in CPU CI) |
| `docs/phase_b/b2_05a_dlcm_training_architecture.md` | Architecture |
| `docs/phase_b/b2_05a_dlcm_training_implementation_plan.md` | This plan |

## Contract → code / tests / story map

| Sections | Production | Tests | Story |
|----------|------------|-------|-------|
| 3–10, 17, 44 | `b2_dlcm.py` | `test_b2_dlcm_model/losses` | 1 |
| 11–16, 18–32 | `b2_dlcm_training.py` | `test_b2_dlcm_training` | 2 |
| 33–39, 40–52 | `b2_dlcm_deployment.py` | `test_b2_dlcm_deployment`, `test_b2_dlcm_evaluation_contracts`, `test_b2_dlcm_gpu_qualification` | 3 |
| 53–55 | config + CLIs + fixtures | `test_b2_dlcm_cli/artifacts/portability` | 3 |
| 56–58 | — | all above + full `tests/rad` | validation |
| 59–60 | docs | — | docs/handoff |

## Story commits

1. `feat: define B2 DLCM architecture and losses`
2. `feat: add deterministic B2 DLCM training lifecycle`
3. `feat: add B2 DLCM qualification and deployment contracts`
4. `feat: complete B2 DLCM GPU qual, unlock, and evaluation contracts`

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
| `B2_DLCM_EVAL_LOCKED` | Missing/invalid evaluation unlock |
| `B2_DLCM_EVAL_CLI_BYPASS` | CLI boolean unlock rejected |
| `B2_DLCM_EVAL_METRIC_UNDEFINED` | Formal localization undefined |
| `B2_DLCM_STATE_MUTATION` | Immutable wrapper state drift |

## Validation

Focused pytest per story; full `tests/rad`; Ruff; scoped mypy; dry-run twice
with argument permutation proving zero artifact writes and
`teacher_forward_count = 0`. GPU entrypoint runs when CUDA is visible; CPU CI
skips with an explicit reason.

## Explicit non-goals

No authoritative seed training, no real DLCM checkpoints, no real evaluation
unlock from accepted artifacts, no residual-gain/LSE/policy, no teacher forward,
no push/PR before review.
