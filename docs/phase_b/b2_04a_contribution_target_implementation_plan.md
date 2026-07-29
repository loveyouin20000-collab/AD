# B2-04A Contribution-Target Implementation Plan

## Base

- Worktree: `/root/autodl-tmp/AD-phase-b2-contribution-target-contract`
- Branch: `phase-b2-contribution-target-contract`
- Tag: `b2-descriptor-tokenize-v1` → `bf68a7ba546603535356cfc3222b2bfe9a0b35f8`
- Python: `/root/miniconda3/envs/rad-visualad/bin/python` (3.10.20)

## Deliverables

| Path | Role |
|------|------|
| `rad/phase_b/b2_contribution_targets.py` | Domain module |
| `tools/create_b2_contribution_targets.py` | CLI |
| `configs/phase_b/b2_contribution_targets_gate_c.json` | Tracked config |
| `tests/rad/b2_contribution_target_fixtures.py` | Hermetic fixtures |
| `tests/rad/test_b2_contribution_targets.py` | Domain tests |
| `tests/rad/test_b2_contribution_targets_cli.py` | CLI tests |
| `tests/rad/test_b2_contribution_targets_portability.py` | Portability / leakage |
| `docs/phase_b/b2_04a_contribution_target_architecture.md` | Architecture |
| `docs/phase_b/b2_04a_contribution_target_implementation_plan.md` | This plan |

## Story 1 — Mathematical contracts

TDD for coalition encoding/fusion, GT calibration, GT/teacher utilities,
exact Shapley, allocation fallback.

Commit: `feat: define B2 dual contribution target mathematics`

## Story 2 — Records, statistics and identities

Sample schema, calibration/normalization artifacts, scientific hashes, split
coverage, collection identity, plan hash, training-access leakage helpers.

Commit: `feat: add B2 contribution target artifact contracts`

## Story 3 — Persistence and dry-run CLI

Atomic dual-hash persistence, final manifest + receipt, fresh-run-only,
complete dry-run, official-mode disabled gate, hermetic CLI tests.

Commit: `feat: add B2 contribution target qualification CLI`

## Validation

After each story: focused pytest. Final:

```bash
CUDA_VISIBLE_DEVICES="" \
  /root/miniconda3/envs/rad-visualad/bin/python \
  -m pytest tests/rad -q --tb=short
```

Plus Ruff on new/modified Python, scoped mypy on domain module and CLI,
hermetic dry-run twice with input-order variation, negative dry-run matrix,
training-access leakage tests.

## Explicit non-goals

No real accepted contribution-target artifacts. No DLCM/LSE/residual-gain/
policy. No teacher forward. No push/PR before review.
