# B3-07 Paper Results Update Design

## Goal

Create a versioned, paper-facing B3-07 result update that consolidates the
accepted DLCM/LSE evidence and the closed early-exit negative result without
changing any B2, B3-06, B4-01, or B4-02 frozen artifact.

## Scope

B3-07 will add a new paper results summary, a machine-readable update manifest,
and a focused evidence index under `docs/phase_b`. The update will state that
DLCM V5 is the accepted sample-adaptive layer-fusion mechanism, LSE is qualified
supporting validation, and early-exit is a negative result under the conservative
contract with full-depth fallback retained.

The update is documentation and evidence closure only. It will not train,
evaluate, access Final content, materialize model artifacts, change beta*, alter
accepted identities, modify existing evidence files, or push/open a PR.

## Alternatives Considered

1. Rewrite B4-02 release documents. Rejected because B4-02 is frozen and its
   manifest and SHA sidecars would no longer describe the released bytes.
2. Add unstructured prose only. Rejected because paper claims need a stable
   machine-readable identity and an auditable source chain.
3. Add a versioned B3-07 update that references frozen source identities.
   Selected because it preserves the existing release and gives the paper a
   precise, independently verifiable result statement.

## Architecture

A pure B3-07 builder will read only four frozen source manifests:

- `b2_08_paper_results_manifest.json` for accepted DLCM/LSE identities and LSE
  qualification values.
- `b3_06_early_exit_phase_closure_manifest.json` for the negative-result
  decision, zero positive signals, and full-depth fallback.
- `b4_01_dlcm_adaptive_weight_evidence_manifest.json` for the sample-adaptive
  weight evidence.
- `b4_02_final_local_paper_release_manifest.json` for the final local release
  identity and its already-frozen paper position.

The builder will fail closed when a required schema, identity, claim, or source
value differs from the frozen contract. On success it writes only new B3-07
files atomically: a canonical JSON manifest plus its SHA-256 sidecar, a Markdown
summary plus sidecar, and a Markdown evidence index plus sidecar.

## Result Contract

The B3-07 manifest will carry these immutable source identities:

```text
accepted_dlcm_identity = 0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116
v5_deployment_identity = c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd
accepted_lse_identity = 3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa
b3_06_phase_closure_identity = a984814c1821dbc6c0b2ee49fbf018be0c8b4f2fe226855f6b3e015eb89e05be
b4_01_weight_evidence_identity = 68bcea45e1fe98ffbee9f9ea51a2b645916b4a623198f787ce8830b1b0f8fe79
final_release_identity = 296191577c12aa42e2e4dbad3d34deaef67b04bbd34d3d0f52be20b9e1c99b93
```

It will also record only existing result values: `beta*=0.54`, LSE qualified
with calibration NLL `0.4768362585455179`, calibration records `8`, early-exit
positive targets/signals `0`, candidate depths `[12, 18]`, and fallback depth
`24`.

## Testing And Verification

Tests will prove successful materialization from valid frozen fixtures, reject
identity/schema/claim drift, reject writes that would overwrite a frozen source,
and ensure the dry run reports no training, evaluation, Final access, run
directory, artifact, or `.pt` output. Verification will include focused tests,
full CPU tests, Ruff, scoped mypy, SHA sidecars, `tracked .pt = 0`, no active
training/evaluation process, and a clean B3-07 worktree after the local commit.

## Deliverables

- `rad/phase_b/b3_paper_results_update.py`
- `tools/close_b3_paper_results_update.py`
- `tests/rad/test_b3_paper_results_update.py`
- `docs/phase_b/b3_07_paper_results_update_manifest.json` and `.sha256`
- `docs/phase_b/b3_07_paper_results_update.md` and `.sha256`
- `docs/phase_b/b3_07_paper_evidence_index.md` and `.sha256`
