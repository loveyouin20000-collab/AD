# Phase B2 Gate C Design

## Scope

This increment contains only B2-00 frozen execution-profile enforcement and
B2-01 deterministic tiny MVTec source splitting. It starts from
`b1-strict-independent-v1` at
`3a751b2784a50eb0a08ed49e1db2df0b53608ccc`.

The transfer direction is MVTec to VisA, but this increment may access only
MVTec. VisA is a forbidden target dataset and must not be enumerated, opened,
or hashed. The seed is 111. Generated artifacts remain ignored beneath
`artifacts/phase_b/b2_gate_c/`.

No teacher caches, anomaly maps, descriptors, normalization statistics,
Shapley targets, DLCM checkpoints, residual-gain targets, LSE checkpoints, or
policy-calibration artifacts are in scope.

## Architecture

`tools/run_with_execution_profile.py` is a strictly standard-library launcher.
It validates the exact profile bytes and schema, sets the required environment
before any possible torch import, exports a bootstrap marker plus validated
profile path and hash, and launches the requested command in a fresh process.
It rejects an empty command and propagates the child exit code exactly.

`rad/runtime/execution_profile.py` applies the validated profile after torch is
imported. Runtime provenance is returned as an immutable controlled object.
The object owns profile identity, launcher/runtime hash agreement, requested
settings, effective settings, environment identities, canary evidence, and the
canonical runtime-attestation SHA-256. Artifact callers receive provenance
fields from this object and cannot construct them manually.

“Controlled” is an API and trusted-process boundary: there is no public
constructor or issuer, and artifact builders reject objects lacking the
per-process seal. It does not claim secrecy against malicious same-process
Python reflection (for example, extracting closure cells), which Python cannot
provide without moving the trust boundary into a native or separate process.

`rad/phase_b/b2_tiny_split.py` contains source-only split logic. It constructs
the production MVTec adapter through the production registry, derives stable
relative identities, validates masks and path provenance, sorts before seeded
stratification, and builds the complete official manifest as a pure function.
The same pure construction path is used by dry-run and official execution.

`tools/create_b2_tiny_split.py` coordinates arguments, runtime attestation,
split construction, and output. Its top level avoids aggregate imports that
transitively load torch before profile bootstrap.

Existing `rad.artifacts.atomic_write_json`, production dataset adapters, and
streaming SHA-256 behavior are reused. Narrow B2 helpers may wrap these
utilities when their current checkpoint-oriented namespace is unsuitable, but
no parallel general configuration, hashing, or manifest framework is created.

## Execution-Profile Contract

The accepted profile is
`configs/execution/frozen_deterministic_math.json` with exact SHA-256
`7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d`.
All missing, altered, late, or ineffective requirements fail closed.

The launcher performs no torch, torchvision, CLIP, VisualAD, or transitive
torch import. Runtime application verifies the bootstrap marker, launcher
hash, current file hash, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and pre-import CUDA
state before applying and observing all frozen backend settings.

The deterministic canary has two checks:

1. self-repeatability: two executions in one configured runtime are identical;
2. independent reconstruction: separately reconstructed inputs and module
   execution are identical under the same profile.

Both results are part of the immutable runtime attestation. Failure of either
check prevents a valid artifact.

## Tiny-Split Contract

Only `bottle` and `carpet` are selected. Each category contributes:

- training: four normal and four anomalous samples;
- calibration: two normal and two anomalous samples;
- evaluation: two normal and two anomalous samples.

The resulting 16/8/8 split contains exactly 32 pairwise-disjoint samples.
Every anomalous sample must resolve to a valid mask. Every record must come
from the production MVTec adapter and must not resolve under tests, fixtures,
examples, synthetic assets, or the forbidden target dataset.

Selection depends only on canonical source records, stable IDs, the tracked
specification, and seed 111. Filesystem iteration order is never an input.
Unknown categories, insufficient strata, missing masks, or changed dataset
state fail rather than substitute data.

## Manifest and Dry-Run Semantics

Manifest construction is pure: validated records, immutable runtime
provenance, repository identity, specification identity, and run metadata are
inputs; a complete manifest dictionary and canonical scientific-content hash
are outputs.

Dry-run executes the complete profile, production-adapter enumeration,
deterministic selection, audit, in-memory official manifest construction, and
canonical scientific-content hashing. It creates no run directory, manifest,
temporary file, or lock.

Official execution uses the same constructed content, then performs
official-only output-collision checks and atomic writing. A parity test requires
the dry-run canonical hash to equal the subsequent official-run hash under the
same profile, source state, and seed.

Canonical scientific content excludes only run ID, creation timestamp, and
output-directory path. It includes sample IDs, memberships, categories,
labels, mask identities, dataset identity, specification hash, complete
source-list hash, and execution-profile hash.

## Testing and Failure Handling

Every production-code change follows an observed failing pytest. B2-00 and
B2-01 are implemented as separate current gates, with only one active
production-code change at a time. Negative CLI checks must exit nonzero, name
the failed requirement, and leave no valid passed manifest.

Focused tests precede the CPU suite. Created or modified Python files are
checked by Ruff and by mypy with explicit package bases. CUDA-heavy execution,
teacher checkpoint loading, and all downstream artifact generation are
forbidden.

## Release Evidence

The sprint review records B1 cleanup and identity, worktree identity,
RED-to-GREEN command evidence, import-boundary proof, runtime attestation and
canary evidence, all selected IDs and audits, dry-run/official parity, two
official reproducibility hashes, negative exit codes, CPU tests, Ruff, mypy,
artifact paths, diff/stat/status, and recommended commit grouping. No commit,
push, merge, B1 history change, or B1 tag movement occurs.
