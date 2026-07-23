# B2-02A Teacher-Cache Architecture

## Scope and fixed identities

B2-02A starts from annotated tag `b2-tiny-split-v1` at commit
`18bac047227754c975b23b46842458a5b41d5e2a`. It designs and CPU-tests the
teacher-cache contract, planner, persistence, resume, and dry-run path. It
must not execute a real GPU teacher or generate a real cache, descriptors,
descriptor statistics, Shapley targets, DLCM checkpoints, residual-gain
targets, LSE checkpoints, or policy artifacts.

The accepted inputs are:

- split scientific hash version 2 and SHA-256
  `91570da1fed6d7859d407196b10403581832ae0ff677a1ea7657ca76b91471f0`;
- execution profile `configs/execution/frozen_deterministic_math.json` and
  SHA-256
  `7af8dba39633743da0380fef9710940cded655f68c9efa8f84f5a52aeddb3c8d`;
- checkpoint
  `/root/autodl-tmp/AD/runs/baseline/mvtec_to_visa/seed_111_official_bs8/checkpoints/epoch_2.pth`
  and SHA-256
  `97bd461163efb96e36cddb1c3adf677e4c4fc2daabb2521021689f30e799b4f4`;
- B1 tag `b1-strict-independent-v1` at
  `3a751b2784a50eb0a08ed49e1db2df0b53608ccc`.

The V1 split hash is migration provenance only and is never an acceptance
identity.

## Reused production paths

B2-02A does not reimplement VisualAD mathematics.

- `rad.data.teacher_inference.load_teacher_bundle` is the sole production
  VisualAD teacher loader.
- `rad.qualification.b1_cuda_equivalence.validate_checkpoint` performs the
  accepted non-model-loading checkpoint path and SHA-256 validation;
  `load_teacher_production` composes it with the production loader.
- `model.visual.forward_staged(image, candidate_layers)` is the staged
  backbone interface and returns `CheckpointOutput` values containing patch,
  anomaly, and normal tokens.
- `rad.data.teacher_inference.build_causal_maps_and_ingredients` is the
  production causal-output path, including checkpoint-conditioned cross
  attention and layer transforms.
- `rad.models.checkpoint_maps.anomaly_map_from_tokens` is the authoritative
  layer anomaly-map constructor.
- `rad.models.dlcm.sum_preserving_fusion` is the authoritative equal-fusion
  implementation. Equal weights produce the sum-scale prediction map.
- `rad.models.descriptors.LayerDescriptorExtractor` is the authoritative
  18-D descriptor implementation. Its internal reference map is an equal
  average, which is distinct from the sum-scale prediction map.
- `rad.inference.adaptive_engine.compute_exit_signals` is the authoritative
  image-score path and must receive the sum-preserving cumulative prediction
  map.
- `rad.losses.localization.sample_localization_error` remains the
  authoritative downstream localization-error function; B2-02A neither
  computes nor persists residual-gain targets.
- `rad.data.adapters.preprocess.preprocess_image` and `preprocess_mask` are
  the production image and nearest-neighbor binary-mask preprocessing paths.
- `rad.artifacts.atomic_write_json` and `refuse_existing_run` are reused for
  JSON publication and run-directory collision protection.
- `rad.data.cache_schema` documents the existing resumable cache, but its
  direct `torch.save` and incomplete integrity contract are not reused as the
  B2-02A persistence contract.

## Domain and CLI boundary

`rad/phase_b/b2_teacher_cache.py` owns deterministic, mostly pure domain
logic:

1. validate configuration, controlled runtime attestation, split V2
   provenance, repository identity, and checkpoint identity;
2. construct the exact 32-sample plan from the accepted split manifest;
3. validate teacher outputs against the configured candidate layers;
4. call production fusion, descriptor, and image-score functions;
5. construct canonical tensor and per-sample scientific hashes;
6. audit complete one-to-one plan coverage;
7. build partial and final manifest content;
8. validate immutable records during explicit resume.

Filesystem coordination remains outside the pure path. The thin
`tools/create_b2_teacher_cache.py` CLI validates bootstrap and inputs,
performs dry-run planning without model loading, and coordinates exclusive
claims, atomic sample writes, partial-manifest updates, and final publication.
Production execution accepts only the production teacher factory. A
test-fixture teacher has no CLI selection route and is rejected if injected
into production mode.

## Causal-map tensor contract

`A_{l|d}` means the anomaly map from candidate layer `l`, conditioned on the
teacher state at checkpoint depth `d`. The first configured layer is 6;
`A_{0|24}` does not exist.

For primary candidate layers `[6, 12, 18, 24]`, the nine-map lattice is:

- depth 12: `A_{6|12}`, `A_{12|12}`;
- depth 18: `A_{6|18}`, `A_{12|18}`, `A_{18|18}`;
- depth 24: `A_{6|24}`, `A_{12|24}`, `A_{18|24}`, `A_{24|24}`.

This is the minimal cache contract that preserves depth-specific causal
outputs without relying on cross-depth equivalence. B2-02A does not assert
equalities such as `A_{6|12} = A_{6|18} = A_{6|24}`.

Every map is identified explicitly by `(checkpoint_depth,
candidate_layer_id)`, never by list position alone. Its scientific metadata
includes dtype, shape, and ordered dimension semantics. Reusable validation
derives the lattice from sorted, unique configured layers and selected
prediction depths; it does not hard-code four layers.

For configured candidate-layer set `L` and prediction-depth set `D`, the exact
required lattice is:

`{(d, l) | d in D, l in L, l <= d}`.

The actual identity set must equal this set exactly. Duplicate identities,
missing identities, identities with `l > d`, unconfigured layers, and
unconfigured depths all fail closed. For the primary configuration the exact
cardinality is nine.

Required cumulative prediction maps are constructed by calling
`sum_preserving_fusion`:

- depth 12 from `A_{6|12}` and `A_{12|12}`;
- depth 18 from `A_{6|18}`, `A_{12|18}`, and `A_{18|18}`;
- depth 24 from all four depth-24 maps.

The depth-24 cumulative map is also `full_depth_map`. The final image score is
obtained by passing that sum-preserving map to `compute_exit_signals`.

## Descriptor dependency contract

The contract fields include:

- `descriptor_contract_version`;
- the authoritative ordered
  `rad.models.descriptors.LAYER_DESCRIPTOR_FEATURE_NAMES`;
- `descriptor_implementation_sha256`, calculated from the exact tracked bytes
  of `rad/models/descriptors.py`, so implementation changes invalidate the
  cache contract even when the public class name is unchanged;
- an extractor-configuration hash covering extractor class identity,
  `top_k_ratio`, authoritative feature order, and contract version;
- `descriptor_source_tensor_kind = "causal_anomaly_maps"`;
- explicit source identities `(checkpoint_depth, candidate_layer_id)`.

Loading fails if any expected descriptor contract field differs.

Feature dependency table:

| Feature(s) | Required cached values | Reconstruction |
| --- | --- | --- |
| `margin_mean`, `margin_std`, `margin_max`, `margin_topk`, `background_contrast` | Each depth-specific causal anomaly map | Existing extractor's per-map statistics |
| `response_topk_mean`, `response_max`, `sparsity` | Each depth-specific causal anomaly map | Existing extractor's absolute response and threshold operations |
| `top_entropy`, `global_entropy` | Each depth-specific causal anomaly map | Existing extractor's spatial distributions |
| `rank_spearman`, `topk_overlap`, `fused_map_change` | Causal maps at one depth plus valid-layer mask | Existing extractor reconstructs its internal equal-average reference |
| `response_comp`, `absolute_comp`, `boundary_comp` | Causal maps at one depth plus valid-layer mask | Existing extractor compares responses/maps/boundaries with its equal-average reference |
| `response_trend`, `entropy_trend` | Causal maps plus explicit ascending candidate-layer identities | Existing extractor computes adjacent-layer differences |

The table proves that the causal maps, explicit ordered layer identities,
valid-layer mask, dtype, shape, and dimension semantics are sufficient for
all 18 features. Raw patch tokens, normalized tokens, anomaly/normal tokens,
and similarity tensors are not descriptor inputs and are not cached for this
purpose. Descriptor reconstruction always calls the existing extractor; the
B2 module does not reproduce feature formulas.

The CPU fake-teacher parity test computes descriptors from live fake outputs,
atomically persists and reloads the scientific payload, reconstructs the
same inputs, and requires exact tensor equality. B2-02B must repeat this
parity check with real production outputs for at least one normal and one
anomalous sample at depths 12, 18, and 24.

## Canonical tensor and record hashing

Canonical tensor content is encoded in sorted logical-name order. Each tensor
contributes:

1. logical name;
2. dtype;
3. shape;
4. ordered dimension semantics;
5. explicit little-endian contiguous CPU bytes.

Tensors are detached, moved to CPU, made contiguous without changing the
authoritative dtype, and checked for finite values. Sparse, quantized,
unsupported layouts, NaN, and Inf fail closed.

The primary B2-02A cache contract requires `torch.float32` for causal anomaly
maps, cumulative maps, `full_depth_map`, the preprocessed binary mask, and the
tensor representation of `image_score`. Shapes and dimension semantics are
fixed by tensor role and validated before hashing. The generic canonical
encoder remains dtype-sensitive, but a primary cache record with a
non-float32 scientific tensor is rejected rather than accepted under a
different hash.

`record_hash_schema_version` versions an explicit scientific-field whitelist.
`record_scientific_sha256` covers exactly:

- record schema version and record-hash schema version;
- stable sample ID, split membership, category, label, anomaly type,
  dataset-relative image identity, and dataset-relative mask identity;
- candidate layers, prediction depths, and the exact causal-map lattice set;
- cache tensor contract version, explicit tensor logical identities, dtype,
  shape, dimension semantics, and canonical tensor content;
- descriptor contract version, authoritative feature order,
  `descriptor_source_tensor_kind`, descriptor implementation digest, and
  extractor-configuration digest;
- split scientific-hash version and V2 hash;
- accepted checkpoint SHA-256;
- execution-profile name and SHA-256.

No dictionary-wide “hash everything except” behavior is permitted. The
whitelist is constructed explicitly for the declared record-hash schema and
rejects missing or unknown scientific fields.

The record scientific hash excludes run ID, output path, timestamp,
`record_file_sha256`, runtime attestation, generation commit/branch,
worktree identity/status, machine environment, Python/Torch/CUDA/cuDNN/driver
identity, and GPU identity. These excluded values remain mandatory,
fail-closed outer partial/final-manifest provenance and are validated before
records can be produced or resumed.

Scientific sample identities are separated from access paths. `image_identity`
and `mask_identity` are canonical dataset-root-relative POSIX identities and
participate in the record hash. Absolute dataset paths, checkpoint paths,
run paths, temporary paths, and sample artifact paths are operational
provenance only; they never substitute for scientific identities and never
participate in `record_scientific_sha256`.

Under the approved Option A envelope, each `.pt` payload contains only the
scientific record and its `record_scientific_sha256`. After an exclusive,
atomic persistence succeeds, the CLI computes SHA-256 over the completed
`.pt` bytes and stores `record_file_sha256` in the partial manifest's
per-sample entry. The final manifest inherits only fully verified entries.
This avoids a self-referential file hash and avoids per-sample sidecars.

## Persistence, resume, and finalization

A new official run refuses an existing run directory. Each sample path is
claimed exclusively and cannot be silently overwritten. Tensor persistence
uses a run-local temporary file, flush/fsync, atomic replacement, completed
file-byte hashing, and immediate reload/scientific verification. Failed
writes cannot produce or preserve a passed final manifest.

Partial state is explicit and cannot claim `status=passed`. Resume is allowed
only with a valid partial manifest whose run-level scientific provenance and
exact plan match the requested run. For every reusable sample, resume:

1. recomputes the `.pt` file SHA-256 and compares it with the partial entry;
2. reloads the payload;
3. recomputes its canonical scientific hash;
4. compares both the embedded and partial-entry scientific hashes;
5. revalidates sample identity, tensor contract, descriptor contract, and
   provenance.

Any mismatch fails; immutable records are never silently replaced. After all
32 planned IDs are present exactly once, a complete coverage audit precedes
atomic publication of the final passed manifest.

Coverage includes an orphan/extra-file audit. The samples directory must
contain exactly the deterministic artifact filename set derived from the 32
planned stable IDs. Finalization and resume reject unknown `.pt` files,
unplanned stable IDs, duplicate identity-to-path mappings, sidecars,
temporary files, lock files, and any other orphan entry. Filesystem paths are
matched to planned identities but remain outside the record scientific hash.

## Dry-run and tests

Dry-run applies the execution profile and validates the split V2 identity,
checkpoint bytes, B2 ancestry, exact 32-sample plan, and output collision
intent. It builds intended manifest metadata in memory and emits the required
structured summary. It does not create a run directory, manifest, temporary
file, or lock; load the VisualAD model; or generate tensors.

Focused tests cover all specified input/provenance, cache-contract,
artifact, resume, coverage, hashing, and production-fixture rejection cases.
Additional binding tests prove:

- configured candidate-layer propagation and explicit map identities;
- exact live-fake-forward versus cache descriptor equality;
- descriptor contract drift rejection;
- cumulative maps call `sum_preserving_fusion`;
- descriptor reconstruction calls `LayerDescriptorExtractor`;
- image scoring calls `compute_exit_signals` with the sum-preserving
  cumulative prediction map rather than the extractor's equal-average
  reference.

All production changes follow observed RED tests. Final validation uses
Python 3.10.20, the complete CPU suite with `CUDA_VISIBLE_DEVICES=""`, Ruff,
mypy with explicit package bases, and the specified fail-closed CLI matrix.
No commit occurs before human review.
