# B2-03B Descriptor Artifact Reproduction Report

## Verdict

**status:** `deterministic_descriptor_artifact_reproduction`  
**scientifically_equivalent:** `True`  
**final_status:** `passed`

## Base

| Field | Value |
|---|---|
| Worktree branch | `phase-b2-descriptor-real-extraction` |
| HEAD / contract tag commit | `c0a48df2305976c5954329d322bb14ddf24a5ef6` |
| Contract tag | `b2-descriptor-contract-v1` |
| Config | `configs/phase_b/b2_descriptor_artifacts_gate_c.json` |

## Identity distinctions

| Kind | Meaning |
|---|---|
| Scientific equality | Collection + per-record scientific hashes, raw descriptor tensors, normalization scientific content |
| Serialized file-byte equality | Diagnostic only (observed equal for descriptors, normalization, and final manifest in this dual run) |
| Source-cache identity | Teacher-cache `cache_scientific_sha256` + `sample_coverage_sha256` |
| Descriptor collection identity | Descriptor collection / coverage / normalization scientific hashes |

## Canonical source teacher cache

| Field | Value |
|---|---|
| Selection rule | First authoritative clean-worktree cache whose complete validation passes |
| Canonical run | `authoritative-run-a-20260723-155404` |
| Canonical manifest file SHA-256 | `3c20d062b34ddf8041a572a42eae0b72199a241719351016953028c8763535e3` |
| External control run | `authoritative-run-b-20260723-155404` |
| Control manifest file SHA-256 | `3c20d062b34ddf8041a572a42eae0b72199a241719351016953028c8763535e3` |
| cache_scientific_sha256 | `66d23807e868696a9c4a68ad83399d82df3d33e743a97d97eeb98ac60c0b1b0a` |
| sample_coverage_sha256 | `6e538b902795c377f9992258e307e58b5c0ba0f99cbbe6c3853a81947ca3d76c` |

Both descriptor Run A and Run B consumed **the same** canonical source cache.

## Dual-run materialization

| | Run A | Run B |
|---|---|---|
| Artifact run name | `authoritative-run-a-20260729-013956` | `authoritative-run-b-20260729-014404` |
| Elapsed seconds | `60.132` | `61.982` |
| Peak CPU memory (kB) | `594836` | `584236` |
| Peak GPU alloc/reserved | `0 / 0` | `0 / 0` |
| CUDA visibility | empty (`cuda_available=false`) | empty (`cuda_available=false`) |
| teacher_forward_count | `0` | `0` |
| final_manifest file SHA-256 | `8f0b5545a20b953011ef260c736f8515aa718c24330f0898748b355c38bd7326` | `8f0b5545a20b953011ef260c736f8515aa718c24330f0898748b355c38bd7326` |

Exact artifact set per run: **32** descriptor `.pt` + **1** `normalization_statistics.pt` + **1** `final_manifest.json` + **1** `final_manifest.json.sha256` (35 files).

## Four scientific collection identities

| Identity | Run A | Run B | Equal |
|---|---|---|---|
| descriptor_collection_scientific_sha256 | `eb967822725e730ee2eb8afa3a5c8e28b4657141aa920d6a688ab370c70c6dd9` | `eb967822725e730ee2eb8afa3a5c8e28b4657141aa920d6a688ab370c70c6dd9` | yes |
| descriptor_sample_coverage_sha256 | `27d064db21b5c699503be32e414d579bd1aa7158f1d9b141de26555fc79bc6df` | `27d064db21b5c699503be32e414d579bd1aa7158f1d9b141de26555fc79bc6df` | yes |
| normalization_statistics_scientific_sha256 | `f77975a94acf87a14b0753aabc9aad6777943ee4e4958b0a2083701cf4528594` | `f77975a94acf87a14b0753aabc9aad6777943ee4e4958b0a2083701cf4528594` | yes |
| normalization_training_coverage_sha256 | `e940f46bf696d326f8b982f15b8639f81e4548ec31a9b09634729811337e4c90` | `e940f46bf696d326f8b982f15b8639f81e4548ec31a9b09634729811337e4c90` | yes |

## Comparisons

- 32-record scientific-hash comparison: **all equal**
- Raw tensor digest comparison: **all equal**
- Normalization-statistics scientific content: **equal**
- File-byte diagnostics: descriptors **equal**, normalization **equal**, final manifest **equal** (diagnostic only)

## Semantics and normalization

- Depths 12/18/24 shapes `[1,2,18]` / `[1,3,18]` / `[1,4,18]`; dtype float32; feature order `LAYER_DESCRIPTOR_FEATURE_NAMES`
- Spot-checks (normal train, anomalous train, calibration, evaluation): reconstructed from teacher causal maps equals Run A and Run B tensors at all depths; **zero teacher forwards**
- Normalization membership: **16 / 0 / 0** (train/cal/eval); axis counts all 16; ddof=0; stats float64; outputs float32; zero-variance divisor 1.0 → normalized 0

## Source-only audit

- MVTec source only; **no VisA / target-domain** paths or records

## Negative controls

All **14** copied-temp negative controls produced **non-passed** collections (fail-closed).

## Dry-run

Production `--dry-run` passed with `artifact_written=false`, `teacher_forward_count=0`, planned 32 samples, normalization from 16 training only; no output directory / manifest / `.pt` / receipt / lock created; `load_teacher_bundle` not called; VisualAD backbone not loaded.

## Scope exclusions

No Shapley targets, DLCM/LSE training, residual-gain, or policy-calibration artifacts were generated. No production code was modified.

## Limitations

- No dedicated B2-03B concise-evidence writer module exists; this manifest was assembled from verify_descriptor_artifact_collection and compare_descriptor_artifact_collections outputs plus runtime attestation files.
- Teacher-cache directories retain a leftover partial_manifest.json from generation; production load_and_validate_accepted_teacher_cache_from_disk accepts the caches and samples/ orphan audit passes.
- Manifest-only split_membership cosmetic edits are not independently fail-closed by verify_descriptor_artifact_collection; membership is enforced via scientific records and collection hashes. Negative control 4 mutated the scientific record membership.
- GPU was not used; CUDA_VISIBLE_DEVICES was empty and cuda_available was false.
