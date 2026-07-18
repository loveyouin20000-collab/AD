# Residual-Aware Adaptive-Depth VisualAD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use a task-by-task execution workflow with a fresh review gate after every task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible VisualAD extension that predicts input-dependent layer contribution, estimates residual localization gain, and performs genuine early exit at configurable ViT checkpoints while preserving zero-shot pixel localization.

**Architecture:** Preserve the official VisualAD anomaly-map path and introduce a thin staged-execution layer around the ViT backbone. At checkpoint `d`, construct only causally available maps `A_{l|d}` for `l <= d`, fuse them with a Dynamic Layer Contribution Module (DLCM), then use a Localization Sufficiency Estimator (LSE) and calibrated policy to either exit or continue. The main paper configuration uses `{6, 12, 18, 24}`; all modules accept an arbitrary sorted candidate-layer list.

**Tech Stack:** Python 3.10, PyTorch 2.0.0 baseline lock, torchvision 0.15.1, CUDA 11.8 default AutoDL profile, pytest, Ruff, mypy, PyYAML, NumPy, scikit-learn, scikit-image, TensorBoard or Weights & Biases, Git, Cursor Remote SSH.

## Global Constraints

- Target platform: AutoDL Linux GPU instance, developed from Cursor over SSH.
- Preserve official VisualAD behavior before introducing adaptive components.
- Main candidate layers: `[6, 12, 18, 24]`; arbitrary sorted checkpoints remain configurable.
- Source-domain data alone may determine normalization statistics, temperatures, exit thresholds, and hyperparameters.
- Default training is staged; optional joint fine-tuning is disabled until every staged gate passes.
- Batch-size-1 adaptive inference is the primary latency claim. Dynamic batch regrouping is a separate extension.
- Every production change follows red-green-refactor TDD and ends in a Git commit.
- No experiment may silently overwrite a checkpoint, split manifest, cache manifest, or result table.
- All stored tensors must include a schema version, sample identifier, checkpoint list, preprocessing hash, and teacher-checkpoint hash.

---

## Repository Map

```text
VisualAD/
├── VisualAD_lib/
│   └── VisualAD.py                 # minimally extended with staged execution APIs
├── rad/
│   ├── config.py                   # validated YAML dataclasses
│   ├── types.py                    # checkpoint state and output dataclasses
│   ├── data/
│   │   ├── split.py                # deterministic source train/calibration split
│   │   ├── cache_schema.py         # versioned cache records
│   │   └── cache_dataset.py        # sharded cache reader
│   ├── models/
│   │   ├── checkpoint_maps.py      # A_{l|d} generation
│   │   ├── descriptors.py          # normalized selector features
│   │   ├── dlcm.py                 # dynamic contribution and fusion
│   │   ├── lse.py                  # residual-gain estimator
│   │   └── policy.py               # conservative/balanced/aggressive rules
│   ├── losses/
│   │   ├── localization.py
│   │   ├── distillation.py
│   │   └── contribution.py
│   ├── targets/
│   │   ├── shapley.py
│   │   └── residual_gain.py
│   ├── calibration/
│   │   ├── temperature.py
│   │   └── policy_search.py
│   ├── trainers/
│   │   ├── fusion_trainer.py
│   │   ├── lse_trainer.py
│   │   └── joint_trainer.py
│   ├── inference/
│   │   └── adaptive_engine.py
│   └── evaluation/
│       ├── policy_metrics.py
│       ├── efficiency.py
│       └── export.py
├── tools/
│   ├── validate_environment.py
│   ├── reproduce_baseline.py
│   ├── build_source_split.py
│   ├── cache_teacher_outputs.py
│   ├── fit_descriptor_stats.py
│   ├── generate_shapley_targets.py
│   ├── train_fusion.py
│   ├── generate_gain_targets.py
│   ├── train_lse.py
│   ├── calibrate_policy.py
│   ├── evaluate_adaptive.py
│   ├── benchmark_latency.py
│   └── export_paper_tables.py
├── configs/rad/
│   ├── base.yaml
│   ├── fusion.yaml
│   ├── lse.yaml
│   ├── policy.yaml
│   └── experiments.yaml
├── tests/rad/
├── scripts/setup_autodl_env.sh
├── .cursor/rules/rad-visualad.mdc
├── requirements-dev.txt
└── pyproject.toml
```

---

### Task 1: Create the AutoDL environment and repository guardrails

**Files:**
- Create: `scripts/setup_autodl_env.sh`
- Create: `requirements-dev.txt`
- Create: `pyproject.toml`
- Create: `.cursor/rules/rad-visualad.mdc`
- Create: `tools/validate_environment.py`
- Test: `tests/rad/test_environment.py`

**Interfaces:**
- Produces: a reproducible environment validator callable as `python tools/validate_environment.py`.

- [x] **Step 1: Clone the official repository and create a working branch**

```bash
git clone https://github.com/7HHHHH/VisualAD.git
cd VisualAD
git checkout -b feat/residual-aware-adaptive-depth
mkdir -p scripts rad tests/rad configs/rad tools .cursor/rules
```

- [x] **Step 2: Write the environment test first**

```python
# tests/rad/test_environment.py
from tools.validate_environment import collect_environment


def test_environment_has_cuda_and_expected_torch_major_minor():
    env = collect_environment()
    assert env["python"].startswith("3.10")
    assert env["torch"].startswith("2.0")
    assert env["cuda_available"] is True
    assert env["gpu_count"] >= 1
```

- [x] **Step 3: Run the test and confirm red state**

```bash
pytest tests/rad/test_environment.py -q
```

Expected: collection error because `tools.validate_environment` does not exist.

- [x] **Step 4: Implement the validator**

```python
# tools/validate_environment.py
from __future__ import annotations

import json
import platform
from typing import Any

import torch


def collect_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


if __name__ == "__main__":
    print(json.dumps(collect_environment(), indent=2))
```

- [x] **Step 5: Add the setup script**

```bash
#!/usr/bin/env bash
# scripts/setup_autodl_env.sh
set -euo pipefail

ENV_NAME="rad-visualad"
conda create -y -n "${ENV_NAME}" python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/visualad-no-torch.txt
python -m pip install -r /tmp/visualad-no-torch.txt
python -m pip install -r requirements-dev.txt
python tools/validate_environment.py
```

```text
# requirements-dev.txt
pytest==8.2.2
pytest-cov==5.0.0
ruff==0.5.5
mypy==1.10.1
tensorboard==2.17.0
pandas==2.2.2
pyarrow==17.0.0
```

- [x] **Step 6: Add quality configuration**

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.mypy]
python_version = "3.10"
warn_unused_ignores = true
disallow_untyped_defs = true
ignore_missing_imports = true
```

- [x] **Step 7: Add Cursor project rules**

```markdown
---
description: RAD-VisualAD repository rules
globs: ["**/*.py", "**/*.yaml", "**/*.sh"]
alwaysApply: true
---
- Do not change official VisualAD behavior without an equivalence test.
- Write a failing pytest before implementation.
- Keep candidate layers configuration-driven; never hard-code four layers in reusable modules.
- Log tensor shapes, config hash, Git SHA, seed, split manifest hash, and checkpoint hash.
- Never tune on target-domain data.
- Every CLI supports `--config`, `--seed`, `--output-dir`, and `--dry-run` where meaningful.
```

- [x] **Step 8: Verify green state and commit**

```bash
bash scripts/setup_autodl_env.sh
pytest tests/rad/test_environment.py -q
ruff check tools tests/rad
mypy tools/validate_environment.py
git add scripts requirements-dev.txt pyproject.toml .cursor tools tests/rad
git commit -m "build: initialize AutoDL research environment"
```

Expected: one test passes; validator prints at least one CUDA GPU.

---

### Task 2: Add validated configuration and experiment manifests

**Files:**
- Create: `rad/config.py`
- Create: `configs/rad/base.yaml`
- Create: `tests/rad/test_config.py`

**Interfaces:**
- Produces: `ExperimentConfig.from_yaml(path: str) -> ExperimentConfig`.

- [x] **Step 1: Write failing validation tests**

```python
# tests/rad/test_config.py
import pytest
from rad.config import BackboneConfig, ExperimentConfig


def test_candidate_layers_must_be_sorted_unique_and_end_at_backbone_depth():
    with pytest.raises(ValueError):
        BackboneConfig(depth=24, candidate_layers=(6, 12, 12, 24))
    with pytest.raises(ValueError):
        BackboneConfig(depth=24, candidate_layers=(12, 6, 24))


def test_main_config_uses_visualad_checkpoints():
    cfg = ExperimentConfig.from_yaml("configs/rad/base.yaml")
    assert cfg.backbone.candidate_layers == (6, 12, 18, 24)
    assert cfg.zero_shot.target_tuning is False
```

- [x] **Step 2: Run and observe failure**

```bash
pytest tests/rad/test_config.py -q
```

Expected: import error for `rad.config`.

- [x] **Step 3: Implement strict dataclasses**

```python
# rad/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BackboneConfig:
    depth: int
    candidate_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.candidate_layers))) != self.candidate_layers:
            raise ValueError("candidate_layers must be strictly increasing and unique")
        if not self.candidate_layers or self.candidate_layers[-1] != self.depth:
            raise ValueError("the final candidate layer must equal backbone depth")


@dataclass(frozen=True)
class ZeroShotConfig:
    source_dataset: str
    target_datasets: tuple[str, ...]
    target_tuning: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    backbone: BackboneConfig
    zero_shot: ZeroShotConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        backbone = BackboneConfig(
            depth=int(raw["backbone"]["depth"]),
            candidate_layers=tuple(raw["backbone"]["candidate_layers"]),
        )
        zero_shot = ZeroShotConfig(
            source_dataset=str(raw["zero_shot"]["source_dataset"]),
            target_datasets=tuple(raw["zero_shot"]["target_datasets"]),
            target_tuning=bool(raw["zero_shot"].get("target_tuning", False)),
        )
        if zero_shot.target_tuning:
            raise ValueError("target-domain tuning is forbidden")
        return cls(seed=int(raw["seed"]), backbone=backbone, zero_shot=zero_shot)
```

```yaml
# configs/rad/base.yaml
seed: 111
backbone:
  depth: 24
  candidate_layers: [6, 12, 18, 24]
zero_shot:
  source_dataset: mvtec
  target_datasets: [visa]
  target_tuning: false
```

- [x] **Step 4: Verify and commit**

```bash
pytest tests/rad/test_config.py -q
ruff check rad tests/rad
git add rad/config.py configs/rad/base.yaml tests/rad/test_config.py
git commit -m "feat: add validated experiment configuration"
```

---

### Task 3: Reproduce and freeze the official VisualAD baseline

**Files:**
- Create: `tools/reproduce_baseline.py`
- Create: `configs/rad/baseline_mvtec_to_visa.yaml`
- Create: `configs/rad/baseline_visa_to_mvtec.yaml`
- Create: `tests/rad/test_baseline_smoke.py`
- Create at runtime: `artifacts/baseline/<run_id>/manifest.json`

**Interfaces:**
- Produces: immutable teacher checkpoint and baseline metric JSON.

- [x] **Step 1: Write a command smoke test**

```python
# tests/rad/test_baseline_smoke.py
import subprocess


def test_baseline_dry_run_resolves_paths_and_command():
    result = subprocess.run(
        ["python", "tools/reproduce_baseline.py", "--config", "configs/rad/baseline_mvtec_to_visa.yaml", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "train.py" in result.stdout
    assert "features_list 6 12 18 24" in result.stdout
```

- [x] **Step 2: Implement a wrapper that never edits official scripts**

The wrapper must validate dataset paths, serialize the exact command, Git SHA, package versions, and a SHA-256 hash of every config file before launching `train.py` and `test.py`. It must fail if an output directory already contains a completed manifest.

- [x] **Step 3: Run the official cross-dataset directions**

```bash
python tools/reproduce_baseline.py --config configs/rad/baseline_mvtec_to_visa.yaml
python tools/reproduce_baseline.py --config configs/rad/baseline_visa_to_mvtec.yaml
```

- [x] **Step 4: Acceptance gate**

Record image AUROC, pixel AUROC, pixel AP, and PRO. Do not begin adaptive changes until repeated inference with the same checkpoint and seed is deterministic within `1e-6` for logits and the reproduced metrics are within a predeclared tolerance of the official result or the discrepancy is documented with evidence.

- [x] **Step 5: Tag the baseline commit**

```bash
git add tools configs/rad tests/rad
git commit -m "test: reproduce official VisualAD baseline"
git tag baseline-visualad-verified
```

---

### Task 4: Define staged-execution tensor contracts

**Files:**
- Create: `rad/types.py`
- Test: `tests/rad/test_types.py`

**Interfaces:**
- Produces:
  - `StageCache(sequence, next_block, patch_tokens, checkpoint_tokens)`
  - `CheckpointOutput(depth, patch_tokens, anomaly_token, normal_token, class_token)`

- [x] **Step 1: Write shape and invariant tests**

```python
# tests/rad/test_types.py
import pytest
import torch
from rad.types import StageCache


def test_stage_cache_rejects_invalid_next_block():
    with pytest.raises(ValueError):
        StageCache(sequence=torch.zeros(10, 2, 8), next_block=0, patch_tokens={})


def test_stage_cache_detach_removes_graph():
    cache = StageCache(sequence=torch.randn(10, 2, 8, requires_grad=True), next_block=7, patch_tokens={6: torch.randn(2, 7, 8, requires_grad=True)})
    detached = cache.detach()
    assert not detached.sequence.requires_grad
    assert not detached.patch_tokens[6].requires_grad
```

- [x] **Step 2: Implement immutable dataclasses with explicit sequence convention `[tokens, batch, width]` and patch convention `[batch, patches, width]`**

- [x] **Step 3: Run and commit**

```bash
pytest tests/rad/test_types.py -q
git add rad/types.py tests/rad/test_types.py
git commit -m "feat: define staged backbone tensor contracts"
```

---

### Task 5: Refactor the ViT for genuine resumable execution

**Files:**
- Modify: `VisualAD_lib/VisualAD.py`
- Create: `tests/rad/test_staged_backbone.py`

**Interfaces:**
- Produces:
  - `VisionTransformer.prepare_stage(image) -> StageCache`
  - `VisionTransformer.run_to(cache, target_layer) -> tuple[CheckpointOutput, StageCache]`
  - `VisionTransformer.forward_staged(image, candidate_layers) -> dict[int, CheckpointOutput]`
- The legacy `forward` and `encode_image` outputs remain unchanged.

- [x] **Step 1: Write numerical equivalence tests before touching production code**

```python
# tests/rad/test_staged_backbone.py
import torch


def test_staged_outputs_match_legacy_outputs(visualad_model, sample_image):
    visualad_model.eval()
    with torch.no_grad():
        legacy = visualad_model.encode_image(sample_image, [6, 12, 18, 24])
        staged = visualad_model.visual.forward_staged(sample_image, [6, 12, 18, 24])
    legacy_patches = legacy[5]
    for idx, depth in enumerate([6, 12, 18, 24]):
        assert torch.allclose(staged[depth].patch_tokens, legacy_patches[idx][:, 3:, :], atol=1e-6, rtol=1e-5)


def test_exit_at_12_executes_exactly_12_blocks(visualad_model, sample_image, block_call_counter):
    cache = visualad_model.visual.prepare_stage(sample_image)
    visualad_model.visual.run_to(cache, 12)
    assert block_call_counter.total == 12
```

- [x] **Step 2: Run the tests and confirm failure**

```bash
pytest tests/rad/test_staged_backbone.py -q
```

Expected: missing staged methods.

- [x] **Step 3: Add minimal staged methods without rewriting VisualAD internals**

Implementation rules:
1. Reuse the official convolution, class embedding, positional embedding, token insertion, pre-layer norm, residual blocks, and post-layer norm.
2. Cache the sequence immediately after the requested residual block.
3. Store candidate patch tokens before final projection, using the same post-layer norm and projection path as legacy inference.
4. Never run a block whose one-based index exceeds `target_layer`.
5. Keep legacy `forward` as a compatibility wrapper; do not redirect it until equivalence tests pass.

- [x] **Step 4: Test continuation as well as direct execution**

```python
def test_continue_12_to_18_matches_direct_18(visualad_model, sample_image):
    cache = visualad_model.visual.prepare_stage(sample_image)
    out12, cache12 = visualad_model.visual.run_to(cache, 12)
    out18, _ = visualad_model.visual.run_to(cache12, 18)
    direct = visualad_model.visual.forward_staged(sample_image, [18])[18]
    assert torch.allclose(out18.patch_tokens, direct.patch_tokens, atol=1e-6, rtol=1e-5)
```

- [x] **Step 5: Verify legacy regression suite and commit**

```bash
pytest tests/rad/test_staged_backbone.py tests/rad/test_baseline_smoke.py -q
git add VisualAD_lib/VisualAD.py tests/rad/test_staged_backbone.py
git commit -m "feat: add numerically equivalent staged ViT execution"
```

---

### Task 6: Build checkpoint-conditioned causal anomaly maps

**Files:**
- Create: `rad/models/checkpoint_maps.py`
- Test: `tests/rad/test_checkpoint_maps.py`

**Interfaces:**
- Consumes: cached patch tokens `P_l` and current checkpoint tokens `t_a^d`, `t_n^d`.
- Produces: `dict[int, Tensor]` of maps `A_{l|d}` with shape `[B, 1, H, W]` for all `l <= d`.

- [x] **Step 1: Write causal availability test**

```python
# tests/rad/test_checkpoint_maps.py

def test_checkpoint_12_cannot_use_deeper_patch_tokens(generator, staged_outputs):
    maps = generator.build(depth=12, outputs=staged_outputs)
    assert set(maps) == {6, 12}
```

- [x] **Step 2: Write full-depth compatibility test**

```python
def test_checkpoint_24_matches_official_map_list(generator, official_map_list, staged_outputs):
    maps = generator.build(depth=24, outputs=staged_outputs)
    for depth, expected in zip([6, 12, 18, 24], official_map_list):
        assert torch.allclose(maps[depth], expected, atol=1e-5, rtol=1e-4)
```

- [x] **Step 3: Implement causal token conditioning**

At checkpoint `d`, use `t_a^d` and `t_n^d` for every cached patch set `P_l, l <= d`. This creates `A_{6|12}` and `A_{12|12}` at layer 12, then recomputes `A_{6|18}`, `A_{12|18}`, and `A_{18|18}` when layer 18 becomes available. At layer 24 the result must match the official full-depth path.

- [x] **Step 4: Verify and commit**

```bash
pytest tests/rad/test_checkpoint_maps.py -q
git add rad/models/checkpoint_maps.py tests/rad/test_checkpoint_maps.py
git commit -m "feat: add causal checkpoint-conditioned anomaly maps"
```

---

### Task 7: Create a deterministic source train/calibration split

**Files:**
- Create: `rad/data/split.py`
- Create: `tools/build_source_split.py`
- Test: `tests/rad/test_source_split.py`

**Interfaces:**
- Produces immutable JSONL manifests with `sample_id`, image path, mask path, category, label, and split.

- [x] Write tests proving no sample overlap, deterministic stratification by category and label, and identical output for the same seed.
- [x] Implement `build_source_split(samples, calibration_fraction=0.2, seed=111)`.
- [x] Refuse to regenerate an existing manifest unless `--force` is supplied.
- [x] Run:

```bash
python tools/build_source_split.py --dataset mvtec --root /root/autodl-tmp/data/mvtec --seed 111 --output artifacts/splits/mvtec_seed111.jsonl
pytest tests/rad/test_source_split.py -q
git commit -am "feat: add leakage-safe source calibration split"
```

Acceptance: all selector normalization and policy calibration loaders consume only rows labeled `calibration`; target datasets have no calibration manifest.

---

### Task 8: Implement versioned teacher-output caching

**Files:**
- Create: `rad/data/cache_schema.py`
- Create: `rad/data/cache_dataset.py`
- Create: `tools/cache_teacher_outputs.py`
- Test: `tests/rad/test_cache_schema.py`

**Interfaces:**
- A shard record stores sample ID, image label, mask reference, maps at every checkpoint, raw descriptor ingredients, teacher logits, preprocessing hash, split hash, checkpoint hash, and schema version.

- [x] Test rejection of stale hashes and incomplete checkpoint lists.
- [x] Store sharded `.pt` files plus a Parquet index; do not create one file per small tensor.
- [x] Add `--resume` that verifies every existing shard before skipping it.
- [x] Run a 16-sample smoke cache before full caching.

```bash
python tools/cache_teacher_outputs.py --config configs/rad/base.yaml --split train --limit 16 --output artifacts/cache/smoke
pytest tests/rad/test_cache_schema.py -q
git commit -am "feat: add versioned teacher-output cache"
```

---

### Task 9: Implement normalized selector descriptors

**Files:**
- Create: `rad/models/descriptors.py`
- Create: `tools/fit_descriptor_stats.py`
- Test: `tests/rad/test_descriptors.py`

**Interfaces:**
- `LayerDescriptorExtractor.forward(...) -> Tensor[B, L, 18]`
- `CheckpointContextExtractor.forward(...) -> Tensor[B, 8]`
- `DescriptorNormalizer.fit()` may read source-train cache only; frozen statistics are used everywhere else.

**18 layer features:** margin mean/std/max/top-k/background contrast; response top-k mean/max/sparsity; top/global entropy; rank Spearman/top-k overlap/fused-map change; response/absolute/boundary complementarity; response and entropy trends.

**8 checkpoint context features:** current depth ratio, map entropy, boundary entropy, image normal confidence, image anomaly confidence, weight entropy, number of available layers, previous fused-map change.

- [x] Test finite outputs on all-zero normal maps and constant maps.
- [x] Test invariance to batch ordering.
- [x] Fit median/IQR statistics on source train only and clamp normalized values to `[-8, 8]`.
- [x] Commit after `pytest tests/rad/test_descriptors.py -q` passes.

---

### Task 10: Implement the Dynamic Layer Contribution Module

**Files:**
- Create: `rad/models/dlcm.py`
- Create: `tests/rad/test_dlcm.py`

**Interfaces:**
- `DLCM.forward(layer_desc, checkpoint_context, layer_ids, valid_mask) -> Tensor[B, L]`
- `sum_preserving_fusion(maps, weights, valid_mask) -> Tensor[B,1,H,W]`

- [ ] Write tests: invalid layers receive exactly zero weight; valid weights sum to one; zero-initialized scorer reproduces equal fusion; gradients reach descriptors and scorer.
- [ ] Implement shared MLP `18 -> 64 -> 32`, learned 16-dimensional layer embedding, 8-dimensional context projection, and final zero-initialized scalar scorer.
- [ ] Apply masked softmax and an optional floor `w'=(1-alpha)w + alpha/n_valid` with linear decay of `alpha` from `0.1` to `0` during the first 20% of fusion training.
- [ ] Define fusion as:

```python
def sum_preserving_fusion(maps, weights, valid_mask):
    n_valid = valid_mask.sum(dim=1).clamp_min(1).to(maps.dtype)
    return (maps * weights[:, :, None, None, None]).sum(dim=1) * n_valid[:, None, None, None]
```

- [ ] Verify equal-fusion compatibility and commit.

---

### Task 11: Implement localization and distillation losses

**Files:**
- Create: `rad/losses/localization.py`
- Create: `rad/losses/distillation.py`
- Test: `tests/rad/test_losses.py`

**Interfaces:**
- `sample_localization_error(logits, mask, image_label) -> Tensor[B]`
- `confidence_weighted_distillation(student, teacher) -> Tensor`

Use the initial differentiable sample error:

```text
E = 1.0 * BCEWithLogits
  + 1.0 * SoftDice
  + 0.2 * BoundaryL1
  + 0.2 * BCE(TopKMeanProbability, image_label)
```

- [ ] Test finite and differentiable error for normal masks, one-pixel anomalies, and full masks.
- [ ] Use Sobel gradients for boundary loss; do not use non-differentiable AP or PRO as training utility.
- [ ] Weight teacher pixels by `1 - normalized_binary_entropy(sigmoid(teacher_logit))`.
- [ ] Commit after unit tests pass.

---

### Task 12: Generate exact checkpoint-specific Shapley targets

**Files:**
- Create: `rad/targets/shapley.py`
- Create: `tools/generate_shapley_targets.py`
- Test: `tests/rad/test_shapley.py`

**Interfaces:**
- `exact_shapley(layer_maps, utility_fn) -> Tensor[L]`
- Generates targets separately at d=12, d=18, and d=24.

- [ ] Test Shapley efficiency: contributions sum to `U(all)-U(empty)`.
- [ ] Test a synthetic perfect layer receives the largest contribution.
- [ ] Enumerate 4, 8, and 16 subsets for checkpoints 12, 18, and 24 respectively.
- [ ] Transform signed contributions into a supervision distribution with `softmax(phi / tau_phi)`; persist raw signed values as well.
- [ ] Generate targets offline from frozen causal maps and commit.

---

### Task 13: Train DLCM and intermediate localization maps without exiting

**Files:**
- Create: `rad/trainers/fusion_trainer.py`
- Create: `tools/train_fusion.py`
- Create: `configs/rad/fusion.yaml`
- Test: `tests/rad/test_fusion_training_step.py`

**Interfaces:**
- Training always reaches layer 24 and returns losses at d=12,18,24.

Default objective:

```text
L_fusion = Σ_d λ_d L_loc(F_d,Y)
         + 0.5 L_map_KD
         + 0.2 L_boundary_KD
         + 0.5 L_contribution
```

Initial `λ_12=0.5`, `λ_18=0.75`, `λ_24=1.0`.

- [ ] Test one optimizer step decreases loss on a fixed micro-batch.
- [ ] Freeze the ViT, SCA, and feature transforms by default.
- [ ] Log per-checkpoint localization error, contribution KL, weight entropy, and average weight.
- [ ] Train three seeds and save the best source-calibration checkpoint by pixel AP subject to no regression at layer 24.

---

### Task 14: Generate residual-gain and sufficiency targets

**Files:**
- Create: `rad/targets/residual_gain.py`
- Create: `tools/generate_gain_targets.py`
- Test: `tests/rad/test_residual_gain.py`

**Interfaces:**
- `g18 = relu(E18-E24)`
- `g12 = relu(E12-min(E18,E24))`
- `sufficient = (gain <= epsilon_gain) AND (E_d <= epsilon_absolute)`

- [ ] Test that a poor but equally poor full-depth map is not marked sufficient.
- [ ] Store raw errors and gains so thresholds can be recalibrated without regeneration.
- [ ] Use frozen DLCM and stop gradients through all target computations.
- [ ] Commit after tests pass.

---

### Task 15: Implement the Localization Sufficiency Estimator

**Files:**
- Create: `rad/models/lse.py`
- Create: `tests/rad/test_lse.py`

**Interfaces:**
- `LSE.forward(state, depth_id) -> GainPrediction(mean, log_variance, sufficiency_logit)`.

- [ ] Test mean is nonnegative, log variance is in `[-8,4]`, and batch shape is stable.
- [ ] Implement a shared `input -> 64 -> 32` MLP plus learned depth embedding for checkpoints 12 and 18.
- [ ] Use `softplus` for mean and clamp log variance.
- [ ] Implement heteroscedastic NLL plus `0.5 * BCEWithLogits` sufficiency loss.
- [ ] Commit after tests pass.

---

### Task 16: Train and validate LSE in isolation

**Files:**
- Create: `rad/trainers/lse_trainer.py`
- Create: `tools/train_lse.py`
- Create: `configs/rad/lse.yaml`
- Test: `tests/rad/test_lse_training_step.py`

- [ ] Split only the source training partition for model fitting; use the fixed source calibration partition for model selection and calibration reports.
- [ ] Report gain MAE/RMSE, beneficial-depth AUROC, Brier score, ECE, and NLL independently at d=12 and d=18.
- [ ] Add early stopping on calibration NLL with patience 10.
- [ ] Save prediction tables, not only aggregate metrics.
- [ ] Commit after a deterministic two-epoch smoke run passes.

---

### Task 17: Calibrate map probabilities and exit profiles on source data only

**Files:**
- Create: `rad/calibration/temperature.py`
- Create: `rad/calibration/policy_search.py`
- Create: `rad/models/policy.py`
- Create: `tools/calibrate_policy.py`
- Test: `tests/rad/test_policy.py`

**Interfaces:**
- `PolicyProfile(name, gain_threshold, kappa, map_uncertainty_threshold, image_confidence_margin, stability_threshold)`.
- `should_exit(prediction, signals, profile) -> bool`.

Profiles:
1. Conservative: gain UCB + map uncertainty + symmetric image confidence + stability.
2. Balanced: gain UCB + map uncertainty + symmetric image confidence.
3. Aggressive: gain UCB only.

- [ ] Test ambiguous image predictions continue, while confidently normal and confidently anomalous samples may exit.
- [ ] Search thresholds under explicit constraints on pixel-AP drop and false-safe-exit rate.
- [ ] Save all feasible Pareto points and the selected three profiles.
- [ ] Refuse any target-dataset path in calibration CLI.
- [ ] Commit after policy tests pass.

---

### Task 18: Implement true batch-size-1 adaptive inference

**Files:**
- Create: `rad/inference/adaptive_engine.py`
- Create: `tools/evaluate_adaptive.py`
- Test: `tests/rad/test_adaptive_engine.py`

**Interfaces:**
- Returns final map, image score, selected depth, checkpoint trace, weights, gain predictions, and timing breakdown.

- [ ] Test an exit at 12 executes exactly 12 blocks and never computes layer-18/24 maps.
- [ ] Test a continue decision reuses the cached sequence rather than restarting from block 1.
- [ ] Test forced-full-depth output matches dynamic-fusion-only output.
- [ ] Warm up 50 iterations before timing; synchronize CUDA before and after each measured segment.
- [ ] Commit after behavioral and call-count tests pass.

---

### Task 19: Add policy, contribution, and efficiency metrics

**Files:**
- Create: `rad/evaluation/policy_metrics.py`
- Create: `rad/evaluation/efficiency.py`
- Test: `tests/rad/test_metrics.py`

Implement:
- gain MAE/RMSE;
- beneficial-depth AUROC;
- Brier and ECE;
- false-safe-exit rate `P(g_D > epsilon | D < 24)`;
- false-continue rate;
- risk-coverage points;
- Spearman/Pearson contribution correlation;
- top-contributing-layer accuracy;
- expected depth and exit histogram;
- measured latency, throughput, peak memory, and selector overhead.

- [ ] Test every metric on a hand-calculated fixture.
- [ ] Export per-sample traces before aggregate summaries.
- [ ] Commit after tests pass.

---

### Task 20: Build a defensible latency benchmark

**Files:**
- Create: `tools/benchmark_latency.py`
- Create: `configs/rad/benchmark.yaml`
- Test: `tests/rad/test_benchmark_smoke.py`

- [ ] Measure batch 1 as the primary adaptive result using CUDA events, 50 warmups, 200 repetitions, median, mean, p90, and p95.
- [ ] Report backbone stage, checkpoint-map generation, descriptors, DLCM, LSE, post-processing, and total latency separately.
- [ ] Record GPU model, driver, CUDA, clocks where available, image resolution, dtype, and memory.
- [ ] Compare equal full depth, dynamic fusion full depth, fixed 12/18, and three adaptive profiles.
- [ ] Treat batched dynamic regrouping as a separate experiment and do not mix it with the primary claim.

---

### Task 21: Encode the paper experiment matrix

**Files:**
- Create: `configs/rad/experiments.yaml`
- Create: `tools/run_experiment_matrix.py`
- Test: `tests/rad/test_experiment_matrix.py`

Required methods:
- original VisualAD;
- fixed exit 12 and 18;
- static learned fusion;
- dynamic fusion only;
- confidence-only exit;
- stability-only exit;
- confidence + stability;
- residual-gain exit with equal fusion;
- full method at conservative, balanced, aggressive profiles;
- random exit matched to exit distribution;
- oracle earliest exit.

Required ablations:
- DLCM with/without Shapley supervision;
- exact Shapley vs leave-one-out;
- selector cumulative and leave-one-out inputs;
- no KD / map KD / map+boundary KD;
- staged vs optional joint training.

- [ ] Make each matrix row a complete immutable config.
- [ ] Add a dry-run mode printing commands and estimated GPU-hours.
- [ ] Commit after schema tests pass.

---

### Task 22: Run zero-shot transfer without target tuning

**Files:**
- Create: `tools/evaluate_zero_shot_transfer.py`
- Create: `rad/evaluation/export.py`
- Test: `tests/rad/test_no_target_tuning.py`

- [ ] Load one source-calibrated policy unchanged on every target dataset.
- [ ] Report per-dataset depth distribution, pixel AP drop, PRO drop, boundary F-score drop, and false-safe-exit rate.
- [ ] Stratify by normal/anomalous, anomaly area, contrast proxy, and boundary complexity.
- [ ] Add a test that monkeypatches calibration and fails if target samples are accessed.
- [ ] Commit per-sample predictions and scripts, not raw datasets.

---

### Task 23: Add optional joint fine-tuning behind a feature flag

**Files:**
- Create: `rad/trainers/joint_trainer.py`
- Create: `configs/rad/joint.yaml`
- Create: `tools/train_joint.py`
- Test: `tests/rad/test_joint_training.py`

- [ ] Refuse to launch unless valid staged DLCM and LSE checkpoints are supplied.
- [ ] Keep the VisualAD backbone frozen by default.
- [ ] Start with learning rate `1e-5`; ramp compute penalty from 0 during the first 20% of epochs.
- [ ] Preserve a no-regression gate on full-depth localization.
- [ ] Treat this as an ablation, not the primary pipeline.

---

### Task 24: Add CI, traceability, and paper-table export

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tools/export_paper_tables.py`
- Create: `docs/traceability.md`
- Test: `tests/rad/test_export_schema.py`

- [ ] CI runs CPU unit tests, Ruff, mypy, and config dry runs; GPU numerical-equivalence tests remain a documented AutoDL gate.
- [ ] Export CSV and LaTeX for main comparison, fusion ablation, exit-strategy comparison, selector ablation, risk-coverage, oracle gap, and zero-shot transfer.
- [ ] Map every hypothesis to requirement, module, test, experiment, and paper artifact.
- [ ] Produce a release manifest containing Git SHA, configs, hashes, environment, checkpoints, and result files.
- [ ] Tag the reproducible paper artifact:

```bash
git tag cvpr-rad-visualad-v1
```

---

## Sprint Plan

| Sprint | Length | Increment | Exit criterion |
|---|---:|---|---|
| 0 | 1 week | AutoDL setup and official baseline | Reproducible full-depth metrics and immutable teacher |
| 1 | 1 week | Staged ViT and causal maps | Numerical equivalence and exact block-count tests |
| 2 | 1 week | Cache, descriptors, DLCM | Full-depth dynamic fusion improves or matches baseline |
| 3 | 1 week | Shapley and intermediate distillation | Layer-12/18 gap characterized and reduced |
| 4 | 1 week | Residual targets and LSE | Calibrated gain prediction metrics pass gate |
| 5 | 1 week | Exit policy and adaptive engine | True early exits with bounded false-safe rate |
| 6 | 1-2 weeks | Main experiments and profiling | Three Pareto operating points and zero-shot tables |
| 7 | 1 week | Optional joint training and paper export | Ablation complete; release manifest generated |

## Definition of Ready

A backlog item is ready only when its scientific hypothesis, input/output contract, test fixture, dataset split, acceptance metric, and estimated GPU cost are explicit.

## Definition of Done

A task is done only when:
1. the failing test was observed before implementation;
2. unit and relevant regression tests pass;
3. the CLI dry run and one small GPU smoke run pass;
4. code is linted and typed;
5. artifacts contain hashes and provenance;
6. results are reviewed against the task acceptance criterion;
7. the change is committed with one focused message;
8. documentation and traceability are updated.

## Cursor Task Prompt Template

Use this prompt for one task at a time:

```text
Implement Task <N> from RAD_VisualAD_Cursor_Implementation_Plan.md.
Read only the files listed under that task plus directly imported dependencies.
First add the specified failing test and run it. Do not write production code until the failure is shown.
Implement the smallest change that passes the test while preserving all legacy VisualAD behavior.
Run the exact task tests, then the affected regression tests, Ruff, and mypy.
Show the diff, commands, and outputs. Do not commit until I approve the review gate.
```

## Ordered Experiment Commands

```bash
# 1. Baseline
python tools/reproduce_baseline.py --config configs/rad/baseline_mvtec_to_visa.yaml

# 2. Source split and cache
python tools/build_source_split.py --dataset mvtec --root /root/autodl-tmp/data/mvtec --seed 111 --output artifacts/splits/mvtec_seed111.jsonl
python tools/cache_teacher_outputs.py --config configs/rad/base.yaml --split train --output artifacts/cache/mvtec_teacher
python tools/fit_descriptor_stats.py --cache artifacts/cache/mvtec_teacher --output artifacts/stats/mvtec_seed111.json

# 3. Contribution targets and fusion
python tools/generate_shapley_targets.py --config configs/rad/fusion.yaml
python tools/train_fusion.py --config configs/rad/fusion.yaml --seed 111

# 4. Gain targets and LSE
python tools/generate_gain_targets.py --config configs/rad/lse.yaml
python tools/train_lse.py --config configs/rad/lse.yaml --seed 111

# 5. Calibration
python tools/calibrate_policy.py --config configs/rad/policy.yaml --split calibration

# 6. Zero-shot evaluation and latency
python tools/evaluate_adaptive.py --config configs/rad/experiments.yaml --dataset visa
python tools/benchmark_latency.py --config configs/rad/benchmark.yaml

# 7. Paper export
python tools/export_paper_tables.py --results artifacts/results --output artifacts/paper
```

## Go/No-Go Research Gates

1. **Baseline gate:** no adaptive work until official behavior is reproduced and frozen.
2. **Staged-execution gate:** no efficiency claim until block-count and numerical-equivalence tests pass.
3. **Intermediate-map gate:** if layer-12 remains substantially below layer-24 after distillation, restrict layer-12 exits or reposition the claim around layer 18.
4. **Fusion gate:** DLCM must beat or match equal fusion at full depth and exhibit positive correlation with causal contribution targets.
5. **Selector gate:** residual-gain policy must reduce false-safe exits versus confidence-only and stability-only at matched expected depth.
6. **Latency gate:** measured batch-1 latency must improve; FLOPs or expected depth alone are insufficient.
7. **Zero-shot gate:** all thresholds remain source-calibrated and target-domain exit distributions are reported.
8. **Novelty gate:** if Shapley supervision and residual-gain prediction provide no measurable benefit, simplify the paper rather than retaining decorative modules.
