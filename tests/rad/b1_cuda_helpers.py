"""Test-facing re-exports for Phase B1 CUDA helpers.

Executable qualification logic lives in ``rad.qualification.b1_cuda_equivalence``.
Keep fixtures and assertions in ``tests/rad/``; do not add new tool imports here.
"""

from __future__ import annotations

from rad.qualification.b1_cuda_equivalence import *  # noqa: F403
from rad.qualification.b1_cuda_equivalence import (  # noqa: F401
    B1_ACCEPTED_CHECKPOINT,
    B1_ACCEPTED_CHECKPOINT_SHA256,
    B1_ATOL,
    B1_RTOL,
    B1_SEED,
    DEFAULT_CANDIDATE_LAYERS,
    _ln_post_patch_tokens,
    apply_deterministic_cuda_settings,
    audit_task_level_category_provenance,
    build_official_full_depth_outputs,
    build_staged_full_depth_outputs,
    deterministic_synthetic,
    diagnose_four_path_divergence,
    install_block_counter,
    load_preprocessed_image,
    load_task_level_category_samples,
    load_teacher_production,
    reset_block_counter,
    run_deterministic_cuda_control,
    run_equivalence_protocol,
    run_operational_noise_envelope,
    run_same_chain_gate,
    tensor_diff,
    tensor_layout_info,
    validate_checkpoint,
)
