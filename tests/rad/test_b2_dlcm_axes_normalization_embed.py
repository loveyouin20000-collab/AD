"""RED→GREEN: production B2-03B axes normalization embeds into deployment."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rad.phase_b import b2_descriptor_artifacts as desc
from rad.phase_b import b2_dlcm as model_mod
from rad.phase_b import b2_dlcm_deployment as subject

_DESC_RUN = Path(
    "/root/autodl-tmp/AD-phase-b2-descriptor-real-extraction/"
    "artifacts/phase_b/b2_descriptor_artifacts/authoritative-run-a-20260729-013956"
)


def _is_accessible_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


@pytest.mark.skipif(not _is_accessible_dir(_DESC_RUN), reason="accepted descriptor Run A absent")
def test_embed_normalization_accepts_production_axes_stats() -> None:
    cfg = desc.load_descriptor_artifacts_config(
        "configs/phase_b/b2_descriptor_artifacts_gate_c.json"
    )
    verified = desc.verify_descriptor_artifact_collection(config=cfg, run_dir=_DESC_RUN)
    stats = dict(verified.normalization_statistics)
    assert "axes" in stats

    embedded = subject.embed_normalization(stats)
    assert embedded["format"] == "b2_03b_axes_v1"
    assert (
        embedded["descriptor_normalization_scientific_sha256"]
        == "f77975a94acf87a14b0753aabc9aad6777943ee4e4958b0a2083701cf4528594"
    )
    assert (
        embedded["descriptor_normalization_training_coverage_sha256"]
        == "e940f46bf696d326f8b982f15b8639f81e4548ec31a9b09634729811337e4c90"
    )
    assert len(embedded["by_depth"]["24"]["mean"]) == 4
    assert len(embedded["by_depth"]["24"]["mean"][0]) == 18

    raw = torch.zeros(1, 4, 18, dtype=torch.float32)
    out = subject.apply_embedded_normalization(raw, embedded, prediction_depth=24)
    assert out.shape == (1, 4, 18)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())


@pytest.mark.skipif(not _is_accessible_dir(_DESC_RUN), reason="accepted descriptor Run A absent")
def test_export_deployment_checkpoint_with_production_axes_stats() -> None:
    cfg = desc.load_descriptor_artifacts_config(
        "configs/phase_b/b2_descriptor_artifacts_gate_c.json"
    )
    verified = desc.verify_descriptor_artifact_collection(config=cfg, run_dir=_DESC_RUN)
    model = model_mod.B2DLCM(seed=17)
    ckpt = subject.export_deployment_checkpoint(
        training_model=model,
        normalization=dict(verified.normalization_statistics),
        canonical_seed=17,
        source_original_best_identity="aa" * 32,
        source_reproduction_best_identity="aa" * 32,
        contribution_target_collection_scientific_sha256="bb" * 32,
    )
    subject.run_cpu_golden_self_test(ckpt)
    assert ckpt["embedded_normalization"]["format"] == "b2_03b_axes_v1"
