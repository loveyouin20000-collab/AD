"""RED/GREEN tests completing B2-05A §§40–52 deployment/evaluation contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rad.phase_b import b2_dlcm as model_mod
from rad.phase_b import b2_dlcm_deployment as subject
from tests.rad.b2_dlcm_fixtures import (
    ACCEPTED_UPSTREAM,
    build_hermetic_dlcm_fixture,
    fixture_normalization_artifact,
    records_by_split,
)


def _toy_checkpoint(seed: int = 17) -> dict:
    model = model_mod.B2DLCM(seed=seed)
    return subject.export_deployment_checkpoint(
        training_model=model,
        normalization=fixture_normalization_artifact(),
        canonical_seed=seed,
        source_original_best_identity="aa" * 32,
        source_reproduction_best_identity="aa" * 32,
        contribution_target_collection_scientific_sha256=ACCEPTED_UPSTREAM[
            "contribution_target_collection_scientific_sha256"
        ],
    )


def test_qualification_cache_positive_only_and_invalidation() -> None:
    subject.clear_qualification_cache()
    ckpt = _toy_checkpoint()
    key_kwargs = dict(
        checkpoint_file_sha256="cc" * 32,
        environment_contract_sha256="dd" * 32,
        device=torch.device("cpu"),
        gpu_uuid="hermetic-a",
    )
    w1 = subject.load_qualified_deployment(ckpt, **key_kwargs)
    w2 = subject.load_qualified_deployment(ckpt, **key_kwargs)
    assert w1 is w2  # positive cache hit
    # Environment change forces requalification.
    w3 = subject.load_qualified_deployment(
        ckpt,
        checkpoint_file_sha256="cc" * 32,
        environment_contract_sha256="ee" * 32,
        device=torch.device("cpu"),
        gpu_uuid="hermetic-a",
    )
    assert w3 is not w1
    # Mutation invalidates and fails closed.
    with torch.no_grad():
        list(w1._model.parameters())[0].add_(1.0)
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_STATE_MUTATION"):
        w1.forward(torch.zeros(1, 2, 18), 12, (6, 12))
    assert w1._qualified is False
    # Failures are not cached as successes for the mutated wrapper key reuse:
    # a fresh load with same key still works because cache stores only positives
    # and mutation clears the entry.
    subject.invalidate_qualification_cache_entry(w1)
    w4 = subject.load_qualified_deployment(ckpt, **key_kwargs)
    assert w4 is not w1
    _ = w4.forward(torch.zeros(1, 2, 18), 12, (6, 12))


def test_batch_independence_b1_b2_b4() -> None:
    subject.clear_qualification_cache()
    ckpt = _toy_checkpoint(29)
    wrapper = subject.load_qualified_deployment(
        ckpt,
        checkpoint_file_sha256="11" * 32,
        environment_contract_sha256="22" * 32,
        device=torch.device("cpu"),
    )
    samples = [torch.randn(1, 2, 18) for _ in range(4)]
    singles = [wrapper.forward(s, 12, (6, 12)) for s in samples]
    for b in (1, 2, 4):
        batch = torch.cat(samples[:b], dim=0)
        out = wrapper.forward(batch, 12, (6, 12))
        for i in range(b):
            assert float((out[i : i + 1] - singles[i]).abs().max()) <= 1e-6
    # Reorder
    reordered = wrapper.forward(torch.cat([samples[3], samples[0]], dim=0), 12, (6, 12))
    assert float((reordered[0:1] - singles[3]).abs().max()) <= 1e-6
    assert float((reordered[1:2] - singles[0]).abs().max()) <= 1e-6


def test_ranking_accuracy_no_valid_pairs_coverage() -> None:
    pred = torch.tensor([0.1, 0.2, 0.3])
    tied = torch.ones(3)
    result = subject.pairwise_ranking_accuracy(pred, tied, tie_tolerance=1e-6)
    assert result["status"] == "no_valid_pairs"
    assert "accuracy" not in result or result["accuracy"] is None
    spread = torch.tensor([1.0, 0.0, -1.0])
    pred2 = torch.tensor([0.9, 0.1, -0.5])
    ok = subject.pairwise_ranking_accuracy(pred2, spread, tie_tolerance=1e-6)
    assert ok["status"] == "ok"
    assert 0.0 <= float(ok["accuracy"]) <= 1.0
    assert ok["valid_pair_count"] > 0


def test_evaluation_physically_locked_without_unlock(tmp_path: Path) -> None:
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_EVAL_LOCKED"):
        subject.require_evaluation_unlocked(tmp_path)
    unlock = subject.build_evaluation_unlock_artifact(
        canonical_selection_identity="11" * 32,
        reproduction_comparison={"status": "passed", "first_mismatch": None, "nodes_equal": True},
        trace_node_comparisons=[{"epoch": 0, "equal": True}],
        best_model_identity="22" * 32,
        last_model_identity="33" * 32,
        best_training_identity="44" * 32,
        last_training_identity="55" * 32,
        checkpoint_bytes_equal=True,
        deployment_scientific_identity="66" * 32,
        environment_identity="77" * 32,
        descriptor_normalization_identity=ACCEPTED_UPSTREAM[
            "descriptor_normalization_scientific_sha256"
        ],
        contribution_target_identity=ACCEPTED_UPSTREAM[
            "contribution_target_collection_scientific_sha256"
        ],
    )
    assert unlock["evaluation_unlocked"] is True
    path = tmp_path / "evaluation_unlock.json"
    subject.persist_evaluation_unlock(path, unlock)
    loaded = subject.require_evaluation_unlocked(tmp_path)
    assert loaded["evaluation_unlock_scientific_sha256"] == unlock[
        "evaluation_unlock_scientific_sha256"
    ]
    # CLI boolean must not unlock.
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_EVAL_CLI_BYPASS"):
        subject.require_evaluation_unlocked(tmp_path, cli_unlock_flag=True)


def test_seed_collection_and_reproduction_comparison() -> None:
    seeds = [
        {
            "seed": 17,
            "seed_scientific_sha256": "a1" * 32,
            "status": "passed",
            "calibration_primary": 0.5,
            "calibration_secondary": 1.0,
            "best_epoch": 3,
            "best_model_state_identity": "b1" * 32,
            "environment_identity": "e1" * 32,
            "file_sha256": "f1" * 32,
        },
        {
            "seed": 29,
            "seed_scientific_sha256": "a2" * 32,
            "status": "passed",
            "calibration_primary": 0.4,
            "calibration_secondary": 1.2,
            "best_epoch": 5,
            "best_model_state_identity": "b2" * 32,
            "environment_identity": "e1" * 32,
            "file_sha256": "f2" * 32,
        },
        {
            "seed": 43,
            "seed_scientific_sha256": "a3" * 32,
            "status": "passed",
            "calibration_primary": 0.45,
            "calibration_secondary": 0.9,
            "best_epoch": 4,
            "best_model_state_identity": "b3" * 32,
            "environment_identity": "e1" * 32,
            "file_sha256": "f3" * 32,
        },
    ]
    coll = subject.build_seed_collection_manifest(
        seeds,
        training_config_identity="c1" * 32,
        upstream_identities=ACCEPTED_UPSTREAM,
    )
    assert coll["ordered_seeds"] == [17, 29, 43]
    assert coll["evaluation_unlocked"] is False
    assert coll["canonical_seed_selected"] is False
    sci = subject.seed_collection_scientific_sha256(coll)
    assert "file_sha256" not in json.dumps(
        subject.seed_collection_scientific_payload(coll)
    )
    # file sha still verified externally
    subject.verify_seed_collection_file_hashes(coll, {17: "f1" * 32, 29: "f2" * 32, 43: "f3" * 32})
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_SEED_FILE_SHA"):
        subject.verify_seed_collection_file_hashes(coll, {17: "00" * 32, 29: "f2" * 32, 43: "f3" * 32})

    # Reproduction equality / first mismatch
    original = {"nodes": [{"epoch": 0, "h": "aa"}, {"epoch": 1, "h": "bb"}], "model": "m1"}
    good = {"nodes": [{"epoch": 0, "h": "aa"}, {"epoch": 1, "h": "bb"}], "model": "m1"}
    bad = {"nodes": [{"epoch": 0, "h": "aa"}, {"epoch": 1, "h": "XX"}], "model": "m1"}
    assert subject.compare_reproduction(original, good)["status"] == "passed"
    failed = subject.compare_reproduction(original, bad)
    assert failed["status"] == "canonical_reproduction_failed"
    assert failed["first_mismatch"]["epoch"] == 1
    _ = sci


def test_evaluation_artifact_schema_and_aggregation() -> None:
    records = build_hermetic_dlcm_fixture()
    eval_records = records_by_split(records)["evaluation"]
    # Hermetic target-fidelity style per-sample metrics.
    per_sample = []
    for rec in eval_records:
        per_sample.append(
            {
                "stable_sample_id": rec.stable_sample_id,
                "category": rec.category,
                "depth_24": {
                    "gt": {"kl": 0.1, "jsd": 0.05},
                    "teacher": {"kl": 0.12, "jsd": 0.06},
                },
            }
        )
    agg = subject.aggregate_target_fidelity(per_sample, metric_path=("depth_24", "gt", "kl"))
    assert "category_macro" in agg
    assert "pooled_diagnostic" in agg
    assert "per_category" in agg
    # Three-seed summary ddof=0
    import math

    values = [0.1, 0.2, 0.3]
    summary = subject.three_seed_summary(values, ddof=0)
    assert summary["mean"] == pytest.approx(0.2)
    mu = 0.2
    pop = math.sqrt(((0.1 - mu) ** 2 + (0.2 - mu) ** 2 + (0.3 - mu) ** 2) / 3)
    assert summary["std"] == pytest.approx(pop)


def test_evaluation_record_binds_required_identities(tmp_path: Path) -> None:
    unlock = subject.build_evaluation_unlock_artifact(
        canonical_selection_identity="11" * 32,
        reproduction_comparison={"status": "passed", "first_mismatch": None, "nodes_equal": True},
        trace_node_comparisons=[{"epoch": 0, "equal": True}],
        best_model_identity="22" * 32,
        last_model_identity="33" * 32,
        best_training_identity="44" * 32,
        last_training_identity="55" * 32,
        checkpoint_bytes_equal=True,
        deployment_scientific_identity="66" * 32,
        environment_identity="77" * 32,
        descriptor_normalization_identity=ACCEPTED_UPSTREAM[
            "descriptor_normalization_scientific_sha256"
        ],
        contribution_target_identity=ACCEPTED_UPSTREAM[
            "contribution_target_collection_scientific_sha256"
        ],
    )
    subject.persist_evaluation_unlock(tmp_path / "evaluation_unlock.json", unlock)
    record = subject.build_evaluation_record(
        evaluated_checkpoint_scientific_identity="66" * 32,
        evaluation_unlock_identity=unlock["evaluation_unlock_scientific_sha256"],
        evaluation_split_coverage_sha256=ACCEPTED_UPSTREAM["evaluation_target_coverage_sha256"],
        no_parameter_update_proof=True,
        depth_results={12: {"diagnostic": True}, 18: {"diagnostic": True}, 24: {"primary": True}},
        per_category={"bottle": {"pixel_ap": 0.5}},
        pooled={"pixel_ap": 0.5},
    )
    for key in (
        "evaluated_checkpoint_scientific_identity",
        "evaluation_unlock_identity",
        "evaluation_split_coverage_sha256",
        "no_parameter_update_proof",
        "depth_results",
        "per_category",
        "pooled",
    ):
        assert key in record
    manifest = subject.build_evaluation_manifest(
        records={
            "seed_17": record,
            "seed_29": record,
            "seed_43": record,
            "canonical_deployment": record,
        },
        unlock_identity=unlock["evaluation_unlock_scientific_sha256"],
    )
    out = tmp_path / "evaluation"
    subject.persist_evaluation_bundle(out, manifest, {
        "seed_17_evaluation.json": record,
        "seed_29_evaluation.json": record,
        "seed_43_evaluation.json": record,
        "canonical_deployment_evaluation.json": record,
    })
    assert (out / "evaluation_manifest.json").is_file()
    assert (out / "evaluation_manifest.json.sha256").is_file()


def test_accepted_manifest_formal_loader_requirements() -> None:
    ids = subject.build_accepted_manifest_identities(
        deploy_identity="d0" * 32,
        qualification_identity="q0" * 32,
        selection_identity="s0" * 32,
        upstream_identities={"descriptor": "u0" * 32},
    )
    manifest = subject.build_accepted_deployment_manifest(
        deploy_identity=ids["deploy_identity"],
        qualification_identity=ids["qualification_identity"],
        accepted_identity=ids["accepted_identity"],
        selection_identity="s0" * 32,
        deployment_scientific_sha256="d0" * 32,
        upstream_identities=ACCEPTED_UPSTREAM,
        deployment_qualified=True,
    )
    subject.verify_accepted_manifest_for_loader(
        manifest,
        deployment_scientific_sha256="d0" * 32,
        deployment_file_sha256="ff" * 32,
        expected_accepted_identity=ids["accepted_identity"],
    )
    bad = dict(manifest)
    bad["deployment_qualified"] = False
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_NOT_ACCEPTED"):
        subject.verify_accepted_manifest_for_loader(
            bad,
            deployment_scientific_sha256="d0" * 32,
            deployment_file_sha256="ff" * 32,
            expected_accepted_identity=ids["accepted_identity"],
        )


def test_undefined_formal_localization_fails_qualification() -> None:
    with pytest.raises(subject.B2DLCMDeploymentError, match="B2_DLCM_EVAL_METRIC_UNDEFINED"):
        subject.require_formal_localization_defined(
            {
                "bottle": {"pixel_ap": 0.1, "pixel_auroc": 0.2, "aupro": 0.3},
                "cable": {"pixel_ap": None, "pixel_auroc": 0.2, "aupro": 0.3},
            }
        )


def test_depth18_epoch0_uniform_fast_path_bitexact() -> None:
    maps = torch.randn(1, 3, 4, 4)
    ref = model_mod.reference_uniform_weights(3).view(1, 3)
    fused, path = model_mod.sum_preserving_fusion(
        maps,
        ref.contiguous(),
        prediction_depth=18,
        player_layer_ids=(6, 12, 18),
        return_path=True,
    )
    assert path == "uniform_baseline"
    assert torch.equal(fused, maps.sum(dim=1))


def test_gpu_qualification_entrypoint_importable() -> None:
    # Dedicated entrypoint module for B2-05B GPU runs; CPU CI skips body.
    from tests.rad import test_b2_dlcm_gpu_qualification as gpu_mod

    assert hasattr(gpu_mod, "test_gpu_golden_qualification_nine_cases")
