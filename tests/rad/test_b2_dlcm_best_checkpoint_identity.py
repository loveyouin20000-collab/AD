"""Best-checkpoint scientific identity must come from best weights, not last."""

from __future__ import annotations

from pathlib import Path

import torch

from rad.phase_b import b2_dlcm as dlcm
from rad.phase_b import b2_dlcm_training as training


def test_hermetic_training_reports_best_checkpoint_identity(tmp_path: Path) -> None:
    from rad.phase_b import b2_dlcm_official as official

    records = official.records_as_namespaces(training.build_hermetic_contract_records())
    env = training.collect_environment_contract(allow_cpu_for_hermetic=True)
    result = training.run_hermetic_contract_training(
        output_root=tmp_path / "run",
        seed=17,
        records=records,
        maximum_epochs=3,
        patience=10,
        device="cpu",
        batch_size=4,
        environment_contract=env,
        allow_existing_output=False,
        mark_real_training_started=False,
    )
    best_path = tmp_path / "run" / "seed_17" / "committed" / "best_training_checkpoint.pt"
    last_path = tmp_path / "run" / "seed_17" / "committed" / "last_training_checkpoint.pt"
    best_ck = torch.load(best_path, map_location="cpu", weights_only=False)
    last_ck = torch.load(last_path, map_location="cpu", weights_only=False)

    best_model = dlcm.B2DLCM(seed=None, initialize=False)
    best_model.load_state_dict(best_ck["model"], strict=True)
    last_model = dlcm.B2DLCM(seed=None, initialize=False)
    last_model.load_state_dict(last_ck["model"], strict=True)

    best_id = dlcm.model_state_scientific_sha256(best_model)
    last_id = dlcm.model_state_scientific_sha256(last_model)
    assert result["model_state_scientific_sha256"] == best_id
    assert result["last_model_state_scientific_sha256"] == last_id
    # With 3 epochs, last may equal best; identity fields must still be consistent.
    if best_ck["epoch"] != last_ck["epoch"]:
        assert best_id != last_id
