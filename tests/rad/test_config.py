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
