from __future__ import annotations

from pathlib import Path

import pytest

from rad.artifacts import assert_json_artifact_eligible_for_evaluation
from rad.errors import ArtifactIntegrityError
from rad.models.descriptors import DescriptorNormalizer

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_real_evaluation_rejects_static_descriptor_stats_fixture() -> None:
    with pytest.raises(ArtifactIntegrityError, match="test fixture"):
        assert_json_artifact_eligible_for_evaluation(
            FIXTURES / "descriptor_stats.json",
            kind="descriptor statistics",
        )


def test_descriptor_normalizer_load_rejects_static_test_fixture() -> None:
    with pytest.raises(ArtifactIntegrityError, match="test fixture"):
        DescriptorNormalizer.load(FIXTURES / "descriptor_stats.json")
