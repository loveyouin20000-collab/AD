"""V3 model identity tests."""

from __future__ import annotations

import inspect

from rad.phase_b import b2_dlcm_v2 as v2
from rad.phase_b import b2_dlcm_v4 as v3


def test_architecture_pins_and_capacity() -> None:
    model = v3.B2DLCMV4(seed=17)
    assert list(model.candidate_layers) == [6, 12, 18, 24]
    assert list(model.prediction_depths) == [12, 18, 24]
    assert model.descriptor_dimension == 18
    assert model.hidden_dimension == 64
    assert v3.ARCHITECTURE_CONTRACT_VERSION == "b2_dlcm_architecture_v4"
    assert v3.MODEL_CLASS_ID.endswith("B2DLCMV4")


def test_v1_v2_history_unchanged() -> None:
    v1_id = v3.v1_immutable_identity()
    assert v1_id["tag"] == "b2-dlcm-unqualified-evidence-v1"
    assert v1_id["commit"] == v2.V1_EVIDENCE_COMMIT
    v2_id = v3.v2_immutable_identity()
    assert v2_id["tag"] == "b2-dlcm-decoupled-contract-v2"
    assert v2_id["commit"] == "e54f2b44eeb962b05cfb7cf74764e55905f1a8f6"


def test_category_absent_from_forward_signature() -> None:
    sig = inspect.signature(v3.B2DLCMV4.forward_training)
    assert "category" not in sig.parameters
