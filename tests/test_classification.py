"""Classification layout and output contract tests."""

from __future__ import annotations

import pytest
import torch

from maea_inference.classification import (
    BINARY_SINGLE_LOGIT,
    FLAT_MULTICLASS,
    STRUCTURED_CLASSIFICATION,
    build_classification_contract,
    build_classification_result,
    parse_classification_contract,
)
from maea_inference.errors import CheckpointError
from tests.support import binary_config, flat_config, structured_config


def test_binary_single_logit_uses_half_threshold() -> None:
    config = binary_config()
    result = build_classification_result(
        torch.tensor([[0.0]]),
        config,
        input_name="sample.png",
        device="cpu",
        threshold_metadata={"selected_threshold": 0.9},
    )

    assert result.layout == BINARY_SINGLE_LOGIT
    assert result.predictions[0].class_id == 1
    assert result.predictions[0].class_value == 1
    assert result.predictions[0].class_values == (0, 1)
    assert result.predictions[0].probabilities == pytest.approx((0.5, 0.5))
    assert result.to_ml_dict()["predictions"] == [[1]]
    assert result.to_ml_dict()["logits"][0] == pytest.approx([0.5, 0.5])
    assert result.threshold_metadata["selected_threshold"] == 0.9


def test_flat_multiclass_decodes_label_value() -> None:
    result = build_classification_result(
        torch.tensor([[0.0, 2.0, 1.0]]),
        flat_config(),
        input_name="sample.png",
        device="cpu",
    )

    assert result.layout == FLAT_MULTICLASS
    assert result.predictions[0].target is None
    assert result.predictions[0].class_id == 1
    assert result.predictions[0].class_value == "dog"
    assert result.predictions[0].class_values == ("cat", "dog", "bird")
    assert result.to_ml_dict()["predictions"] == [1]


def test_legacy_structured_preserves_class_target_axes() -> None:
    logits = torch.tensor([[4.0, 0.0, 0.0, 5.0, 1.0, 1.0]])
    result = build_classification_result(
        logits,
        structured_config(),
        input_name="sample.png",
        device="cpu",
    )

    assert result.layout == STRUCTURED_CLASSIFICATION
    assert result.to_ml_dict()["predictions"] == [[0, 1]]
    assert result.predictions[0].class_value == "low"
    assert result.predictions[0].class_values == ("low", "medium", "high")
    assert result.predictions[1].class_value == "oval"
    probabilities = result.to_ml_dict()["logits"]
    assert len(probabilities) == 1
    assert len(probabilities[0]) == 3
    assert all(len(class_values) == 2 for class_values in probabilities[0])


def test_contract_round_trip_has_exact_identity() -> None:
    contract = build_classification_contract(structured_config())
    assert contract is not None
    assert parse_classification_contract(contract.to_dict()) == contract
    assert contract.head_width == 6
    assert contract.layout_version == "legacy_structured_v1"


def test_unknown_contract_schema_is_rejected() -> None:
    contract = build_classification_contract(binary_config())
    assert contract is not None
    payload = contract.to_dict()
    payload["schema_version"] = 99

    with pytest.raises(CheckpointError, match=r"Unsupported.*schema"):
        parse_classification_contract(payload)
