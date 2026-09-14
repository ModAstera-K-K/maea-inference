"""Artifact/config validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from maea_inference.classification import build_classification_contract
from maea_inference.configuration import (
    resolve_image_size,
    resolve_runtime_config,
    validate_runtime_config,
)
from maea_inference.errors import CheckpointError, ConfigurationError
from tests.support import (
    binary_config,
    copy_config,
    flat_config,
    segmentation_config,
    structured_config,
)


def _checkpoint(config: dict[str, object]) -> dict[str, object]:
    contract = build_classification_contract(config)
    return {
        "config": config,
        "classification_contract": contract.to_dict() if contract else None,
        "model_state_dict": {},
    }


def test_checkpoint_path_overrides_exported_model_path(tmp_path: Path) -> None:
    config = binary_config()
    config["architecture"]["model_path"] = "configured-model.pth"
    checkpoint_path = tmp_path / "model.pth"
    resolved = resolve_runtime_config(
        config,
        _checkpoint(config),
        checkpoint_path=checkpoint_path,
    )

    assert resolved["architecture"]["model_path"] == str(checkpoint_path.resolve())


def test_checkpoint_architecture_mismatch_is_rejected(tmp_path: Path) -> None:
    embedded = binary_config()
    supplied = copy_config(embedded)
    supplied["architecture"]["module"] = "timm:resnet34"

    with pytest.raises(CheckpointError, match="architecture mismatch"):
        resolve_runtime_config(
            supplied,
            _checkpoint(embedded),
            checkpoint_path=tmp_path / "model.pth",
        )


def test_checkpoint_contract_target_mismatch_is_rejected(tmp_path: Path) -> None:
    embedded = binary_config()
    supplied = copy_config(embedded)
    supplied["training"]["target"]["labels"] = ["other"]

    with pytest.raises(CheckpointError, match="target label mismatch"):
        resolve_runtime_config(
            supplied,
            _checkpoint(embedded),
            checkpoint_path=tmp_path / "model.pth",
        )


def test_legacy_structured_head_width_does_not_conflict_with_target_count(
    tmp_path: Path,
) -> None:
    embedded = structured_config()
    supplied = copy_config(embedded)
    supplied["architecture"]["params"]["num_classes"] = 2

    resolved = resolve_runtime_config(
        supplied,
        {"config": embedded, "model_state_dict": {}},
        checkpoint_path=tmp_path / "model.pth",
    )

    assert resolved["training"]["target"]["class_count"] == 3
    assert (
        resolved["training"]["target"]["classification_output_layout_version"]
        == "legacy_structured_v1"
    )


def test_segmentation_num_classes_mismatch_is_rejected(tmp_path: Path) -> None:
    embedded = segmentation_config()
    supplied = copy_config(embedded)
    supplied["architecture"]["params"]["num_classes"] = 2

    with pytest.raises(CheckpointError, match="architecture parameter mismatch"):
        resolve_runtime_config(
            supplied,
            {"config": embedded, "model_state_dict": {}},
            checkpoint_path=tmp_path / "model.pth",
        )


def test_checkpoint_image_size_mismatch_is_rejected(tmp_path: Path) -> None:
    embedded = binary_config()
    embedded["architecture"]["input_size"] = [32, 32]
    supplied = copy_config(embedded)
    supplied["architecture"]["input_size"] = [64, 64]

    with pytest.raises(CheckpointError, match="image size mismatch"):
        resolve_runtime_config(
            supplied,
            _checkpoint(embedded),
            checkpoint_path=tmp_path / "model.pth",
        )


def test_malformed_embedded_checkpoint_config_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match=r"checkpoint\.config must be an object"):
        resolve_runtime_config(
            binary_config(),
            {"config": [], "model_state_dict": {}},
            checkpoint_path=tmp_path / "model.pth",
        )


def test_image_size_precedence_matches_export_contract() -> None:
    config = binary_config()
    config["architecture"]["input_size"] = [31, 32]
    config["architecture"]["params"]["input_size"] = [41, 42]
    config["training"]["image_size"] = [51, 52]
    config["training"]["target"]["image_size"] = [61, 62]

    assert resolve_image_size(config) == (31, 32)


def test_semantic_class_count_must_match_foreground_labels() -> None:
    config = segmentation_config()
    config["architecture"]["params"]["num_classes"] = 2

    with pytest.raises(ConfigurationError, match="foreground target labels"):
        validate_runtime_config(config)


def test_semantic_class_count_is_required() -> None:
    config = segmentation_config()
    del config["architecture"]["params"]["num_classes"]

    with pytest.raises(ConfigurationError, match=r"requires.*num_classes"):
        validate_runtime_config(config)


def test_flat_imaging_encoding_must_be_on_training_target() -> None:
    config = flat_config()
    del config["training"]["target"]["target_encoding"]
    config["dataset"]["classification_target_encoding"] = "one_hot_multiclass"

    with pytest.raises(ConfigurationError, match=r"training\.target"):
        validate_runtime_config(config)


@pytest.mark.parametrize(
    "task_type",
    ["detection", "instance_segmentation", "panoptic_segmentation"],
)
def test_unsupported_imaging_tasks_are_explicit(task_type: str) -> None:
    config = segmentation_config()
    config["training"]["target"]["type"] = task_type

    with pytest.raises(ConfigurationError, match="outside maea-inference"):
        validate_runtime_config(config)
