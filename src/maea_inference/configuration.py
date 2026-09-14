"""Load and validate MAEA imaging inference configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from maea_inference.checkpoints import checkpoint_config
from maea_inference.classification import (
    BINARY_SINGLE_LOGIT,
    ClassificationContract,
    build_classification_contract,
    contract_from_checkpoint,
    explicit_class_count,
    explicit_layout_version,
    target_config,
    target_encoding,
    target_labels,
)
from maea_inference.constants import SEGMENTATION_MODELS
from maea_inference.errors import CheckpointError, ConfigurationError


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mutable_mapping(value: Any, field: str) -> MutableMapping[str, Any]:
    if not isinstance(value, MutableMapping):
        raise ConfigurationError("%s must be a JSON object" % field)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON task configuration from a local file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError("Configuration file not found: %s" % config_path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            value = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            "Could not read JSON configuration %s: %s" % (config_path, str(exc))
        ) from exc
    if not isinstance(value, dict):
        raise ConfigurationError("Configuration root must be a JSON object")
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(_mapping(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _architecture_module(config: Mapping[str, Any]) -> str:
    architecture = _mapping(config.get("architecture"))
    return str(architecture.get("module") or architecture.get("type") or "").strip()


def _target_type(config: Mapping[str, Any]) -> str:
    return str(target_config(config).get("type") or "").strip()


def _validate_overlapping_artifact_identity(
    supplied: Mapping[str, Any], embedded: Mapping[str, Any]
) -> None:
    supplied_module = _architecture_module(supplied)
    embedded_module = _architecture_module(embedded)
    if supplied_module and embedded_module and supplied_module != embedded_module:
        raise CheckpointError(
            "Checkpoint/config architecture mismatch: checkpoint=%s config=%s"
            % (embedded_module, supplied_module)
        )
    supplied_type = _target_type(supplied)
    embedded_type = _target_type(embedded)
    if supplied_type and embedded_type and supplied_type != embedded_type:
        raise CheckpointError(
            "Checkpoint/config target type mismatch: checkpoint=%s config=%s"
            % (embedded_type, supplied_type)
        )
    supplied_labels = target_labels(supplied)
    embedded_labels = target_labels(embedded)
    if supplied_labels and embedded_labels and supplied_labels != embedded_labels:
        raise CheckpointError(
            "Checkpoint/config target label mismatch: checkpoint=%s config=%s"
            % (embedded_labels, supplied_labels)
        )
    supplied_size = configured_image_size(supplied)
    embedded_size = configured_image_size(embedded)
    if supplied_size and embedded_size and supplied_size != embedded_size:
        raise CheckpointError(
            "Checkpoint/config image size mismatch: checkpoint=%s config=%s"
            % (embedded_size, supplied_size)
        )
    _validate_overlapping_architecture_params(supplied, embedded)


def _validate_overlapping_architecture_params(
    supplied: Mapping[str, Any], embedded: Mapping[str, Any]
) -> None:
    supplied_params = _mapping(_mapping(supplied.get("architecture")).get("params"))
    embedded_params = _mapping(_mapping(embedded.get("architecture")).get("params"))
    ignored = {"pretrained", "encoder_weights"}
    if _target_type(supplied) == _target_type(embedded) == "classification":
        # Historical structured checkpoints store the flattened output-head
        # width here, while exported runtime configs store the target count.
        # The classification contract below validates the semantic class count
        # and exact head width before the state dictionary is loaded.
        ignored.add("num_classes")
    mismatches = sorted(
        key
        for key in set(supplied_params) & set(embedded_params)
        if key not in ignored and supplied_params[key] != embedded_params[key]
    )
    if mismatches:
        raise CheckpointError(
            "Checkpoint/config architecture parameter mismatch: %s" % mismatches
        )


def _set_nested_contract_fields(
    config: MutableMapping[str, Any], contract: ClassificationContract
) -> None:
    training = _mutable_mapping(config.setdefault("training", {}), "training")
    target = _mutable_mapping(training.setdefault("target", {}), "training.target")
    dataset = _mutable_mapping(config.setdefault("dataset", {}), "dataset")

    configured_labels = target_labels(config)
    if configured_labels and tuple(configured_labels) != contract.ordered_targets:
        raise CheckpointError(
            "Classification checkpoint/config ordered target mismatch"
        )
    configured_count = explicit_class_count(config)
    if configured_count is not None and configured_count != contract.class_count:
        raise CheckpointError(
            "Classification checkpoint/config class count mismatch: "
            "checkpoint=%d config=%d" % (contract.class_count, configured_count)
        )
    configured_version = explicit_layout_version(config)
    if configured_version and configured_version != (contract.layout_version or ""):
        raise CheckpointError(
            "Classification checkpoint/config layout version mismatch"
        )
    configured_encoding = target_encoding(config)
    if configured_encoding and configured_encoding != (contract.target_encoding or ""):
        raise CheckpointError(
            "Classification checkpoint/config target encoding mismatch"
        )

    target["labels"] = list(contract.ordered_targets)
    target["class_count"] = contract.class_count
    dataset["classification_class_count"] = contract.class_count
    if contract.layout_version:
        target["classification_output_layout_version"] = contract.layout_version
    if contract.target_encoding:
        target["target_encoding"] = contract.target_encoding
    if contract.layout == BINARY_SINGLE_LOGIT and not _mapping(
        config.get("criterion")
    ).get("module"):
        config["criterion"] = {
            "module": "nn.BCEWithLogitsLoss",
            "params": {},
        }


def _validate_contract_identity(
    config: Mapping[str, Any], artifact: ClassificationContract
) -> None:
    current = build_classification_contract(config)
    if current is None:
        raise CheckpointError(
            "Classification checkpoint was paired with a non-classification config"
        )
    mismatches = [
        field
        for field in (
            "layout",
            "layout_version",
            "head_width",
            "ordered_targets",
            "class_count",
            "target_encoding",
        )
        if getattr(current, field) != getattr(artifact, field)
    ]
    if mismatches:
        raise CheckpointError(
            "Classification checkpoint/config contract mismatch: %s" % mismatches
        )


def _configured_image_size_candidates(config: Mapping[str, Any]) -> tuple[Any, ...]:
    architecture = _mapping(config.get("architecture"))
    training = _mapping(config.get("training"))
    target = _mapping(training.get("target"))
    return (
        architecture.get("input_size"),
        _mapping(architecture.get("params")).get("input_size"),
        training.get("image_size"),
        target.get("image_size"),
    )


def configured_image_size(config: Mapping[str, Any]) -> tuple[int, int] | None:
    """Resolve an explicitly configured image size without applying a default."""
    for candidate in _configured_image_size_candidates(config):
        if not isinstance(candidate, (list, tuple)) or len(candidate) != 2:
            continue
        try:
            first, second = int(candidate[0]), int(candidate[1])
        except (TypeError, ValueError):
            continue
        if first > 0 and second > 0:
            return first, second
    return None


def resolve_image_size(config: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve the image-size tuple using the MAEA export precedence."""
    return configured_image_size(config) or (224, 224)


def _normalize_runtime_config(
    config: MutableMapping[str, Any],
) -> tuple[MutableMapping[str, Any], str, str, str, list[str]]:
    architecture = _mutable_mapping(config.get("architecture"), "architecture")
    module = _architecture_module(config)
    if not module or ":" not in module:
        raise ConfigurationError(
            "architecture.module must use a supported prefix, for example timm:resnet18"
        )
    architecture["module"] = module
    architecture["params"] = dict(_mapping(architecture.get("params")))
    training = _mutable_mapping(config.get("training"), "training")
    target = _mutable_mapping(training.get("target"), "training.target")
    task_type = str(target.get("type") or "").strip()
    labels = target_labels(config)
    if not labels:
        raise ConfigurationError("training.target.labels must be a non-empty list")
    target["labels"] = labels
    if not isinstance(config.get("dataset"), MutableMapping):
        config["dataset"] = {}
    source, name = module.split(":", 1)
    return architecture, task_type, source, name, labels


def _validate_classification_architecture(
    config: Mapping[str, Any], source: str, name: str
) -> None:
    if source not in {"timm", "efficientnet"} or not name:
        raise ConfigurationError(
            "Classification supports only timm:* and efficientnet:* architectures"
        )
    if build_classification_contract(config) is None:
        raise ConfigurationError("Could not resolve classification contract")


def _validate_segmentation_architecture(
    architecture: Mapping[str, Any],
    source: str,
    name: str,
    labels: list[str],
) -> None:
    if source != "segmentation" or name.lower() not in SEGMENTATION_MODELS:
        raise ConfigurationError(
            "Unsupported semantic segmentation architecture: %s:%s" % (source, name)
        )
    configured_classes = _mapping(architecture.get("params")).get("num_classes")
    if configured_classes is None:
        raise ConfigurationError(
            "Semantic segmentation requires architecture.params.num_classes"
        )
    try:
        foreground_classes = int(configured_classes)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "architecture.params.num_classes must be a positive integer"
        ) from exc
    if foreground_classes != len(labels):
        raise ConfigurationError(
            "Semantic segmentation num_classes must equal the number of "
            "foreground target labels: num_classes=%d labels=%d"
            % (foreground_classes, len(labels))
        )


def validate_runtime_config(config: MutableMapping[str, Any]) -> None:
    """Validate task and architecture boundaries for standalone v0.1.0."""
    architecture, task_type, source, name, labels = _normalize_runtime_config(config)
    if task_type == "classification":
        _validate_classification_architecture(config, source, name)
        return
    if task_type == "segmentation":
        _validate_segmentation_architecture(architecture, source, name, labels)
        return
    if task_type in {
        "detection",
        "box",
        "instance_segmentation",
        "panoptic_segmentation",
    }:
        raise ConfigurationError(
            "Task type %s is outside maea-inference v0.1.0" % task_type
        )
    raise ConfigurationError("Unsupported imaging target type: %s" % task_type)


def resolve_runtime_config(
    supplied_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Merge artifact metadata, validate identity, and set the local model path."""
    embedded = checkpoint_config(checkpoint)
    if embedded is not None:
        _validate_overlapping_artifact_identity(supplied_config, embedded)
        resolved = _deep_merge(embedded, supplied_config)
    else:
        resolved = deepcopy(dict(supplied_config))

    artifact_contract = contract_from_checkpoint(checkpoint)
    if artifact_contract is not None:
        _set_nested_contract_fields(resolved, artifact_contract)

    architecture = _mutable_mapping(
        resolved.setdefault("architecture", {}), "architecture"
    )
    architecture["model_path"] = str(Path(checkpoint_path).expanduser().resolve())
    validate_runtime_config(resolved)
    if artifact_contract is not None:
        _validate_contract_identity(resolved, artifact_contract)
    return resolved
