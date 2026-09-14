"""MAEA imaging-classification compatibility contracts and output handling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from maea_inference.errors import CheckpointError, ConfigurationError
from maea_inference.results import ClassificationPrediction, ClassificationResult

BINARY_SINGLE_LOGIT = "binary_single_logit"
FLAT_MULTICLASS = "flat_multiclass"
STRUCTURED_CLASSIFICATION = "structured_classification"
LEGACY_STRUCTURED_V1 = "legacy_structured_v1"
FLAT_MULTICLASS_V2 = "flat_multiclass_v2"
CONTRACT_SCHEMA_VERSION = 1

BINARY_LOSSES = frozenset({"nn.BCEWithLogitsLoss", "nn.BCELoss", "maea.FocalLoss"})
ONE_HOT_ENCODINGS = frozenset({"one_hot_multiclass", "one_hot_or_multi_target"})
IDENTITY_FIELDS = (
    "schema_version",
    "layout",
    "layout_version",
    "head_width",
    "ordered_targets",
    "class_count",
    "target_encoding",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def target_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the nested training target mapping."""
    return _mapping(_mapping(config.get("training")).get("target"))


def target_labels(config: Mapping[str, Any]) -> list[str]:
    """Return non-empty target labels in artifact order."""
    raw_labels = target_config(config).get("labels")
    if not isinstance(raw_labels, list):
        return []
    return [str(label).strip() for label in raw_labels if str(label).strip()]


def target_encoding(config: Mapping[str, Any]) -> str:
    """Resolve target encoding using the MAEA export precedence."""
    direct = str(target_config(config).get("target_encoding") or "").strip().lower()
    if direct:
        return direct
    dataset_value = (
        str(_mapping(config.get("dataset")).get("classification_target_encoding") or "")
        .strip()
        .lower()
    )
    if dataset_value:
        return dataset_value
    runtime = _mapping(_mapping(config.get("runtime_metadata")).get("classification"))
    return str(runtime.get("target_encoding") or "").strip().lower()


def explicit_class_count(config: Mapping[str, Any]) -> int | None:
    """Return an explicitly configured positive classification class count."""
    candidates = (
        target_config(config).get("class_count"),
        _mapping(config.get("dataset")).get("classification_class_count"),
        _mapping(config.get("dataset")).get("max_num_classes"),
    )
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def classification_class_count(config: Mapping[str, Any]) -> int:
    """Resolve class count with legacy label-count fallback."""
    return explicit_class_count(config) or max(1, len(target_labels(config)))


def _criterion_module(config: Mapping[str, Any]) -> str:
    criterion = _mapping(config.get("criterion"))
    return str(criterion.get("module") or criterion.get("type") or "").strip()


def _uses_binary_loss(config: Mapping[str, Any]) -> bool:
    module = _criterion_module(config)
    if module == "maea.FocalLoss":
        mode = str(
            _mapping(_mapping(config.get("criterion")).get("params")).get("mode")
            or "auto"
        )
        return mode.strip().lower() != "multiclass"
    return module in BINARY_LOSSES


def explicit_layout_version(config: Mapping[str, Any]) -> str:
    """Read and validate an explicit imaging-classification layout version."""
    version = (
        str(target_config(config).get("classification_output_layout_version") or "")
        .strip()
        .lower()
    )
    if version not in {"", LEGACY_STRUCTURED_V1, FLAT_MULTICLASS_V2}:
        raise ConfigurationError(
            "Unsupported classification output layout version: %s" % version
        )
    return version


def classification_layout(config: Mapping[str, Any]) -> str:
    """Resolve the current MAEA imaging-classification tensor layout."""
    version = explicit_layout_version(config)
    labels = target_labels(config)
    if version == FLAT_MULTICLASS_V2:
        direct_encoding = (
            str(target_config(config).get("target_encoding") or "").strip().lower()
        )
        if len(labels) <= 1 or direct_encoding not in ONE_HOT_ENCODINGS:
            raise ConfigurationError(
                "flat_multiclass_v2 requires multiple labels and an explicit "
                "training.target one-hot target encoding"
            )
        return FLAT_MULTICLASS
    if (
        len(labels) == 1
        and classification_class_count(config) <= 2
        and _uses_binary_loss(config)
    ):
        return BINARY_SINGLE_LOGIT
    return STRUCTURED_CLASSIFICATION


def classification_layout_version(config: Mapping[str, Any]) -> str | None:
    """Return the versioned layout marker saved by MAEA checkpoints."""
    version = explicit_layout_version(config)
    if version:
        return version
    if len(target_labels(config)) > 1:
        return LEGACY_STRUCTURED_V1
    return None


def classification_head_width(config: Mapping[str, Any]) -> int:
    """Resolve the model output width for the current classification layout."""
    layout = classification_layout(config)
    if layout == BINARY_SINGLE_LOGIT:
        return 1
    class_count = classification_class_count(config)
    labels = target_labels(config)
    if layout == FLAT_MULTICLASS:
        return max(1, len(labels))
    if len(labels) > 1:
        return max(1, class_count * len(labels))
    if class_count > 1:
        return class_count
    configured = _mapping(_mapping(config.get("architecture")).get("params")).get(
        "num_classes"
    )
    if configured is None:
        return 1
    try:
        return max(1, int(configured))
    except (TypeError, ValueError):
        return 1


@dataclass(frozen=True)
class ClassificationContract:
    """Versioned identity contract persisted in MAEA checkpoints."""

    schema_version: int
    layout: str
    layout_version: str | None
    head_width: int
    ordered_targets: tuple[str, ...]
    class_count: int
    target_encoding: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contract with the checkpoint field names."""
        return {
            "schema_version": self.schema_version,
            "layout": self.layout,
            "layout_version": self.layout_version,
            "head_width": self.head_width,
            "ordered_targets": list(self.ordered_targets),
            "class_count": self.class_count,
            "target_encoding": self.target_encoding,
        }


def build_classification_contract(
    config: Mapping[str, Any],
) -> ClassificationContract | None:
    """Build the contract implied by a runtime config."""
    if str(target_config(config).get("type") or "").strip() != "classification":
        return None
    encoding = target_encoding(config)
    return ClassificationContract(
        schema_version=CONTRACT_SCHEMA_VERSION,
        layout=classification_layout(config),
        layout_version=classification_layout_version(config),
        head_width=classification_head_width(config),
        ordered_targets=tuple(target_labels(config)),
        class_count=classification_class_count(config),
        target_encoding=encoding or None,
    )


def _contract_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError("Classification checkpoint contract must be an object")
    missing = sorted(set(IDENTITY_FIELDS) - set(value))
    if missing:
        raise CheckpointError(
            "Classification checkpoint contract is missing identity fields: %s"
            % missing
        )
    return value


def _contract_numbers(value: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        numbers = (
            int(value["schema_version"]),
            int(value["head_width"]),
            int(value["class_count"]),
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            "Classification contract numeric fields must be integers"
        ) from exc
    schema_version, head_width, class_count = numbers
    if schema_version != CONTRACT_SCHEMA_VERSION:
        raise CheckpointError(
            "Unsupported classification checkpoint contract schema: %s" % schema_version
        )
    if head_width <= 0 or class_count <= 0:
        raise CheckpointError(
            "Classification contract head_width and class_count must be positive"
        )
    return numbers


def _contract_layout(value: Mapping[str, Any]) -> tuple[str, str | None]:
    layout = str(value["layout"] or "").strip().lower()
    if layout not in {
        BINARY_SINGLE_LOGIT,
        FLAT_MULTICLASS,
        STRUCTURED_CLASSIFICATION,
    }:
        raise CheckpointError(
            "Unsupported classification checkpoint layout: %s" % layout
        )
    version = str(value["layout_version"] or "").strip().lower() or None
    if version not in {None, LEGACY_STRUCTURED_V1, FLAT_MULTICLASS_V2}:
        raise CheckpointError(
            "Unsupported classification checkpoint layout version: %s" % version
        )
    return layout, version


def _contract_targets(value: Mapping[str, Any]) -> tuple[str, ...]:
    raw_targets = value["ordered_targets"]
    if not isinstance(raw_targets, list):
        raise CheckpointError(
            "Classification checkpoint ordered_targets must be a list"
        )
    targets = tuple(str(label).strip() for label in raw_targets if str(label).strip())
    if not targets:
        raise CheckpointError(
            "Classification checkpoint ordered_targets cannot be empty"
        )
    return targets


def _validate_contract_layout(
    layout: str,
    version: str | None,
    ordered_targets: tuple[str, ...],
) -> None:
    valid_version = (
        version == LEGACY_STRUCTURED_V1 and layout == STRUCTURED_CLASSIFICATION
    ) or (version == FLAT_MULTICLASS_V2 and layout == FLAT_MULTICLASS)
    if version is not None and not valid_version:
        raise CheckpointError(
            "Classification checkpoint layout and layout_version are inconsistent"
        )
    if (
        layout == STRUCTURED_CLASSIFICATION
        and len(ordered_targets) > 1
        and version != LEGACY_STRUCTURED_V1
    ):
        raise CheckpointError(
            "Multi-target structured checkpoints require legacy_structured_v1"
        )


def parse_classification_contract(
    value: Any,
) -> ClassificationContract:
    """Parse an explicit checkpoint contract and fail closed on unknown shapes."""
    contract = _contract_mapping(value)
    schema_version, head_width, class_count = _contract_numbers(contract)
    layout, version = _contract_layout(contract)
    ordered_targets = _contract_targets(contract)
    _validate_contract_layout(layout, version, ordered_targets)
    encoding = str(contract["target_encoding"] or "").strip().lower() or None
    return ClassificationContract(
        schema_version=schema_version,
        layout=layout,
        layout_version=version,
        head_width=head_width,
        ordered_targets=ordered_targets,
        class_count=class_count,
        target_encoding=encoding,
    )


def contract_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> ClassificationContract | None:
    """Read explicit contract metadata or derive it from the saved config."""
    if "classification_contract" in checkpoint:
        return parse_classification_contract(checkpoint["classification_contract"])
    saved_config = checkpoint.get("config")
    if isinstance(saved_config, Mapping):
        return build_classification_contract(saved_config)
    return None


def _class_values(
    config: Mapping[str, Any], target: str | None
) -> Sequence[Any] | None:
    target_data = target_config(config)
    direct = target_data.get("class_values")
    if isinstance(direct, list):
        return direct
    if isinstance(direct, Mapping) and target is not None:
        target_values = direct.get(target)
        if isinstance(target_values, list):
            return target_values
    candidates = (
        target_data.get("vocabularies"),
        _mapping(_mapping(config.get("runtime_metadata")).get("classification")).get(
            "vocabularies"
        ),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping) and target is not None:
            values = candidate.get(target)
            if isinstance(values, list):
                return values
    return None


def _available_class_values(
    config: Mapping[str, Any],
    target: str | None,
    class_count: int,
    *,
    fallback: Sequence[Any] | None = None,
) -> tuple[Any, ...]:
    configured = _class_values(config, target)
    if configured is not None and len(configured) == class_count:
        return tuple(configured)
    if fallback is not None and len(fallback) == class_count:
        return tuple(fallback)
    return tuple(range(class_count))


def _selected_class_value(values: Sequence[Any], class_id: int) -> Any | None:
    if class_id < 0 or class_id >= len(values):
        return None
    return values[class_id]


def _classification_sample_shape(config: Mapping[str, Any]) -> tuple[int, ...]:
    layout = classification_layout(config)
    if layout == BINARY_SINGLE_LOGIT:
        return (1,)
    if layout == FLAT_MULTICLASS:
        return (classification_head_width(config),)
    return classification_class_count(config), max(1, len(target_labels(config)))


def reshape_classification_output(
    logits: torch.Tensor, config: Mapping[str, Any]
) -> torch.Tensor:
    """Apply the per-sample shape defined by the compatibility contract."""
    if logits.dim() == 0:
        raise ConfigurationError(
            "Classification output must include a sample dimension"
        )
    sample_shape = _classification_sample_shape(config)
    values_per_sample = int(np.prod(sample_shape))
    batch_size = 1 if logits.dim() == 1 and values_per_sample > 1 else logits.shape[0]
    if logits.numel() != int(batch_size) * values_per_sample:
        raise ConfigurationError(
            "Classification output shape %s is incompatible with layout %s; "
            "expected %d values per sample"
            % (tuple(logits.shape), classification_layout(config), values_per_sample)
        )
    return logits.reshape(int(batch_size), *sample_shape)


@dataclass(frozen=True)
class _ClassificationOutput:
    predictions: torch.Tensor
    probabilities: torch.Tensor
    client_predictions: tuple[ClassificationPrediction, ...]


def _binary_output(
    output: torch.Tensor,
    config: Mapping[str, Any],
    labels: list[str],
    threshold_metadata: Mapping[str, Any],
) -> _ClassificationOutput:
    positive = torch.sigmoid(output.detach().float())
    predictions = (positive >= 0.5).long()
    probabilities = torch.cat([1.0 - positive, positive], dim=1)
    class_id = int(predictions[0, 0].item())
    fallback_values = (
        threshold_metadata.get("negative_value", 0),
        threshold_metadata.get("positive_value", 1),
    )
    class_values = _available_class_values(
        config,
        labels[0],
        2,
        fallback=fallback_values,
    )
    client = ClassificationPrediction(
        target=labels[0],
        class_id=class_id,
        class_value=_selected_class_value(class_values, class_id),
        class_values=class_values,
        probabilities=tuple(float(value) for value in probabilities[0]),
    )
    return _ClassificationOutput(predictions, probabilities, (client,))


def _flat_output(
    output: torch.Tensor,
    labels: list[str],
) -> _ClassificationOutput:
    probabilities = torch.softmax(output, dim=1)
    predictions = torch.argmax(output, dim=1)
    class_id = int(predictions[0].item())
    class_values = tuple(labels)
    client = ClassificationPrediction(
        target=None,
        class_id=class_id,
        class_value=_selected_class_value(class_values, class_id),
        class_values=class_values,
        probabilities=tuple(float(value) for value in probabilities[0]),
    )
    return _ClassificationOutput(predictions, probabilities, (client,))


def _structured_output(
    output: torch.Tensor,
    config: Mapping[str, Any],
    labels: list[str],
) -> _ClassificationOutput:
    probabilities = torch.softmax(output, dim=1)
    predictions = torch.argmax(output, dim=1)
    client_predictions = []
    for target_index, label in enumerate(labels):
        class_id = int(predictions[0, target_index].item())
        class_values = _available_class_values(
            config,
            label,
            probabilities.shape[1],
        )
        client_predictions.append(
            ClassificationPrediction(
                target=label,
                class_id=class_id,
                class_value=_selected_class_value(class_values, class_id),
                class_values=class_values,
                probabilities=tuple(
                    float(probabilities[0, class_index, target_index].item())
                    for class_index in range(probabilities.shape[1])
                ),
            )
        )
    return _ClassificationOutput(
        predictions,
        probabilities,
        tuple(client_predictions),
    )


def _classify_output(
    output: torch.Tensor,
    config: Mapping[str, Any],
    layout: str,
    labels: list[str],
    threshold_metadata: Mapping[str, Any],
) -> _ClassificationOutput:
    if layout == BINARY_SINGLE_LOGIT:
        return _binary_output(output, config, labels, threshold_metadata)
    if layout == FLAT_MULTICLASS:
        return _flat_output(output, labels)
    return _structured_output(output, config, labels)


def build_classification_result(
    logits: torch.Tensor,
    config: Mapping[str, Any],
    *,
    input_name: str,
    device: str,
    threshold_metadata: Mapping[str, Any] | None = None,
) -> ClassificationResult:
    """Transform one model output into client and legacy-compatible results."""
    output = reshape_classification_output(logits, config)
    layout = classification_layout(config)
    labels = target_labels(config)
    metadata = dict(threshold_metadata or {})
    classified = _classify_output(output, config, layout, labels, metadata)

    return ClassificationResult(
        input_name=input_name,
        device=device,
        layout=layout,
        predictions=classified.client_predictions,
        threshold_metadata=metadata,
        ml_predictions=classified.predictions.detach().cpu().numpy().tolist(),
        ml_probabilities=classified.probabilities.detach().cpu().numpy().tolist(),
    )
