"""Typed public result objects for standalone inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maea_inference.constants import RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class ClassificationPrediction:
    """One target-level classification decision."""

    target: str | None
    class_id: int
    class_value: Any | None
    class_values: tuple[Any, ...]
    probabilities: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible target prediction."""
        return {
            "target": self.target,
            "class_id": self.class_id,
            "class_value": self.class_value,
            "class_values": list(self.class_values),
            "probabilities": list(self.probabilities),
        }


@dataclass(frozen=True)
class ClassificationResult:
    """Client-friendly and legacy-compatible classification output."""

    input_name: str
    device: str
    layout: str
    predictions: tuple[ClassificationPrediction, ...]
    threshold_metadata: dict[str, Any]
    ml_predictions: list[Any]
    ml_probabilities: list[Any]
    schema_version: int = RESULT_SCHEMA_VERSION
    task_type: str = "classification"

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned client-facing result payload."""
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "input": self.input_name,
            "device": self.device,
            "layout": self.layout,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "threshold_metadata": self.threshold_metadata,
        }

    def to_ml_dict(self) -> dict[str, Any]:
        """Return the legacy compatibility keys and tensor layouts."""
        return {
            "predictions": self.ml_predictions,
            "explainer": [],
            "logits": self.ml_probabilities,
        }


@dataclass(frozen=True)
class SegmentationResult:
    """Client-friendly semantic-segmentation result and local artifacts."""

    input_name: str
    device: str
    labels: tuple[str, ...]
    mask: np.ndarray
    probabilities: np.ndarray
    colors: tuple[tuple[int, int, int], ...]
    polygons: tuple[dict[str, Any], ...]
    source_size: tuple[int, int]
    model_size: tuple[int, int]
    mask_path: Path | None = None
    overlay_path: Path | None = None
    probabilities_path: Path | None = None
    schema_version: int = RESULT_SCHEMA_VERSION
    task_type: str = "segmentation"

    def to_dict(self, *, include_dense: bool = False) -> dict[str, Any]:
        """Return client metadata, optionally including dense arrays inline."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "input": self.input_name,
            "device": self.device,
            "labels": ["background", *self.labels],
            "colors": [list(color) for color in self.colors],
            "polygons": list(self.polygons),
            "dimensions": {
                "source": {
                    "width": self.source_size[0],
                    "height": self.source_size[1],
                },
                "model": {
                    "width": self.model_size[0],
                    "height": self.model_size[1],
                },
            },
            "artifacts": {
                "mask": str(self.mask_path) if self.mask_path else None,
                "overlay": str(self.overlay_path) if self.overlay_path else None,
                "probabilities": (
                    str(self.probabilities_path) if self.probabilities_path else None
                ),
            },
        }
        if include_dense:
            payload["mask"] = self.mask.tolist()
            payload["probabilities"] = self.probabilities.tolist()
        return payload

    def to_ml_dict(self) -> dict[str, Any]:
        """Return the legacy semantic-segmentation compatibility contract."""
        return {
            "data": {
                "colors": [list(color) for color in self.colors[1:]],
                "num_classes": len(self.labels),
                "polygons": list(self.polygons),
                "segments": [],
                "panoptic": None,
                "task_type": "segmentation",
            },
            "predictions": str(self.overlay_path) if self.overlay_path else "",
            "explainer": [],
            "logits": [self.probabilities.tolist()],
        }


InferenceResult = ClassificationResult | SegmentationResult
