"""Public standalone MAEA imaging predictor."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from maea_inference.artifacts import artifact_stem, save_segmentation_artifacts
from maea_inference.checkpoints import (
    checkpoint_threshold_metadata,
    load_checkpoint,
)
from maea_inference.classification import (
    build_classification_result,
    target_config,
    target_labels,
)
from maea_inference.configuration import load_config, resolve_runtime_config
from maea_inference.devices import resolve_device
from maea_inference.errors import ConfigurationError
from maea_inference.models import load_model
from maea_inference.preprocessing import PreparedImage, prepare_bytes, prepare_path
from maea_inference.results import InferenceResult, SegmentationResult
from maea_inference.segmentation import (
    class_colors,
    extract_polygons,
    segmentation_arrays,
)

logger = logging.getLogger(__name__)


class MaeaPredictor:
    """Load one exported model once and run local imaging inference."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.config = dict(config)
        self.checkpoint = dict(checkpoint)
        self.model = model
        self.device = device
        self.threshold_metadata = checkpoint_threshold_metadata(checkpoint)

    @classmethod
    def from_artifacts(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: str = "auto",
    ) -> MaeaPredictor:
        """Load, validate, and prepare a checkpoint/config artifact pair."""
        resolved_device = resolve_device(device)
        supplied_config = load_config(config_path)
        checkpoint = load_checkpoint(checkpoint_path, resolved_device)
        config = resolve_runtime_config(
            supplied_config,
            checkpoint,
            checkpoint_path=checkpoint_path,
        )
        model = load_model(config, checkpoint, resolved_device)
        logger.info(
            "Loaded %s on %s", config["architecture"]["module"], resolved_device
        )
        return cls(
            config=config,
            checkpoint=checkpoint,
            model=model,
            device=resolved_device,
        )

    def predict(
        self,
        path: str | Path,
        output_dir: str | Path | None = None,
        *,
        save_probabilities: bool = False,
    ) -> InferenceResult:
        """Run inference for one local image path."""
        input_path = Path(path).expanduser().resolve()
        prepared = prepare_path(input_path, self.config, self.device)
        stem = artifact_stem(input_path.name, str(input_path))
        return self._predict_prepared(
            prepared,
            stem=stem,
            output_dir=output_dir,
            save_probabilities=save_probabilities,
        )

    def predict_bytes(
        self,
        contents: bytes,
        filename: str,
        output_dir: str | Path | None = None,
        *,
        save_probabilities: bool = False,
    ) -> InferenceResult:
        """Run inference for named in-memory image bytes."""
        prepared = prepare_bytes(contents, filename, self.config, self.device)
        digest = hashlib.sha256(contents).hexdigest()
        stem = artifact_stem(filename, "%s:%s" % (filename, digest))
        return self._predict_prepared(
            prepared,
            stem=stem,
            output_dir=output_dir,
            save_probabilities=save_probabilities,
        )

    def _predict_prepared(
        self,
        prepared: PreparedImage,
        *,
        stem: str,
        output_dir: str | Path | None,
        save_probabilities: bool,
    ) -> InferenceResult:
        with torch.inference_mode():
            output = self.model(prepared.tensor)
        if not isinstance(output, torch.Tensor):
            raise ConfigurationError(
                "Model inference must return a tensor for v0.1.0 imaging tasks"
            )
        task_type = str(target_config(self.config).get("type") or "")
        if task_type == "classification":
            return build_classification_result(
                output,
                self.config,
                input_name=prepared.filename,
                device=str(self.device),
                threshold_metadata=self.threshold_metadata,
            )
        if task_type == "segmentation":
            return self._segmentation_result(
                output,
                prepared,
                stem=stem,
                output_dir=output_dir,
                save_probabilities=save_probabilities,
            )
        raise ConfigurationError("Unsupported inference task: %s" % task_type)

    def _segmentation_result(
        self,
        output: torch.Tensor,
        prepared: PreparedImage,
        *,
        stem: str,
        output_dir: str | Path | None,
        save_probabilities: bool,
    ) -> SegmentationResult:
        labels = tuple(target_labels(self.config))
        mask, probabilities = segmentation_arrays(
            output,
            expected_classes=len(labels) + 1,
        )
        colors = class_colors(len(labels) + 1)
        polygons = extract_polygons(mask, labels)
        mask_path = None
        overlay_path = None
        probabilities_path = None
        if output_dir is not None:
            mask_path, overlay_path, probabilities_path = save_segmentation_artifacts(
                output_dir=output_dir,
                stem=stem,
                source_image=prepared.display_image,
                mask=mask,
                probabilities=probabilities,
                colors=colors,
                save_probabilities=save_probabilities,
            )
        return SegmentationResult(
            input_name=prepared.filename,
            device=str(self.device),
            labels=labels,
            mask=mask,
            probabilities=probabilities,
            colors=colors,
            polygons=polygons,
            source_size=prepared.source_size,
            model_size=(int(mask.shape[1]), int(mask.shape[0])),
            mask_path=mask_path,
            overlay_path=overlay_path,
            probabilities_path=probabilities_path,
        )
