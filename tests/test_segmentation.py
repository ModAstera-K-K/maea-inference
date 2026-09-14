"""Semantic mask, polygon, and artifact tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from maea_inference.artifacts import save_segmentation_artifacts
from maea_inference.errors import ConfigurationError
from maea_inference.segmentation import (
    class_colors,
    extract_polygons,
    segmentation_arrays,
)


def _semantic_logits() -> torch.Tensor:
    logits = torch.zeros((1, 2, 16, 16), dtype=torch.float32)
    logits[:, 0, :, :] = 2.0
    logits[:, 1, 3:13, 4:14] = 5.0
    return logits


def test_semantic_softmax_mask_and_polygon_contract() -> None:
    mask, probabilities = segmentation_arrays(_semantic_logits(), expected_classes=2)
    polygons = extract_polygons(mask, ["foreground"])

    assert mask.shape == (16, 16)
    assert probabilities.shape == (2, 16, 16)
    assert np.allclose(probabilities.sum(axis=0), 1.0)
    assert mask[5, 5] == 1
    assert mask[0, 0] == 0
    assert polygons
    assert polygons[0]["label"] == "foreground"


def test_segmentation_channel_mismatch_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="expected 3"):
        segmentation_arrays(_semantic_logits(), expected_classes=3)


def test_segmentation_artifacts_preserve_mask_and_source_size(
    tmp_path: Path,
) -> None:
    mask, probabilities = segmentation_arrays(_semantic_logits(), expected_classes=2)
    source = Image.new("RGB", (40, 20), color=(200, 200, 200))

    mask_path, overlay_path, probabilities_path = save_segmentation_artifacts(
        output_dir=tmp_path,
        stem="sample_output",
        source_image=source,
        mask=mask,
        probabilities=probabilities,
        colors=class_colors(2),
        save_probabilities=True,
    )

    persisted_mask = np.asarray(Image.open(mask_path))
    assert persisted_mask.dtype in {np.dtype("uint16"), np.dtype("int32")}
    assert np.array_equal(persisted_mask, mask)
    assert Image.open(overlay_path).size == (40, 20)
    assert probabilities_path is not None
    archive = np.load(probabilities_path)
    assert np.allclose(archive["probabilities"], probabilities)
