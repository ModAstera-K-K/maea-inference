"""Semantic-segmentation result processing for MAEA exports."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch

from maea_inference.constants import CLASS_COLORS
from maea_inference.errors import ConfigurationError


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Return the deterministic color used by the MAEA export contract."""
    if class_id < len(CLASS_COLORS):
        return CLASS_COLORS[class_id]
    hue = (class_id * 137.5) % 360
    saturation = 0.7 + (class_id % 3) * 0.1
    value = 0.8 + (class_id % 2) * 0.2
    red, green, blue = colorsys.hsv_to_rgb(hue / 360, saturation, value)
    return int(red * 255), int(green * 255), int(blue * 255)


def class_colors(class_count: int) -> tuple[tuple[int, int, int], ...]:
    """Return colors including class zero's black background entry."""
    return tuple(color_for_class(class_id) for class_id in range(class_count))


def colored_mask(
    mask: np.ndarray,
    colors: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Create the RGB foreground mask used for MAEA overlays."""
    if mask.ndim != 2:
        raise ConfigurationError("Expected a 2D segmentation mask")
    rendered = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(1, len(colors)):
        rendered[mask == class_id] = colors[class_id]
    return rendered


def extract_polygons(
    mask: np.ndarray,
    labels: Sequence[str],
    *,
    minimum_area: float = 50.0,
) -> tuple[dict[str, Any], ...]:
    """Extract foreground contours using the current ML polygon contract."""
    height, width = mask.shape
    polygons: list[dict[str, Any]] = []
    for class_id, label in enumerate(labels, start=1):
        binary_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area or len(contour) < 3:
                continue
            epsilon = 0.0001 * cv2.arcLength(contour, True)
            approximate = cv2.approxPolyDP(contour, epsilon, True)
            points: list[float] = []
            for point in approximate:
                x_coordinate = int(point[0][0])
                y_coordinate = int(point[0][1])
                points.extend(
                    [
                        round((x_coordinate / width) * 100, 2),
                        round((y_coordinate / height) * 100, 2),
                    ]
                )
            if len(points) >= 6:
                polygons.append({"label": label, "points": points})
    return tuple(polygons)


def segmentation_arrays(
    output: torch.Tensor,
    *,
    expected_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply softmax and argmax to one dense semantic model output."""
    if output.dim() != 4 or output.shape[0] != 1:
        raise ConfigurationError(
            "Semantic segmentation output must have shape [1, classes, height, width]"
        )
    if output.shape[1] != expected_classes:
        raise ConfigurationError(
            "Semantic segmentation output has %d classes; expected %d"
            % (output.shape[1], expected_classes)
        )
    probabilities = torch.softmax(output, dim=1)
    mask = torch.argmax(probabilities, dim=1).squeeze(0)
    return (
        mask.detach().cpu().numpy(),
        probabilities.squeeze(0).detach().cpu().numpy(),
    )
