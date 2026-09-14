"""Collision-safe local artifact persistence for CLI and Python callers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from maea_inference.errors import MaeaInferenceError
from maea_inference.segmentation import colored_mask


def artifact_stem(input_name: str, source_key: str | None = None) -> str:
    """Build a readable stable name that cannot collide across source paths."""
    lowered = input_name.lower()
    base_name = (
        input_name[:-7] if lowered.endswith(".nii.gz") else Path(input_name).stem
    )
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_name).strip("._")
    sanitized = sanitized[:96] or "input"
    digest_source = source_key or input_name
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:8]
    return "%s_%s" % (sanitized, digest)


def create_overlay(
    source_image: Image.Image,
    mask: np.ndarray,
    colors: Sequence[tuple[int, int, int]],
    *,
    alpha: float = 0.3,
) -> Image.Image:
    """Blend a nearest-neighbor resized color mask over the source image."""
    source = source_image.convert("RGB")
    rendered = Image.fromarray(colored_mask(mask, colors), mode="RGB")
    if rendered.size != source.size:
        rendered = rendered.resize(source.size, Image.Resampling.NEAREST)
    return Image.blend(source, rendered, alpha)


def save_segmentation_artifacts(
    *,
    output_dir: str | Path,
    stem: str,
    source_image: Image.Image,
    mask: np.ndarray,
    probabilities: np.ndarray,
    colors: Sequence[tuple[int, int, int]],
    save_probabilities: bool,
) -> tuple[Path, Path, Path | None]:
    """Persist a uint16 mask, source-sized overlay, and optional dense scores."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    maximum = int(mask.max()) if mask.size else 0
    if maximum > np.iinfo(np.uint16).max:
        raise MaeaInferenceError(
            "Segmentation class id %d exceeds uint16 mask capacity" % maximum
        )
    mask_path = destination / (stem + "_mask.png")
    overlay_path = destination / (stem + "_overlay.png")
    Image.fromarray(mask.astype(np.uint16)).save(mask_path)
    create_overlay(source_image, mask, colors).save(overlay_path)
    probabilities_path: Path | None = None
    if save_probabilities:
        probabilities_path = destination / (stem + "_probabilities.npz")
        np.savez_compressed(probabilities_path, probabilities=probabilities)
    return mask_path, overlay_path, probabilities_path
