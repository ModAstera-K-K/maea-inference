"""Local raster and optional medical-image preprocessing."""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from maea_inference.configuration import resolve_image_size
from maea_inference.constants import MEDICAL_SUFFIXES, RASTER_SUFFIXES
from maea_inference.errors import PreprocessingError


@dataclass(frozen=True)
class PreparedImage:
    """Tensor input plus the source image needed for client artifacts."""

    tensor: torch.Tensor
    display_image: Image.Image
    filename: str
    source_size: tuple[int, int]


def input_suffix(filename: str) -> str:
    """Resolve compound medical suffixes before ordinary path suffixes."""
    lowered = filename.lower()
    if lowered.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lowered).suffix


def is_supported_path(path: Path) -> bool:
    """Return whether a path has a supported raster or medical suffix."""
    return input_suffix(path.name) in RASTER_SUFFIXES | MEDICAL_SUFFIXES


def _missing_medical_dependency(package: str) -> PreprocessingError:
    return PreprocessingError(
        "Medical input requires %s; install maea-inference[medical]" % package
    )


def _raster_image(contents: bytes) -> Image.Image:
    with Image.open(io.BytesIO(contents)) as opened:
        opened.seek(0)
        image = cast(Image.Image, opened.convert("RGB"))
        image.load()
    return image


def _normalize_uint8(values: np.ndarray) -> np.ndarray:
    pixels = values.astype(np.float64)
    minimum = float(pixels.min())
    extent = float(pixels.max() - minimum)
    if extent == 0:
        return np.zeros_like(pixels, dtype=np.uint8)
    return ((pixels - minimum) / extent * 255).astype(np.uint8)


def _dicom_image(contents: bytes) -> Image.Image:
    try:
        import pydicom
    except ImportError as exc:
        raise _missing_medical_dependency("pydicom") from exc
    try:
        dataset = pydicom.dcmread(io.BytesIO(contents))
        pixels = np.asarray(dataset.pixel_array)
    except Exception as exc:
        raise PreprocessingError("Could not decode DICOM input: %s" % str(exc)) from exc
    if pixels.ndim != 2:
        raise PreprocessingError(
            "v0.1.0 supports only single-frame 2D DICOM; received shape %s"
            % (pixels.shape,)
        )
    return Image.fromarray(_normalize_uint8(pixels)).convert("RGB")


def _load_nifti(contents: bytes, suffix: str) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise _missing_medical_dependency("nibabel") from exc
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(contents)
            temporary_path = temporary.name
        loaded = cast(Any, nib.load(temporary_path))
        return np.asarray(loaded.get_fdata())
    except Exception as exc:
        raise PreprocessingError("Could not decode NIfTI input: %s" % str(exc)) from exc
    finally:
        if temporary_path:
            with suppress(OSError):
                os.unlink(temporary_path)


def _nifti_images(contents: bytes, suffix: str) -> tuple[np.ndarray, Image.Image]:
    data = _load_nifti(contents, suffix)
    if data.ndim != 2:
        raise PreprocessingError(
            "v0.1.0 supports only 2D NIfTI; received shape %s" % (data.shape,)
        )
    display = Image.fromarray(_normalize_uint8(data)).convert("RGB")
    return data, display


def _apply_transform(
    image: Image.Image | np.ndarray, config: Mapping[str, Any]
) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolve_image_size(config)),
        ]
    )
    tensor = cast(torch.Tensor, transform(image)).float()
    if tensor.dim() != 3:
        raise PreprocessingError(
            "Preprocessed image must have shape [channels, height, width]"
        )
    return tensor.unsqueeze(0)


def prepare_bytes(
    contents: bytes,
    filename: str,
    config: Mapping[str, Any],
    device: torch.device,
) -> PreparedImage:
    """Preprocess named bytes with the current MAEA image rules."""
    if not filename:
        raise PreprocessingError("A filename is required when predicting from bytes")
    suffix = input_suffix(filename)
    try:
        if suffix in RASTER_SUFFIXES:
            display_image = _raster_image(contents)
            transform_input: Image.Image | np.ndarray = display_image
        elif suffix in {".dcm", ".dicom"}:
            display_image = _dicom_image(contents)
            transform_input = display_image
        elif suffix in {".nii", ".nii.gz"}:
            transform_input, display_image = _nifti_images(contents, suffix)
        else:
            raise PreprocessingError("Unsupported file type: %s" % suffix)
        tensor = _apply_transform(transform_input, config).to(device)
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(
            "Failed to preprocess %s: %s" % (filename, str(exc))
        ) from exc
    return PreparedImage(
        tensor=tensor,
        display_image=display_image,
        filename=filename,
        source_size=display_image.size,
    )


def prepare_path(
    path: str | Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> PreparedImage:
    """Read and preprocess one local input path."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise PreprocessingError("Input file not found: %s" % input_path)
    try:
        contents = input_path.read_bytes()
    except OSError as exc:
        raise PreprocessingError(
            "Could not read input %s: %s" % (input_path, str(exc))
        ) from exc
    return prepare_bytes(contents, input_path.name, config, device)
