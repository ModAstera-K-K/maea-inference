"""Shared inference constants."""

from __future__ import annotations

from typing import Final

RESULT_SCHEMA_VERSION: Final = 1

RASTER_SUFFIXES: Final = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
)
MEDICAL_SUFFIXES: Final = frozenset({".dcm", ".dicom", ".nii", ".nii.gz"})
SUPPORTED_SUFFIXES: Final = RASTER_SUFFIXES | MEDICAL_SUFFIXES

SEGMENTATION_MODELS: Final = frozenset(
    {
        "unet",
        "unetplusplus",
        "deeplabv3",
        "deeplabv3plus",
        "fpn",
        "linknet",
        "pspnet",
        "pan",
    }
)

CLASS_COLORS: Final = (
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 128, 128),
    (255, 128, 0),
    (128, 0, 255),
)
