"""Raster, DICOM, and NIfTI preprocessing tests."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from maea_inference.errors import PreprocessingError
from maea_inference.preprocessing import input_suffix, prepare_bytes
from tests.support import binary_config, png_bytes


def test_raster_is_rgb_then_tensor_resized() -> None:
    prepared = prepare_bytes(
        png_bytes(mode="L", color=127),
        "sample.png",
        binary_config(),
        torch.device("cpu"),
    )

    assert prepared.display_image.mode == "RGB"
    assert prepared.source_size == (20, 10)
    assert prepared.tensor.shape == (1, 3, 16, 16)
    assert prepared.tensor.dtype == torch.float32


@pytest.mark.parametrize(
    ("image_format", "filename"),
    [("JPEG", "sample.jpg"), ("BMP", "sample.bmp"), ("TIFF", "sample.tiff")],
)
def test_supported_raster_formats_are_decoded(
    image_format: str,
    filename: str,
) -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (12, 7), color=(10, 20, 30, 255)).convert("RGB").save(
        buffer,
        format=image_format,
    )

    prepared = prepare_bytes(
        buffer.getvalue(),
        filename,
        binary_config(),
        torch.device("cpu"),
    )

    assert prepared.display_image.mode == "RGB"
    assert prepared.source_size == (12, 7)


def test_gif_uses_first_frame() -> None:
    buffer = io.BytesIO()
    first = Image.new("RGB", (6, 5), color=(255, 0, 0))
    second = Image.new("RGB", (6, 5), color=(0, 255, 0))
    first.save(buffer, format="GIF", save_all=True, append_images=[second])

    prepared = prepare_bytes(
        buffer.getvalue(),
        "sample.gif",
        binary_config(),
        torch.device("cpu"),
    )

    assert prepared.display_image.getpixel((0, 0)) == (255, 0, 0)


def test_compound_nifti_suffix_is_detected() -> None:
    assert input_suffix("volume.NII.GZ") == ".nii.gz"


def _dicom_bytes(values: np.ndarray) -> bytes:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    metadata = FileMetaDataset()
    metadata.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    metadata.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset("", {}, file_meta=metadata, preamble=b"\0" * 128)
    dataset.SOPClassUID = metadata.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID
    if values.ndim == 3:
        dataset.NumberOfFrames = values.shape[0]
        dataset.Rows, dataset.Columns = values.shape[1:]
    else:
        dataset.Rows, dataset.Columns = values.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = values.astype(np.uint16).tobytes()
    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, dataset, enforce_file_format=True)
    return buffer.getvalue()


def test_constant_dicom_is_guarded_against_zero_range() -> None:
    prepared = prepare_bytes(
        _dicom_bytes(np.full((8, 8), 7, dtype=np.uint16)),
        "scan.dcm",
        binary_config(),
        torch.device("cpu"),
    )

    assert prepared.tensor.shape == (1, 3, 16, 16)
    assert torch.count_nonzero(prepared.tensor) == 0


def test_multiframe_dicom_is_rejected() -> None:
    contents = _dicom_bytes(np.ones((2, 8, 8), dtype=np.uint16))

    with pytest.raises(PreprocessingError, match="single-frame 2D DICOM"):
        prepare_bytes(
            contents,
            "scan.dcm",
            binary_config(),
            torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("package", "filename"),
    [("pydicom", "scan.dcm"), ("nibabel", "scan.nii")],
)
def test_missing_medical_extra_has_actionable_error(
    package: str,
    filename: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, package, None)

    with pytest.raises(PreprocessingError, match=r"maea-inference\[medical\]"):
        prepare_bytes(
            b"not decoded before dependency check",
            filename,
            binary_config(),
            torch.device("cpu"),
        )


def _nifti_bytes(tmp_path: Path, values: np.ndarray, suffix: str) -> bytes:
    nib = pytest.importorskip("nibabel")
    destination = tmp_path / ("volume" + suffix)
    nib.save(nib.Nifti1Image(values, np.eye(4)), destination)
    return destination.read_bytes()


def test_2d_nifti_is_supported(tmp_path: Path) -> None:
    prepared = prepare_bytes(
        _nifti_bytes(tmp_path, np.ones((8, 9), dtype=np.float32), ".nii.gz"),
        "volume.nii.gz",
        binary_config(),
        torch.device("cpu"),
    )

    assert prepared.source_size == (9, 8)
    assert prepared.tensor.shape == (1, 1, 16, 16)


def test_volumetric_nifti_is_rejected(tmp_path: Path) -> None:
    contents = _nifti_bytes(
        tmp_path,
        np.ones((8, 9, 3), dtype=np.float32),
        ".nii",
    )

    with pytest.raises(PreprocessingError, match="only 2D NIfTI"):
        prepare_bytes(
            contents,
            "volume.nii",
            binary_config(),
            torch.device("cpu"),
        )
