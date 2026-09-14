"""Reusable filesystem and payload helpers for the command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from maea_inference.artifacts import artifact_stem
from maea_inference.errors import PreprocessingError
from maea_inference.preprocessing import is_supported_path
from maea_inference.results import InferenceResult, SegmentationResult


def discover_inputs(input_path: Path, output_dir: Path) -> list[Path]:
    """Resolve a single file or recursively discover supported inputs."""
    resolved_input = input_path.expanduser().resolve()
    resolved_output = output_dir.expanduser().resolve()
    if resolved_input.is_file():
        if not is_supported_path(resolved_input):
            raise PreprocessingError(
                "Unsupported input file type: %s" % resolved_input.name
            )
        return [resolved_input]
    if not resolved_input.is_dir():
        raise PreprocessingError("Input path not found: %s" % resolved_input)
    paths = [
        path
        for path in resolved_input.rglob("*")
        if path.is_file()
        and is_supported_path(path)
        and not path.resolve().is_relative_to(resolved_output)
    ]
    return sorted(paths)


def result_payload(
    result: InferenceResult,
    *,
    ml_compatible: bool,
) -> dict[str, Any]:
    """Select the client-facing or legacy-compatible response view."""
    if ml_compatible:
        return result.to_ml_dict()
    if isinstance(result, SegmentationResult):
        return result.to_dict(include_dense=False)
    return result.to_dict()


def result_path(input_path: Path, output_dir: Path) -> Path:
    """Return the collision-safe per-input JSON destination."""
    stem = artifact_stem(input_path.name, str(input_path.resolve()))
    return output_dir.expanduser().resolve() / (stem + "_result.json")


def error_payload(input_path: Path, exc: Exception) -> dict[str, Any]:
    """Return a stable per-input error record."""
    return {
        "status": "error",
        "input": str(input_path),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }
