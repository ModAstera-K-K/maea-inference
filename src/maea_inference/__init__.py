"""Standalone inference for MAEA imaging models."""

from maea_inference.predictor import MaeaPredictor
from maea_inference.results import ClassificationResult, SegmentationResult

__all__ = [
    "ClassificationResult",
    "MaeaPredictor",
    "SegmentationResult",
]

__version__ = "0.1.0"
