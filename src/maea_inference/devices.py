"""Inference-device resolution."""

from __future__ import annotations

import torch

from maea_inference.errors import ConfigurationError


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` using the MAEA export device policy."""
    normalized = str(requested or "auto").strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ConfigurationError(
                "CUDA was requested but is not available in this PyTorch runtime"
            )
        return torch.device(normalized)
    raise ConfigurationError(
        "Unsupported device %s; choose auto, cpu, cuda, or cuda:N" % requested
    )
