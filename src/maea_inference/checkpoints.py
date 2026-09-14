"""Safe, local-only loading of MAEA PyTorch checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from maea_inference.errors import CheckpointError


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a trusted local checkpoint using PyTorch's restricted loader."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointError("Checkpoint file not found: %s" % checkpoint_path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except Exception as exc:
        raise CheckpointError(
            "Could not safely load checkpoint %s: %s. Only use trusted MAEA "
            "exports created with compatible PyTorch versions."
            % (checkpoint_path, str(exc))
        ) from exc
    if not isinstance(payload, Mapping):
        raise CheckpointError("Checkpoint root must be a mapping")
    return dict(payload)


def extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Return a canonical ``model_state_dict`` or a raw state dictionary."""
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, Mapping):
            raise CheckpointError("checkpoint.model_state_dict must be a mapping")
        return state_dict
    if checkpoint and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint.items()
    ):
        return checkpoint  # type: ignore[return-value]
    raise CheckpointError(
        "Checkpoint must contain model_state_dict or be a raw tensor state dictionary"
    )


def checkpoint_config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return embedded task configuration when present."""
    if "config" not in checkpoint:
        return None
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise CheckpointError("checkpoint.config must be an object when present")
    return config


def checkpoint_threshold_metadata(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return binary threshold metadata without changing prediction behavior."""
    metadata = checkpoint.get("threshold_metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}
