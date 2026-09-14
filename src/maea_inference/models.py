"""Standalone factories for model families exported by MAEA."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

import segmentation_models_pytorch as smp
import timm
import torch
from efficientnet_pytorch import EfficientNet

from maea_inference.checkpoints import extract_state_dict
from maea_inference.classification import classification_head_width, target_config
from maea_inference.constants import SEGMENTATION_MODELS
from maea_inference.errors import CheckpointError, UnsupportedModelError


def _architecture(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("architecture")
    return value if isinstance(value, Mapping) else {}


def _architecture_parts(config: Mapping[str, Any]) -> tuple[str, str]:
    architecture = _architecture(config)
    module = str(architecture.get("module") or architecture.get("type") or "")
    if ":" not in module:
        raise UnsupportedModelError("Invalid architecture module: %s" % module)
    source, name = module.split(":", 1)
    return source, name


def _classification_model(config: Mapping[str, Any]) -> torch.nn.Module:
    source, name = _architecture_parts(config)
    params = deepcopy(dict(_architecture(config).get("params") or {}))
    params["num_classes"] = classification_head_width(config)
    if source == "efficientnet":
        params.pop("pretrained", None)
        try:
            return cast(torch.nn.Module, EfficientNet.from_name(name, **params))
        except Exception as exc:
            raise UnsupportedModelError(
                "Could not create EfficientNet model %s: %s" % (name, str(exc))
            ) from exc
    if source == "timm":
        params["pretrained"] = False
        try:
            return cast(torch.nn.Module, timm.create_model(name, **params))
        except Exception as exc:
            raise UnsupportedModelError(
                "Could not create timm model %s: %s" % (name, str(exc))
            ) from exc
    raise UnsupportedModelError("Unsupported classification model source: %s" % source)


def _segmentation_model(config: Mapping[str, Any]) -> torch.nn.Module:
    source, name = _architecture_parts(config)
    normalized = name.lower()
    if source != "segmentation" or normalized not in SEGMENTATION_MODELS:
        raise UnsupportedModelError(
            "Unsupported semantic segmentation model: %s:%s" % (source, name)
        )
    params = deepcopy(dict(_architecture(config).get("params") or {}))
    encoder_name = params.pop("encoder_name", "resnet34")
    params.pop("encoder_weights", None)
    in_channels = int(params.pop("in_channels", 3))
    classes = int(params.pop("num_classes")) + 1 if "num_classes" in params else 1
    activation = params.pop("activation", None)
    model_classes: dict[str, type[torch.nn.Module]] = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "linknet": smp.Linknet,
        "pspnet": smp.PSPNet,
        "pan": smp.PAN,
    }
    model_class = model_classes[normalized]
    try:
        return model_class(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **params,
        )
    except Exception as exc:
        raise UnsupportedModelError(
            "Could not create segmentation model %s: %s" % (name, str(exc))
        ) from exc


def create_model(config: Mapping[str, Any]) -> torch.nn.Module:
    """Create a checkpoint-ready model without downloading base weights."""
    task_type = str(target_config(config).get("type") or "").strip()
    if task_type == "classification":
        return _classification_model(config)
    if task_type == "segmentation":
        return _segmentation_model(config)
    raise UnsupportedModelError("Unsupported model task: %s" % task_type)


def load_model(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Build the configured model and require an exact checkpoint load."""
    model = create_model(config)
    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise CheckpointError(
            "Checkpoint weights do not exactly match the configured model: %s"
            % str(exc)
        ) from exc
    model = model.to(device)
    model.eval()
    return model
