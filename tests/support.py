"""Reusable test data builders."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image


def binary_config() -> dict[str, Any]:
    return {
        "architecture": {
            "module": "timm:resnet18",
            "input_size": [16, 16],
            "params": {"num_classes": 1, "pretrained": True},
        },
        "criterion": {"module": "nn.BCEWithLogitsLoss", "params": {}},
        "dataset": {"format": "image", "max_num_classes": 2},
        "training": {
            "target": {
                "type": "classification",
                "labels": ["example_label"],
                "class_count": 2,
            }
        },
    }


def flat_config() -> dict[str, Any]:
    config = binary_config()
    config["criterion"] = {"module": "nn.CrossEntropyLoss", "params": {}}
    config["dataset"]["max_num_classes"] = 3
    config["training"]["target"] = {
        "type": "classification",
        "labels": ["cat", "dog", "bird"],
        "class_count": 3,
        "classification_output_layout_version": "flat_multiclass_v2",
        "target_encoding": "one_hot_multiclass",
    }
    return config


def structured_config() -> dict[str, Any]:
    config = binary_config()
    config["criterion"] = {"module": "nn.CrossEntropyLoss", "params": {}}
    config["architecture"]["params"]["num_classes"] = 6
    config["dataset"]["max_num_classes"] = 3
    config["training"]["target"] = {
        "type": "classification",
        "labels": ["grade", "shape"],
        "class_count": 3,
        "classification_output_layout_version": "legacy_structured_v1",
        "vocabularies": {
            "grade": ["low", "medium", "high"],
            "shape": ["round", "oval", "irregular"],
        },
    }
    return config


def segmentation_config(model_name: str = "unet") -> dict[str, Any]:
    return {
        "architecture": {
            "module": "segmentation:%s" % model_name,
            "input_size": [16, 16],
            "params": {
                "encoder_name": "resnet18",
                "encoder_weights": "imagenet",
                "in_channels": 3,
                "num_classes": 1,
                "activation": None,
            },
        },
        "dataset": {"format": "segmentation"},
        "training": {
            "target": {
                "type": "segmentation",
                "labels": ["foreground"],
            }
        },
    }


def copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config)


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config), encoding="utf-8")


def png_bytes(
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (20, 10),
    color: Any = (120, 30, 10),
) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()
