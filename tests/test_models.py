"""Model-factory and exact checkpoint-loading tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import maea_inference.models as model_module
from maea_inference.errors import CheckpointError
from maea_inference.models import create_model, load_model
from tests.support import binary_config, segmentation_config


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def test_timm_factory_disables_pretrained_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create(name: str, **params: Any) -> DummyModel:
        captured.update({"name": name, **params})
        return DummyModel()

    monkeypatch.setattr(model_module.timm, "create_model", fake_create)
    create_model(binary_config())

    assert captured["name"] == "resnet18"
    assert captured["pretrained"] is False
    assert captured["num_classes"] == 1


def test_efficientnet_factory_uses_local_architecture_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_from_name(name: str, **params: Any) -> DummyModel:
        captured.update({"name": name, **params})
        return DummyModel()

    monkeypatch.setattr(model_module.EfficientNet, "from_name", fake_from_name)
    config = binary_config()
    config["architecture"]["module"] = "efficientnet:efficientnet-b0"
    create_model(config)

    assert captured["name"] == "efficientnet-b0"
    assert "pretrained" not in captured
    assert captured["num_classes"] == 1


@pytest.mark.parametrize(
    "model_name",
    [
        "unet",
        "unetplusplus",
        "deeplabv3",
        "deeplabv3plus",
        "fpn",
        "linknet",
        "pspnet",
        "pan",
    ],
)
def test_segmentation_factories_disable_encoder_weights(
    model_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_create(**params: Any) -> DummyModel:
        captured.update(params)
        return DummyModel()

    for attribute in (
        "Unet",
        "UnetPlusPlus",
        "DeepLabV3",
        "DeepLabV3Plus",
        "FPN",
        "Linknet",
        "PSPNet",
        "PAN",
    ):
        monkeypatch.setattr(model_module.smp, attribute, fake_create)

    create_model(segmentation_config(model_name))

    assert captured["encoder_weights"] is None
    assert captured["classes"] == 2


def test_exact_weight_loading_rejects_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_module, "create_model", lambda _config: DummyModel())

    with pytest.raises(CheckpointError, match="exactly match"):
        load_model(binary_config(), {"model_state_dict": {}}, torch.device("cpu"))


def test_raw_state_dictionary_loads_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "create_model", lambda _config: DummyModel())
    state_dict = {"weight": torch.ones(1)}

    loaded = load_model(binary_config(), state_dict, torch.device("cpu"))

    assert torch.equal(loaded.state_dict()["weight"], torch.ones(1))
