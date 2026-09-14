"""Public predictor API tests with deterministic local artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import maea_inference.predictor as predictor_module
from maea_inference.classification import build_classification_contract
from maea_inference.predictor import MaeaPredictor
from maea_inference.results import ClassificationResult, SegmentationResult
from tests.support import binary_config, png_bytes, segmentation_config, write_config


class ConstantClassifier(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.zeros((values.shape[0], 1), device=values.device)


class ConstantSegmenter(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            (values.shape[0], 2, values.shape[2], values.shape[3]),
            device=values.device,
        )
        output[:, 1, 2:14, 2:14] = 4.0
        return output


def _write_checkpoint(path: Path, config: dict[str, object]) -> None:
    contract = build_classification_contract(config)
    payload = {
        "model_state_dict": {},
        "config": config,
        "threshold_metadata": {"selected_threshold": 0.8},
    }
    if contract is not None:
        payload["classification_contract"] = contract.to_dict()
    torch.save(payload, path)


def test_from_artifacts_and_predict_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = binary_config()
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "model.pth"
    write_config(config_path, config)
    _write_checkpoint(checkpoint_path, config)
    monkeypatch.setattr(
        predictor_module,
        "load_model",
        lambda _config, _checkpoint, _device: ConstantClassifier(),
    )

    predictor = MaeaPredictor.from_artifacts(checkpoint_path, config_path)
    result = predictor.predict_bytes(png_bytes(), "sample.png")

    assert isinstance(result, ClassificationResult)
    assert result.predictions[0].class_id == 1
    assert result.threshold_metadata["selected_threshold"] == 0.8


def test_segmentation_predict_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = segmentation_config()
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "model.pth"
    output_dir = tmp_path / "outputs"
    write_config(config_path, config)
    _write_checkpoint(checkpoint_path, config)
    monkeypatch.setattr(
        predictor_module,
        "load_model",
        lambda _config, _checkpoint, _device: ConstantSegmenter(),
    )

    predictor = MaeaPredictor.from_artifacts(checkpoint_path, config_path)
    result = predictor.predict_bytes(
        png_bytes(size=(32, 24)),
        "sample.png",
        output_dir,
        save_probabilities=True,
    )

    assert isinstance(result, SegmentationResult)
    assert result.mask.shape == (16, 16)
    assert result.mask_path is not None and result.mask_path.is_file()
    assert result.overlay_path is not None and result.overlay_path.is_file()
    assert result.probabilities_path is not None
    assert result.to_ml_dict()["data"]["task_type"] == "segmentation"
