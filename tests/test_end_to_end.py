"""Real model construction, strict checkpoint reload, and inference smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch

from maea_inference.classification import build_classification_contract
from maea_inference.models import create_model
from maea_inference.predictor import MaeaPredictor
from maea_inference.results import ClassificationResult, SegmentationResult
from tests.support import binary_config, png_bytes, segmentation_config, write_config


def _save_export_checkpoint(
    path: Path,
    config: dict[str, object],
    model: torch.nn.Module,
) -> None:
    payload = {
        "epoch": 1,
        "metric_value": 0.5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "config": config,
        "threshold_metadata": {"selected_threshold": 0.37},
        "checkpoint_reason": "test",
    }
    contract = build_classification_contract(config)
    if contract is not None:
        payload["classification_contract"] = contract.to_dict()
    torch.save(payload, path)


def test_real_timm_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    config = binary_config()
    config_path = tmp_path / "classification.json"
    checkpoint_path = tmp_path / "classification.pth"
    input_path = tmp_path / "sample.png"
    write_config(config_path, config)
    input_path.write_bytes(png_bytes())
    _save_export_checkpoint(checkpoint_path, config, create_model(config))

    predictor = MaeaPredictor.from_artifacts(
        checkpoint_path,
        config_path,
        device="cpu",
    )
    result = predictor.predict(input_path)

    assert isinstance(result, ClassificationResult)
    assert len(result.predictions) == 1
    assert len(result.predictions[0].probabilities) == 2


def test_real_unet_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(11)
    config = segmentation_config()
    config["architecture"]["input_size"] = [32, 32]
    config_path = tmp_path / "segmentation.json"
    checkpoint_path = tmp_path / "segmentation.pth"
    input_path = tmp_path / "sample.png"
    output_dir = tmp_path / "outputs"
    write_config(config_path, config)
    input_path.write_bytes(png_bytes(size=(40, 30)))
    _save_export_checkpoint(checkpoint_path, config, create_model(config))

    predictor = MaeaPredictor.from_artifacts(
        checkpoint_path,
        config_path,
        device="cpu",
    )
    result = predictor.predict(input_path, output_dir)

    assert isinstance(result, SegmentationResult)
    assert result.mask.shape == (32, 32)
    assert result.probabilities.shape == (2, 32, 32)
    assert result.mask_path is not None and result.mask_path.is_file()
    assert result.overlay_path is not None and result.overlay_path.is_file()
