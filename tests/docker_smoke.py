"""Self-contained CPU inference smoke used by the built Docker image."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image

from maea_inference.classification import build_classification_contract
from maea_inference.models import create_model


def _config() -> dict[str, object]:
    return {
        "architecture": {
            "module": "timm:mobilenetv3_small_050",
            "input_size": [32, 32],
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


def _write_artifacts(directory: Path) -> tuple[Path, Path, Path]:
    config = _config()
    config_path = directory / "config.json"
    checkpoint_path = directory / "model.pth"
    input_path = directory / "input.png"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    model = create_model(config)
    contract = build_classification_contract(config)
    assert contract is not None
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "classification_contract": contract.to_dict(),
        },
        checkpoint_path,
    )
    Image.new("RGB", (40, 24), color=(100, 50, 25)).save(input_path)
    return checkpoint_path, config_path, input_path


def main() -> None:
    """Create local artifacts and require a successful installed CLI prediction."""
    with tempfile.TemporaryDirectory(prefix="maea-docker-smoke-") as temporary:
        root = Path(temporary)
        checkpoint_path, config_path, input_path = _write_artifacts(root)
        completed = subprocess.run(
            [
                "maea-infer",
                "predict",
                "--checkpoint",
                str(checkpoint_path),
                "--config",
                str(config_path),
                "--input",
                str(input_path),
                "--output-dir",
                str(root / "outputs"),
                "--device",
                "cpu",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["schema_version"] == 1
        assert payload["task_type"] == "classification"
        assert len(payload["predictions"][0]["probabilities"]) == 2


if __name__ == "__main__":
    main()
