"""CLI single-file and partial-directory behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import maea_inference.cli as cli_module
from maea_inference.classification import build_classification_result
from maea_inference.cli import main
from maea_inference.errors import PreprocessingError
from maea_inference.results import ClassificationResult
from tests.support import binary_config, png_bytes


class FakePredictor:
    def predict(
        self,
        path: Path,
        output_dir: Path,
        *,
        save_probabilities: bool,
    ) -> ClassificationResult:
        del output_dir, save_probabilities
        if path.name.startswith("bad"):
            raise PreprocessingError("broken image")
        import torch

        return build_classification_result(
            torch.tensor([[2.0]]),
            binary_config(),
            input_name=path.name,
            device="cpu",
        )


def _patch_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module.MaeaPredictor,
        "from_artifacts",
        lambda **_kwargs: FakePredictor(),
    )


def test_single_file_prints_and_writes_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_predictor(monkeypatch)
    image_path = tmp_path / "good.png"
    image_path.write_bytes(png_bytes())
    output_dir = tmp_path / "outputs"

    status = main(
        [
            "predict",
            "--checkpoint",
            str(tmp_path / "model.pth"),
            "--config",
            str(tmp_path / "config.json"),
            "--input",
            str(image_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["task_type"] == "classification"
    assert len(list(output_dir.glob("*_result.json"))) == 1


def test_directory_continues_and_exits_nonzero_on_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_predictor(monkeypatch)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "good.png").write_bytes(png_bytes())
    (input_dir / "bad.png").write_bytes(b"not an image")
    output_dir = tmp_path / "outputs"

    status = main(
        [
            "predict",
            "--checkpoint",
            str(tmp_path / "model.pth"),
            "--config",
            str(tmp_path / "config.json"),
            "--input",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 1
    records = [
        json.loads(line)
        for line in (output_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 2
    assert {record["status"] for record in records} == {"ok", "error"}
