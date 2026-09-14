# MAEA Inference

Standalone local inference for imaging models trained and exported by MAEA.
Version `0.1.0` supports image classification and classic semantic segmentation
without requiring MAEA platform runtime dependencies.

For copy-paste classification and segmentation workflows using placeholder
artifacts, see [Generic Integration Examples](examples/README.md).

## Supported artifacts

Every invocation requires two local files:

- a trusted PyTorch `.pth` checkpoint exported by MAEA; and
- the matching JSON task configuration.

The runtime accepts MAEA checkpoint dictionaries containing
`model_state_dict`, their embedded config and compatibility metadata, or a raw
tensor state dictionary paired with a complete config. Checkpoint weights are
loaded strictly. Model paths found inside the config are ignored in favor of
the explicit checkpoint argument.

The following model identifiers are supported:

| Task | Architecture modules |
|---|---|
| Classification | `timm:*`, `efficientnet:*` |
| Semantic segmentation | `segmentation:unet`, `unetplusplus`, `deeplabv3`, `deeplabv3plus`, `fpn`, `linknet`, `pspnet`, `pan` |

Detection, instance segmentation, panoptic segmentation, adaptive output-head
loading, and model conversion are intentionally rejected in this release.

## Install

Create a Python 3.11 environment and install the repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Install optional DICOM and NIfTI readers with:

```bash
python -m pip install '.[medical]'
```

Install a compatible CUDA build of PyTorch before this package when GPU
inference is required.

## Python API

```python
from maea_inference import MaeaPredictor

predictor = MaeaPredictor.from_artifacts(
    checkpoint_path="model.pth",
    config_path="config.json",
    device="auto",
)

result = predictor.predict("image.png", output_dir="outputs")
print(result.to_dict())

# Existing integrations can consume the legacy compatibility view.
print(result.to_ml_dict())
```

Use `predict_bytes(contents, filename, output_dir=...)` when an application
already owns the input bytes. The filename is required because it selects the
decoder.

## CLI

Run one image:

```bash
maea-infer predict \
  --checkpoint model.pth \
  --config config.json \
  --input image.png \
  --output-dir outputs
```

The client result is printed to stdout and also saved in `outputs`. Logs and
errors use stderr, making stdout safe to pipe into another program.

Run all supported files below a directory:

```bash
maea-infer predict \
  --checkpoint model.pth \
  --config config.json \
  --input images/ \
  --output-dir outputs/ \
  --save-probabilities
```

Directory mode continues after individual input errors, writes one result JSON
per input plus `results.jsonl`, and exits with status `1` if any item failed.
Use `--ml-compatible` to emit the legacy compatibility dictionary instead of
the versioned client view. For segmentation this includes the dense
probability array and can create a large JSON file.

## Input behavior

| Input | Extensions | Installation | v0.1.0 behavior |
|---|---|---|---|
| Raster | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tif`, `.tiff` | Base package | RGB conversion; first GIF frame |
| DICOM | `.dcm`, `.dicom` | `maea-inference[medical]` | Single-frame 2D; min/max conversion to 8-bit RGB |
| NIfTI | `.nii`, `.nii.gz` | `maea-inference[medical]` | 2D arrays only |

Raster inputs are converted to RGB, converted to a tensor, and resized using
the MAEA export contract precedence:

1. `architecture.input_size`
2. `architecture.params.input_size`
3. `training.image_size`
4. `training.target.image_size`
5. `[224, 224]`

There is no additional ImageNet normalization unless it is already part of the
model itself, matching the MAEA export preprocessing contract.

The `medical` extra adds:

- single-frame 2D DICOM, normalized from its pixel range to 8-bit RGB; and
- 2D `.nii` and `.nii.gz` NIfTI arrays.

Multi-frame DICOM and 3D/4D NIfTI are rejected because MAEA does not yet have a
stable volumetric inference contract.

## Result contracts

Classification returns schema version, layout, ordered target decisions,
numeric `class_id`, the selected raw `class_value`, all available
`class_values`, probabilities, and checkpoint threshold metadata. Binary
single-logit models retain the current `0.5` decision threshold even when the
checkpoint records a separately optimized evaluation threshold. Flat and
structured multiclass models use argmax.

Semantic segmentation returns an in-memory model-resolution class mask and
probability tensor. Class `0` is background and configured labels map to
classes `1..N`. With an output directory it writes:

- `<input>_<hash>_mask.png`: lossless uint16 model-resolution class IDs;
- `<input>_<hash>_overlay.png`: source-sized RGB overlay; and
- `<input>_<hash>_probabilities.npz`: `[classes, height, width]` probabilities
  when `--save-probabilities` is requested.

The client JSON references dense artifacts instead of embedding them. Python
callers can access `result.mask` and `result.probabilities` directly.

## Docker

Build the reproducible CPU image:

```bash
docker build -t maea-inference:0.1.0 .
```

Run with artifacts mounted read-only and results mounted separately:

```bash
mkdir -p outputs
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/work,readonly \
  --mount type=bind,src="$PWD/outputs",dst=/outputs \
  maea-inference:0.1.0 predict \
  --checkpoint /work/model.pth \
  --config /work/config.json \
  --input /work/image.png \
  --output-dir /outputs \
  --device cpu
```

## Security and offline use

PyTorch checkpoints are serialized artifacts. Only load exports from a trusted
MAEA source or another trusted producer. The runtime uses PyTorch's restricted
`weights_only` loader and never falls back to unrestricted pickle loading.

After dependencies are installed, inference is local-only: it does not fetch
URLs, contact MAEA services, or download pretrained weights. Config/checkpoint
identity mismatches and partial state-dictionary loads are fatal.

## Development

```bash
python -m pip install '.[dev,medical]'
ruff check .
mypy src tests
pytest
```
