# Generic Integration Examples

These examples demonstrate classification and semantic-segmentation inference
with placeholder paths. Replace each placeholder with the checkpoint, matching
configuration, and input image supplied for your model.

No trained models, real input images, deployment identifiers, or expected
predictions are included in this repository. Model outputs depend on the
artifacts and inputs you provide.

## Arrange local artifacts

Keep model files outside the repository when possible:

```text
local-artifacts/
├── classification/
│   ├── model.pth
│   ├── config.json
│   └── sample.png
└── segmentation/
    ├── model.pth
    ├── config.json
    └── sample.png
```

The checkpoint and configuration must come from the same exported model
version. Do not edit architecture, label, class-count, or output-layout values
to force a mismatched checkpoint to load.

## Classification CLI

```bash
maea-infer predict \
  --checkpoint local-artifacts/classification/model.pth \
  --config local-artifacts/classification/config.json \
  --input local-artifacts/classification/sample.png \
  --output-dir local-outputs/classification \
  --device cpu
```

The command prints a schema-v1 result and writes a JSON result below the output
directory. Each classification decision includes its target, selected class,
available class values, and normalized probabilities.

Use `--ml-compatible` only when adapting an existing integration that expects
the legacy compatibility dictionary. The compatibility field named `logits`
contains probabilities after sigmoid or softmax has been applied.

## Segmentation CLI

```bash
maea-infer predict \
  --checkpoint local-artifacts/segmentation/model.pth \
  --config local-artifacts/segmentation/config.json \
  --input local-artifacts/segmentation/sample.png \
  --output-dir local-outputs/segmentation \
  --device cpu \
  --save-probabilities
```

Segmentation writes a lossless uint16 class-index mask, a source-sized RGB
overlay, a schema-v1 JSON result, and—when requested—a compressed NumPy
probability archive. Class `0` is background; configured foreground labels map
to classes `1..N`.

## Python API

Create a predictor once and reuse it for multiple requests:

```python
from pathlib import Path

from maea_inference import ClassificationResult, MaeaPredictor

artifact_dir = Path("local-artifacts/classification")
predictor = MaeaPredictor.from_artifacts(
    checkpoint_path=artifact_dir / "model.pth",
    config_path=artifact_dir / "config.json",
    device="auto",
)

result = predictor.predict(
    artifact_dir / "sample.png",
    output_dir="local-outputs/classification",
)
if not isinstance(result, ClassificationResult):
    raise TypeError("Expected a classification result")

for prediction in result.predictions:
    print(
        prediction.target,
        prediction.class_id,
        prediction.class_value,
        prediction.probabilities,
    )
```

For application-owned bytes:

```python
contents = (artifact_dir / "sample.png").read_bytes()
result = predictor.predict_bytes(
    contents,
    filename="sample.png",
    output_dir="local-outputs/classification",
)
```

Use `result.to_dict()` for the versioned client schema or
`result.to_ml_dict()` for the legacy compatibility schema. The segmentation API
uses the same predictor methods and returns a `SegmentationResult`.

## Docker

Build the CPU image:

```bash
docker build -t maea-inference:0.1.0 .
mkdir -p local-outputs
```

Run with artifacts mounted read-only and networking disabled:

```bash
docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD/local-artifacts",dst=/artifacts,readonly \
  --mount type=bind,src="$PWD/local-outputs",dst=/outputs \
  maea-inference:0.1.0 predict \
  --checkpoint /artifacts/classification/model.pth \
  --config /artifacts/classification/config.json \
  --input /artifacts/classification/sample.png \
  --output-dir /outputs/classification \
  --device cpu
```

## Integration checks

Before integrating an export, confirm that:

1. The checkpoint and configuration came from the same model version.
2. The predictor loads without a contract or exact-key error.
3. Classification probabilities are normalized for each decision.
4. Segmentation masks contain only class IDs declared by the result labels.
5. Production inference runs with network access disabled after installation.

Only load checkpoints from a trusted source. PyTorch checkpoints are serialized
artifacts and must be treated like executable software.
