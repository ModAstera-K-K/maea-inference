"""Minimal application integration example."""

from maea_inference import MaeaPredictor


def main() -> None:
    predictor = MaeaPredictor.from_artifacts(
        checkpoint_path="model.pth",
        config_path="config.json",
    )
    result = predictor.predict("image.png", output_dir="outputs")
    print(result.to_dict())


if __name__ == "__main__":
    main()
