"""Command-line interface for standalone MAEA imaging inference."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from maea_inference.cli_support import (
    discover_inputs,
    error_payload,
    result_path,
    result_payload,
)
from maea_inference.errors import MaeaInferenceError
from maea_inference.predictor import MaeaPredictor
from maea_inference.serialization import json_text, write_json, write_jsonl

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maea-infer",
        description="Run exported MAEA imaging models locally.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logs on stderr.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict", help="Run imaging inference.")
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--config", required=True, type=Path)
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output-dir", required=True, type=Path)
    predict.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or cuda:N.",
    )
    predict.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save semantic probability maps as a compressed NumPy archive.",
    )
    predict.add_argument(
        "--ml-compatible",
        action="store_true",
        help="Write the legacy compatibility dictionary instead of schema v1.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _predict_one(
    predictor: MaeaPredictor,
    input_path: Path,
    output_dir: Path,
    *,
    save_probabilities: bool,
    ml_compatible: bool,
) -> dict[str, object]:
    result = predictor.predict(
        input_path,
        output_dir,
        save_probabilities=save_probabilities,
    )
    payload = result_payload(result, ml_compatible=ml_compatible)
    destination = result_path(input_path, output_dir)
    write_json(destination, payload)
    return {
        "status": "ok",
        "input": str(input_path),
        "result_path": str(destination),
        "result": payload,
    }


def _run_predict(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = discover_inputs(args.input, output_dir)
    if not inputs:
        raise MaeaInferenceError("No supported image files found under %s" % args.input)
    predictor = MaeaPredictor.from_artifacts(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    directory_mode = args.input.expanduser().resolve().is_dir()
    outcomes: list[dict[str, object]] = []
    for input_path in inputs:
        try:
            outcome = _predict_one(
                predictor,
                input_path,
                output_dir,
                save_probabilities=args.save_probabilities,
                ml_compatible=args.ml_compatible,
            )
        except Exception as exc:
            if not directory_mode:
                raise
            logger.error("Inference failed for %s: %s", input_path, str(exc))
            outcome = error_payload(input_path, exc)
        outcomes.append(outcome)
    if directory_mode:
        write_jsonl(output_dir / "results.jsonl", outcomes)
        print(
            json_text(
                {
                    "status": "completed" if all_ok(outcomes) else "partial_failure",
                    "processed": len(outcomes),
                    "failed": sum(item["status"] == "error" for item in outcomes),
                    "results": str(output_dir / "results.jsonl"),
                }
            )
        )
        return 0 if all_ok(outcomes) else 1
    print(json_text(outcomes[0]["result"]))  # type: ignore[arg-type]
    return 0


def all_ok(outcomes: Sequence[dict[str, object]]) -> bool:
    """Return whether every batch item completed successfully."""
    return all(item.get("status") == "ok" for item in outcomes)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the selected command, and return a process status."""
    parser = _parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        if args.command == "predict":
            return _run_predict(args)
    except (MaeaInferenceError, OSError, ValueError) as exc:
        logger.error("%s", str(exc))
        print(
            json_text(
                {
                    "status": "error",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            ),
            file=sys.stderr,
        )
        return 1
    parser.error("Unknown command")
    return 2
