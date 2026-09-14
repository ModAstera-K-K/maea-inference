"""JSON serialization helpers shared by CLI flows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def json_text(payload: Mapping[str, Any]) -> str:
    """Serialize a result without permitting non-standard NaN values."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a readable UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, payloads: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic newline-delimited JSON outcomes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json_text(payload) + "\n" for payload in payloads)
    path.write_text(body, encoding="utf-8")
