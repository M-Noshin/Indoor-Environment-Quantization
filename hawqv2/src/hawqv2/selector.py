"""Public HAWQ-v2 selector API."""

from __future__ import annotations

import json
from pathlib import Path

from .initializer import StandaloneHAWQPrecisionInitializer
from .initializer import run_hawqv2


def save_hawq_result(result: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


__all__ = ["StandaloneHAWQPrecisionInitializer", "run_hawqv2", "save_hawq_result"]
