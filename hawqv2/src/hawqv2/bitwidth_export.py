"""Helpers for loading and saving selected layer bitwidths."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


def extract_layer_weight_bits(payload) -> OrderedDict[str, int]:
    """Extract ``{layer_name: bits}`` from supported JSON payloads."""

    if isinstance(payload, (str, Path)):
        payload = _load_json(Path(payload))

    if isinstance(payload, dict):
        if (
            "selected_config" in payload
            and payload["selected_config"] is not None
            and "layer_bits" in payload["selected_config"]
        ):
            return _normalize_layer_bits(payload["selected_config"]["layer_bits"])
        if "layer_bits" in payload:
            return _normalize_layer_bits(payload["layer_bits"])
        if "bitwidth_per_scope" in payload:
            rows = payload["bitwidth_per_scope"]
            return _normalize_layer_bits(OrderedDict((str(scope), int(bits)) for bits, scope in rows))

    if isinstance(payload, list):
        if payload and isinstance(payload[0], (list, tuple)) and len(payload[0]) == 2:
            return _normalize_layer_bits(OrderedDict((str(scope), int(bits)) for bits, scope in payload))

    raise ValueError("Could not extract layer bitwidths from the provided payload")


def load_bitwidths(path: str | Path) -> OrderedDict[str, int]:
    return extract_layer_weight_bits(_load_json(Path(path)))


def save_bitwidths_json(bitwidths: dict[str, int] | Iterable[tuple[str, int]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    layer_bits = _normalize_layer_bits(bitwidths)
    output.write_text(json.dumps({"layer_bits": layer_bits}, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_layer_bits(bitwidths) -> OrderedDict[str, int]:
    if isinstance(bitwidths, dict):
        items = bitwidths.items()
    else:
        items = bitwidths
    ordered = OrderedDict()
    for layer_name, bits in items:
        ordered[str(layer_name)] = int(bits)
    return ordered
