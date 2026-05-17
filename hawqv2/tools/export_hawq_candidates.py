#!/usr/bin/env python3
"""Export HAWQ-v2 candidates to a CSV consumable by the PTQ runner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        metavar="ALPHA:HAWQ_JSON",
        help="Input-length/alpha and HAWQ result path. Can be repeated.",
    )
    parser.add_argument(
        "--candidate-set",
        choices=("frontier", "evaluated"),
        default="frontier",
        help="Export the HAWQ Pareto frontier or all evaluated HAWQ configs.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def parse_item(item: str) -> tuple[int, Path]:
    if ":" not in item:
        raise ValueError(f"Invalid --item {item!r}; expected ALPHA:HAWQ_JSON")
    alpha_text, path_text = item.split(":", 1)
    return int(alpha_text), Path(path_text)


def config_from_layer_bits(row: dict[str, Any], layer_names: list[str]) -> tuple[str, list[int]]:
    bits = [int(row["layer_bits"][layer]) for layer in layer_names]
    return "INT " + "-".join(str(bit) for bit in bits), bits


def load_rows(alpha: int, path: Path, candidate_set: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "pareto_frontier" if candidate_set == "frontier" else "evaluated_configs"
    layer_names = payload["layer_names"]
    rows = []
    for rank, candidate in enumerate(payload.get(key, []), 1):
        config, bits = config_from_layer_bits(candidate, layer_names)
        rows.append(
            {
                "input_length": alpha,
                "alpha": alpha,
                "config": config,
                "conv1_bits": bits[0],
                "conv2_bits": bits[1],
                "fc1_bits": bits[2],
                "fc2_bits": bits[3],
                "hawq_rank": rank,
                "omega": float(candidate.get("omega", candidate["metric"])),
                "hawq_compression_ratio": float(candidate["compression_ratio"]),
                "hawq_bit_complexity": float(candidate["bit_complexity"]),
                "hawq_source": str(path),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for item in args.item:
        alpha, path = parse_item(item)
        rows.extend(load_rows(alpha, path, args.candidate_set))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_length",
        "alpha",
        "config",
        "conv1_bits",
        "conv2_bits",
        "fc1_bits",
        "fc2_bits",
        "hawq_rank",
        "omega",
        "hawq_compression_ratio",
        "hawq_bit_complexity",
        "hawq_source",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} candidates to {output}")


if __name__ == "__main__":
    main()
