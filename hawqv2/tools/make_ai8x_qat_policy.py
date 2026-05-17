#!/usr/bin/env python3
"""Convert selected HAWQ-v2 layer bitwidths into an ai8x QAT policy override file."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "hawqv2" / "src"
sys.path.insert(0, str(SRC_ROOT))

from hawqv2.bitwidth_export import load_bitwidths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitwidths", required=True, help="HAWQ result JSON or layer-bits JSON")
    parser.add_argument("--output", required=True, help="Output ai8x QAT policy YAML")
    parser.add_argument("--start-epoch", type=int, default=8)
    parser.add_argument("--default-weight-bits", type=int, default=8)
    parser.add_argument("--shift-quantile", type=float, default=0.95)
    parser.add_argument("--outlier-removal-z-score", type=float, default=2.0)
    args = parser.parse_args()

    try:
        layer_bits = load_bitwidths(args.bitwidths)
    except ValueError as exc:
        raise SystemExit(
            f"{exc}. If the HAWQ result was produced in pareto mode, rerun with "
            "--selection compression_ratio or --selection min_metric before exporting an ai8x policy."
        ) from exc
    if not layer_bits:
        raise SystemExit("No layer weight bits were found in the provided payload.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_policy(layer_bits, args), encoding="utf-8")
    print(f"Wrote ai8x QAT policy to {output}")
    print("Selected layer bits:", dict(layer_bits))


def _render_policy(layer_bits: OrderedDict[str, int], args: argparse.Namespace) -> str:
    lines = [
        "---",
        "# Generated from standalone HAWQ-v2 weight-only precision selection.",
        f"start_epoch: {args.start_epoch}",
        f"weight_bits: {args.default_weight_bits}",
        f"shift_quantile: {args.shift_quantile}",
        f"outlier_removal_z_score: {args.outlier_removal_z_score}",
        "overrides:",
    ]
    for layer_name, bits in layer_bits.items():
        lines.extend(
            [
                f"  {layer_name}:",
                f"    weight_bits: {bits}",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
