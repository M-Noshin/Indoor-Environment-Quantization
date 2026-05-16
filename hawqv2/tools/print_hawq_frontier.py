#!/usr/bin/env python3
"""Print a readable table from a standalone HAWQ-v2 result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the Pareto frontier or top-ranked evaluated configurations "
            "from a HAWQ-v2 result JSON."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a HAWQ-v2 result JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of rows to print. Default 0 means print all rows.",
    )
    parser.add_argument(
        "--show",
        choices=("frontier", "evaluated"),
        default="frontier",
        help=(
            "Which table to print. 'frontier' prints pareto_frontier when present "
            "and falls back to the best evaluated configs. 'evaluated' always prints "
            "best evaluated configs sorted by Omega."
        ),
    )
    return parser.parse_args()


def format_bits(row: dict, layer_names: list[str]) -> str:
    bit_map = row.get("layer_bits", {})
    return "-".join(str(bit_map[layer]) for layer in layer_names)


def metric_sort_key(row: dict) -> tuple[float, float]:
    return (row.get("omega", row["metric"]), row["bit_complexity"])


def print_table(rows: list[dict], layer_names: list[str], title: str) -> None:
    if not rows:
        print(f"{title}: no rows")
        return

    cfg_width = max(4, len("-".join(["bits"] * max(1, len(layer_names)))))
    omega_width = 14
    ratio_width = 10
    complexity_width = 14

    print(title)
    header = (
        f"{'rank':>4}  "
        f"{'bits':<{cfg_width}}  "
        f"{'omega':>{omega_width}}  "
        f"{'ratio':>{ratio_width}}  "
        f"{'bit_complexity':>{complexity_width}}"
    )
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(rows, 1):
        bits = format_bits(row, layer_names)
        omega = row.get("omega", row["metric"])
        print(
            f"{idx:>4}  "
            f"{bits:<{cfg_width}}  "
            f"{omega:>{omega_width}.12g}  "
            f"{row['compression_ratio']:>{ratio_width}.12g}  "
            f"{row['bit_complexity']:>{complexity_width}.12g}"
        )


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text())

    layer_names = payload["layer_names"]
    frontier = payload.get("pareto_frontier", [])
    evaluated = sorted(payload.get("evaluated_configs", []), key=metric_sort_key)
    selected = payload.get("selected_config")

    print(f"file: {args.input}")
    print(f"method: {payload.get('method', 'unknown')}")
    print(f"selection: {payload.get('config', {}).get('selection', 'unknown')}")
    print(f"layers: {', '.join(layer_names)}")
    print(f"trace_order_low_to_high: {', '.join(payload.get('trace_order_low_to_high', []))}")
    print()

    if selected:
        print(
            "selected_config: "
            f"{format_bits(selected, layer_names)} "
            f"(omega={selected.get('omega', selected['metric']):.12g}, "
            f"ratio={selected['compression_ratio']:.12g})"
        )
        print()
    else:
        print("selected_config: none")
        print()

    if args.show == "frontier" and frontier:
        rows = frontier
        title = f"pareto_frontier ({len(frontier)} rows)"
    elif args.show == "frontier":
        rows = evaluated
        title = (
            "pareto_frontier unavailable; showing best evaluated configs "
            f"({len(evaluated)} rows total)"
        )
    else:
        rows = evaluated
        title = f"evaluated_configs sorted by Omega ({len(evaluated)} rows total)"

    if args.limit > 0:
        rows = rows[: args.limit]

    print_table(rows, layer_names, title)


if __name__ == "__main__":
    main()
