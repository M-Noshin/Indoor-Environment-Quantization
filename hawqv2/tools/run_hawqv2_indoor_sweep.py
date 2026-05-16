#!/usr/bin/env python3
"""Run standalone HAWQ-v2 across an indoor alpha/input-length sweep."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from run_hawqv2_indoor import build_parser
from run_hawqv2_indoor import run_indoor_hawq
from run_hawqv2_indoor import write_ai8x_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "hawqv2" / "src"

for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep HAWQ-v2 over multiple indoor alpha/input-length checkpoints."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a JSON manifest describing the alpha/input-length/checkpoint sweep items.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-item HAWQ results, policies, and aggregate summaries will be written.",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path for the aggregate JSON summary. Defaults to <output-dir>/summary.json",
    )
    parser.add_argument(
        "--summary-csv",
        help="Optional path for the aggregate CSV summary. Defaults to <output-dir>/summary.csv",
    )
    parser.add_argument(
        "--policy-dir",
        help="Optional directory for generated ai8x policies. Defaults to <output-dir>/policies",
    )
    parser.add_argument(
        "--skip-policy",
        action="store_true",
        help="Do not emit per-item ai8x policy YAML files.",
    )

    indoor_parser = build_parser()
    for action in indoor_parser._actions:
        if not action.option_strings:
            continue
        if action.dest in {"output", "policy_output", "input_length", "checkpoint"}:
            continue
        if action.dest == "help":
            continue
        parser._add_action(deepcopy(action))
    return parser


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("items")
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError("Manifest must be a JSON list or an object with an 'items' list.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry {index} is not an object.")
        if "alpha" not in item:
            raise ValueError(f"Manifest entry {index} is missing required field 'alpha'.")
        if "input_length" not in item:
            raise ValueError(f"Manifest entry {index} is missing required field 'input_length'.")
        if "checkpoint" not in item:
            raise ValueError(f"Manifest entry {index} is missing required field 'checkpoint'.")
        normalized.append(item)
    return normalized


def sanitize_tag(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value.strip())
    return clean or "item"


def build_item_namespace(base_args: argparse.Namespace, item: dict[str, Any], result_path: Path) -> argparse.Namespace:
    values = vars(base_args).copy()
    values["checkpoint"] = str(Path(item["checkpoint"]).expanduser())
    values["input_length"] = int(item["input_length"])
    values["output"] = str(result_path)

    if "data_dir" in item:
        values["data_dir"] = str(Path(item["data_dir"]).expanduser())
    if "layers" in item:
        values["layers"] = list(item["layers"])
    if "bits" in item:
        values["bits"] = [int(bit) for bit in item["bits"]]
    if "selection" in item:
        values["selection"] = str(item["selection"])
    if "compression_ratio" in item:
        values["compression_ratio"] = float(item["compression_ratio"])
    if "num_data_points" in item:
        values["num_data_points"] = int(item["num_data_points"])
    if "max_trace_iters" in item:
        values["max_trace_iters"] = int(item["max_trace_iters"])
    if "tolerance" in item:
        values["tolerance"] = float(item["tolerance"])
    if "search" in item:
        values["search"] = str(item["search"])
    if "quantization" in item:
        values["quantization"] = str(item["quantization"])
    if "per_channel" in item:
        values["per_channel"] = bool(item["per_channel"])
    if "batch_size" in item:
        values["batch_size"] = int(item["batch_size"])
    if "num_workers" in item:
        values["num_workers"] = int(item["num_workers"])
    if "seed" in item:
        values["seed"] = int(item["seed"])
    if "act_mode_8bit" in item:
        values["act_mode_8bit"] = bool(item["act_mode_8bit"])
    return argparse.Namespace(**values)


def build_summary_row(
    item: dict[str, Any],
    tag: str,
    result: dict[str, Any],
    result_path: Path,
    policy_path: Path | None,
) -> dict[str, Any]:
    selected = result["selected_config"]
    layer_bits = dict(selected["layer_bits"]) if selected is not None else {}
    activation_bits = dict(selected.get("activation_bits", {})) if selected is not None else {}
    row = {
        "tag": tag,
        "alpha": item["alpha"],
        "input_length": int(item["input_length"]),
        "seed": item.get("seed"),
        "checkpoint": str(Path(item["checkpoint"]).expanduser()),
        "result_path": str(result_path),
        "policy_path": str(policy_path) if policy_path else None,
        "selection": result["config"]["selection"],
        "compression_ratio_target": result["config"]["compression_ratio"],
        "compression_ratio_selected": selected["compression_ratio"] if selected is not None else None,
        "metric": selected["metric"] if selected is not None else None,
        "bit_complexity": selected["bit_complexity"] if selected is not None else None,
        "layer_bits": layer_bits,
        "activation_bits": activation_bits,
        "pareto_frontier_size": len(result.get("pareto_frontier", [])),
        "trace_order_low_to_high": result.get("trace_order_low_to_high", []),
    }
    if "notes" in item:
        row["notes"] = item["notes"]
    return row


def write_summary_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "hawqv2_indoor_sweep",
        "num_items": len(rows),
        "items": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag",
        "alpha",
        "input_length",
        "seed",
        "checkpoint",
        "result_path",
        "policy_path",
        "selection",
        "compression_ratio_target",
        "compression_ratio_selected",
        "metric",
        "bit_complexity",
        "pareto_frontier_size",
        "layer_bits_json",
        "activation_bits_json",
        "trace_order_low_to_high_json",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "tag": row["tag"],
                    "alpha": row["alpha"],
                    "input_length": row["input_length"],
                    "seed": row.get("seed"),
                    "checkpoint": row["checkpoint"],
                    "result_path": row["result_path"],
                    "policy_path": row["policy_path"],
                    "selection": row["selection"],
                    "compression_ratio_target": row["compression_ratio_target"],
                    "compression_ratio_selected": row["compression_ratio_selected"],
                    "metric": row["metric"],
                    "bit_complexity": row["bit_complexity"],
                    "pareto_frontier_size": row["pareto_frontier_size"],
                    "layer_bits_json": json.dumps(row["layer_bits"], sort_keys=True),
                    "activation_bits_json": json.dumps(row["activation_bits"], sort_keys=True),
                    "trace_order_low_to_high_json": json.dumps(row["trace_order_low_to_high"]),
                    "notes": row.get("notes", ""),
                }
            )


def main() -> None:
    parser = build_sweep_parser()
    args = parser.parse_args()

    manifest_items = load_manifest(args.manifest)
    output_dir = Path(args.output_dir).expanduser().resolve()
    results_dir = output_dir / "results"
    policies_dir = Path(args.policy_dir).expanduser().resolve() if args.policy_dir else output_dir / "policies"
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_dir / "summary.json"
    summary_csv = Path(args.summary_csv).expanduser().resolve() if args.summary_csv else output_dir / "summary.csv"

    from hawqv2.selector import save_hawq_result

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(manifest_items):
        tag = sanitize_tag(str(item.get("tag", f"alpha_{item['alpha']}_L{item['input_length']}")))
        result_path = results_dir / f"{tag}.json"
        item_args = build_item_namespace(args, item, result_path)

        print(
            f"[{index + 1}/{len(manifest_items)}] alpha={item['alpha']} "
            f"input_length={item['input_length']} checkpoint={item['checkpoint']}"
        )
        result = run_indoor_hawq(item_args)
        save_hawq_result(result, result_path)

        policy_path: Path | None = None
        if not args.skip_policy and result["selected_config"] is not None:
            policy_path = policies_dir / f"{tag}.yaml"
            write_ai8x_policy(result["selected_config"]["layer_bits"], str(policy_path), item_args.start_epoch)

        rows.append(build_summary_row(item, tag, result, result_path, policy_path))
        if result["selected_config"] is not None:
            print(f"  selected bits: {dict(result['selected_config']['layer_bits'])}")
        else:
            print(f"  pareto frontier points: {len(result.get('pareto_frontier', []))}")
        print(f"  wrote result: {result_path}")
        if policy_path is not None:
            print(f"  wrote policy: {policy_path}")

    write_summary_json(rows, summary_json)
    write_summary_csv(rows, summary_csv)
    print(f"Wrote sweep summary JSON to {summary_json}")
    print(f"Wrote sweep summary CSV to {summary_csv}")


if __name__ == "__main__":
    main()
