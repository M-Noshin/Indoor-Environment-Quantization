#!/usr/bin/env python3
"""Generic CLI for the standalone HAWQ-v2 selector."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "hawqv2" / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from hawqv2.selector import run_hawqv2
from hawqv2.selector import save_hawq_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factory",
        required=True,
        help="Import path in the form module:function. The function must return model, data_loader, criterion, "
             "and optionally criterion_fn.",
    )
    parser.add_argument(
        "--factory-kwargs",
        default="{}",
        help="JSON object passed as keyword arguments to the factory function",
    )
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--selection", choices=["pareto", "compression_ratio", "min_metric"], default="pareto")
    parser.add_argument("--compression-ratio", type=float, default=None)
    parser.add_argument("--num-data-points", type=int, default=100)
    parser.add_argument("--max-trace-iters", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--search", choices=["monotonic", "all"], default="monotonic")
    parser.add_argument("--quantization", choices=["asymmetric", "symmetric"], default="asymmetric")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--layers", nargs="+", help="Optional explicit target layer names")
    args = parser.parse_args()

    factory_kwargs = json.loads(args.factory_kwargs)
    model, data_loader, criterion, criterion_fn = load_factory_outputs(args.factory, factory_kwargs)
    result = run_hawqv2(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        layer_names=args.layers,
        candidate_bits=args.bits,
        device=args.device,
        criterion_fn=criterion_fn,
        num_data_points=args.num_data_points,
        max_trace_iters=args.max_trace_iters,
        tolerance=args.tolerance,
        selection=args.selection,
        compression_ratio=args.compression_ratio,
        quantization_mode=args.quantization,
        per_channel=args.per_channel,
        search=args.search,
        eval_mode=True,
    )
    save_hawq_result(result, args.output)
    print(f"Wrote HAWQ-v2 result to {args.output}")
    if result["selected_config"] is not None:
        print("Selected layer bits:", dict(result["selected_config"]["layer_bits"]))
    else:
        print(f"Pareto frontier points: {len(result.get('pareto_frontier', []))}")


def load_factory_outputs(factory_path: str, factory_kwargs: dict):
    module_name, func_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    built = factory(**factory_kwargs)
    if isinstance(built, dict):
        model = built["model"]
        data_loader = built["data_loader"]
        criterion = built["criterion"]
        criterion_fn = built.get("criterion_fn")
        return model, data_loader, criterion, criterion_fn
    if isinstance(built, (tuple, list)):
        if len(built) == 3:
            return built[0], built[1], built[2], None
        if len(built) == 4:
            return built[0], built[1], built[2], built[3]
    raise ValueError(
        "Factory must return either a dict with model/data_loader/criterion[/criterion_fn] "
        "or a tuple of length 3 or 4"
    )


if __name__ == "__main__":
    main()
