#!/usr/bin/env python3
"""Run the standalone HAWQ-v2 selector on the indoor ai8x model."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "hawqv2" / "src"
TRAINING_ROOT = REPO_ROOT / "training"
TRAINING_DATASETS_ROOT = TRAINING_ROOT / "datasets"
TRAINING_MODELS_ROOT = TRAINING_ROOT / "models"

for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ai8x-root",
        help="Path to a full ai8x-training checkout. If omitted, uses $AI8X_TRAINING_ROOT or local training/ only.",
    )
    parser.add_argument("--data-dir", required=True, help="Path to the indoor .mat dataset directory")
    parser.add_argument("--checkpoint", help="Optional float checkpoint to load before HAWQ analysis")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--policy-output", help="Optional ai8x QAT policy YAML output")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split and HAWQ trace sampling")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-length", type=int, default=101)
    parser.add_argument(
        "--selection",
        choices=["pareto", "compression_ratio", "min_metric"],
        default="pareto",
        help="HAWQ-V2 selection mode. 'pareto' exports the frontier without choosing one config.",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=None,
        help="Optional compression-ratio target used only with --selection compression_ratio.",
    )
    parser.add_argument("--num-data-points", type=int, default=100)
    parser.add_argument("--max-trace-iters", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--search", choices=["monotonic", "all"], default="monotonic")
    parser.add_argument("--quantization", choices=["asymmetric", "symmetric"], default="asymmetric")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--layers", nargs="+", default=["conv1", "conv2", "fc1", "fc2"])
    parser.add_argument("--start-epoch", type=int, default=8)
    parser.add_argument("--act-mode-8bit", action="store_true")
    return parser


def resolve_device(device: str) -> str:
    resolved_device = device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"
    return resolved_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_and_loader(args: argparse.Namespace) -> tuple[torch.nn.Module, DataLoader, str]:
    configure_import_paths(args.ai8x_root)
    set_seed(args.seed)

    import ai8x
    from ai85net_indoor_env_v2 import ai85indoorenvnetv2
    from indoor_environment_1D import indoor_environment_get_datasets

    resolved_device = resolve_device(args.device)

    ai8x.set_device(87, args.act_mode_8bit, False, verbose=False)
    model = ai85indoorenvnetv2(
        pretrained=False,
        num_classes=4,
        num_channels=2,
        dimensions=(args.input_length, 1),
        bias=True,
    )
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, resolved_device)
    model.eval()

    data_args = SimpleNamespace(input_1d_length=args.input_length, act_mode_8bit=args.act_mode_8bit)
    train_dataset, _ = indoor_environment_get_datasets((args.data_dir, data_args), load_train=True, load_test=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(resolved_device == "cuda"),
    )
    return model, train_loader, resolved_device


def run_indoor_hawq(args: argparse.Namespace) -> dict[str, Any]:
    from hawqv2.selector import run_hawqv2

    model, train_loader, resolved_device = build_model_and_loader(args)

    return run_hawqv2(
        model=model,
        data_loader=train_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        layer_names=args.layers,
        candidate_bits=args.bits,
        device=resolved_device,
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


def main() -> None:
    args = build_parser().parse_args()

    from hawqv2.selector import save_hawq_result

    result = run_indoor_hawq(args)
    save_hawq_result(result, args.output)
    print(f"Wrote HAWQ-v2 result to {args.output}")
    if result["selected_config"] is not None:
        print("Selected layer bits:", dict(result["selected_config"]["layer_bits"]))
    else:
        print(f"Pareto frontier points: {len(result.get('pareto_frontier', []))}")

    if args.policy_output and result["selected_config"] is not None:
        write_ai8x_policy(result["selected_config"]["layer_bits"], args.policy_output, args.start_epoch)
        print(f"Wrote ai8x QAT policy to {args.policy_output}")
    elif args.policy_output:
        print("Skipped ai8x policy export because no single config was selected in pareto mode.")


def load_checkpoint(model, checkpoint_path: str, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = value
    model.load_state_dict(cleaned, strict=False)


def write_ai8x_policy(layer_bits: dict[str, int], output_path: str, start_epoch: int) -> None:
    lines = [
        "---",
        "# Generated from standalone HAWQ-v2 selection.",
        f"start_epoch: {start_epoch}",
        "weight_bits: 8",
        "shift_quantile: 0.95",
        "outlier_removal_z_score: 2.0",
        "overrides:",
    ]
    for layer_name, bits in layer_bits.items():
        lines.extend(
            [
                f"  {layer_name}:",
                f"    weight_bits: {bits}",
            ]
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_import_paths(ai8x_root_arg: str | None) -> None:
    ai8x_root = ai8x_root_arg or os.environ.get("AI8X_TRAINING_ROOT")

    candidate_paths = [
        str(REPO_ROOT),
        str(SRC_ROOT),
        str(TRAINING_ROOT),
        str(TRAINING_DATASETS_ROOT),
        str(TRAINING_MODELS_ROOT),
    ]

    if ai8x_root:
        ai8x_root_path = Path(ai8x_root).expanduser().resolve()
        if not ai8x_root_path.exists():
            raise FileNotFoundError(f"ai8x root does not exist: {ai8x_root_path}")
        candidate_paths.insert(0, str(ai8x_root_path))

    for path in reversed(candidate_paths):
        if path not in sys.path:
            sys.path.insert(0, path)


if __name__ == "__main__":
    main()
