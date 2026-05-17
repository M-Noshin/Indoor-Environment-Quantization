#!/usr/bin/env python3
"""Extract selected layer bitwidths from a HAWQ-v2 result JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "hawqv2" / "src"
sys.path.insert(0, str(SRC_ROOT))

from hawqv2.bitwidth_export import load_bitwidths
from hawqv2.bitwidth_export import save_bitwidths_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HAWQ result JSON or layer-bits JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    bitwidths = load_bitwidths(args.input)
    save_bitwidths_json(bitwidths, args.output)
    print(f"Wrote {len(bitwidths)} bitwidth rows to {args.output}")


if __name__ == "__main__":
    main()
