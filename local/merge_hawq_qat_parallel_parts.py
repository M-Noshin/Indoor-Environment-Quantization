#!/usr/bin/env python3
"""
Merge hawq_qat_sweep_{results,summary}.csv from parallel Slurm parts (hawq_qat_frontier_p01..p05).

Recomputes the summary from the combined per-run results so ACE can use a single CSV:
  python hawqv2/tools/select_ace_from_hawq.py ... --eval-csv <merged>/hawq_qat_sweep_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Match training/train_indoor_1D_hawq_qat_sweep.write_results aggregation."""
    if results.empty or "status" not in results.columns:
        return pd.DataFrame(columns=["input_length", "config", "runs", "mean_acc", "std_acc"])
    succ = results[results["status"] == "success"].copy()
    if succ.empty:
        return pd.DataFrame(columns=["input_length", "config", "runs", "mean_acc", "std_acc"])
    summary = succ.groupby(["input_length", "config"], as_index=False).agg(
        runs=("test_accuracy", "count"),
        mean_acc=("test_accuracy", "mean"),
        std_acc=("test_accuracy", "std"),
        total_time=("train_seconds", "sum"),
    )
    summary["conv1_bits"] = summary["config"].str.extract(r"INT (\d+)-")[0].astype(int)
    summary["conv2_bits"] = summary["config"].str.extract(r"-(\d+)-")[0].astype(int)
    summary["fc1_bits"] = summary["config"].str.extract(r"-(\d+)-(\d+)$")[0].astype(int)
    summary["fc2_bits"] = summary["config"].str.extract(r"-(\d+)$")[0].astype(int)
    summary.sort_values(by=["mean_acc", "input_length"], ascending=[False, False], inplace=True)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ai8x-training-root",
        type=Path,
        default=Path("/shared/b00090279/testMax/ai8x-training"),
        help="Root that contains hawq_qat_frontier_p01, ...",
    )
    p.add_argument(
        "--part-prefix",
        default="hawq_qat_frontier_p",
        help="Directory name prefix; parts are <prefix>01 .. <prefix>05 by default.",
    )
    p.add_argument(
        "--parts",
        type=str,
        default="01,02,03,04,05",
        help="Comma-separated two-digit suffixes (default 01..05).",
    )
    p.add_argument(
        "--merged-dir-name",
        default="hawq_qat_frontier_merged",
        help="Subdirectory under ai8x-training-root to write merged CSVs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.ai8x_training_root.resolve()
    suffixes = [s.strip() for s in args.parts.split(",") if s.strip()]
    frames: list[pd.DataFrame] = []
    for suf in suffixes:
        part_dir = root / f"{args.part_prefix}{suf}"
        res_path = part_dir / "hawq_qat_sweep_results.csv"
        if not res_path.is_file():
            print(f"[skip] missing: {res_path}")
            continue
        df = pd.read_csv(res_path)
        if df.empty:
            print(f"[skip] empty: {res_path}")
            continue
        df["_merge_part"] = f"p{suf}"
        frames.append(df)
        print(f"[ok] {res_path} rows={len(df)}")

    if not frames:
        raise SystemExit("[err] No results files found; nothing to merge.")

    merged = pd.concat(frames, ignore_index=True)
    key_cols = ["input_length", "seed", "candidate_index"]
    if all(c in merged.columns for c in key_cols):
        before = len(merged)
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
        if len(merged) < before:
            print(f"[info] dropped {before - len(merged)} duplicate rows on {key_cols}")

    out_dir = root / args.merged_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "hawq_qat_sweep_results.csv"
    summary_path = out_dir / "hawq_qat_sweep_summary.csv"
    drop_cols = [c for c in ("_merge_part",) if c in merged.columns]
    merged.drop(columns=drop_cols, errors="ignore").to_csv(results_path, index=False)
    summary = build_summary(merged.drop(columns=drop_cols, errors="ignore"))
    summary.to_csv(summary_path, index=False)

    print(f"[write] {results_path} rows={len(merged)}")
    print(f"[write] {summary_path} rows={len(summary)}")
    print(
        "[next] select_ace_from_hawq.py ... "
        f'--eval-csv "{summary_path}" --eval-source-label QAT ...'
    )


if __name__ == "__main__":
    main()
