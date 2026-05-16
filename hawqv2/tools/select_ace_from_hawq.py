#!/usr/bin/env python3
"""Apply ACE selection to candidates produced by HAWQ-v2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join HAWQ-v2 candidates with QAT sweep results and select the best "
            "configuration using the paper's ACE rule."
        )
    )
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        metavar="ALPHA:HAWQ_JSON",
        help="Input-length/alpha and HAWQ result path. Can be repeated.",
    )
    parser.add_argument(
        "--sweep-csv",
        required=True,
        help="CSV with columns input_length, config, size_KB_total, mean_acc, std_acc.",
    )
    parser.add_argument(
        "--candidate-set",
        choices=("frontier", "evaluated"),
        default="frontier",
        help="Use the HAWQ Pareto frontier or all evaluated HAWQ configs.",
    )
    parser.add_argument("--target-acc", type=float, default=99.2)
    parser.add_argument("--beta1", type=float, default=1.0)
    parser.add_argument("--beta2", type=float, default=0.0)
    parser.add_argument(
        "--s-base-kb",
        type=float,
        default=None,
        help="Baseline model size in kB. Defaults to the largest size in the sweep CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Rows to print after ACE sorting. Use 0 to print all.",
    )
    parser.add_argument("--output", help="Optional JSON output path for the joined result.")
    parser.add_argument(
        "--report-md",
        help="Optional Markdown report path with the selected result and top candidates.",
    )
    parser.add_argument(
        "--report-csv",
        help="Optional compact CSV path with ACE-ranked candidates.",
    )
    return parser.parse_args()


def parse_item(item: str) -> tuple[int, Path]:
    if ":" not in item:
        raise ValueError(f"Invalid --item {item!r}; expected ALPHA:HAWQ_JSON")
    alpha_text, path_text = item.split(":", 1)
    return int(alpha_text), Path(path_text)


def config_from_layer_bits(row: dict[str, Any], layer_names: list[str]) -> str:
    bits = row["layer_bits"]
    return "INT " + "-".join(str(bits[layer]) for layer in layer_names)


def load_sweep_rows(path: str | Path) -> tuple[dict[tuple[int, str], dict[str, Any]], float]:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    max_size = 0.0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            alpha = int(row["input_length"])
            config = row["config"]
            size_kb = float(row["size_KB_total"])
            max_size = max(max_size, size_kb)
            rows[(alpha, config)] = {
                "alpha": alpha,
                "config": config,
                "size_kb": size_kb,
                "mean_acc": float(row["mean_acc"]),
                "std_acc": float(row["std_acc"]),
                "runs": int(row["runs"]),
            }
    return rows, max_size


def ace_score(acc: float, size_kb: float, s_base_kb: float, target_acc: float, beta1: float, beta2: float) -> float:
    if acc >= target_acc:
        compactness = 1.0 - size_kb / s_base_kb
        accuracy_surplus = (acc - target_acc) / (100.0 - target_acc)
        return beta1 * compactness + beta2 * accuracy_surplus
    return -((target_acc - acc) / target_acc)


def load_hawq_candidates(alpha: int, path: Path, candidate_set: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "pareto_frontier" if candidate_set == "frontier" else "evaluated_configs"
    candidates = payload.get(key, [])
    layer_names = payload["layer_names"]
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        rows.append(
            {
                "alpha": alpha,
                "hawq_rank": rank,
                "config": config_from_layer_bits(candidate, layer_names),
                "omega": float(candidate.get("omega", candidate["metric"])),
                "hawq_compression_ratio": float(candidate["compression_ratio"]),
                "hawq_bit_complexity": float(candidate["bit_complexity"]),
                "hawq_source": str(path),
            }
        )
    return rows


def join_rows(
    hawq_rows: list[dict[str, Any]],
    sweep_rows: dict[tuple[int, str], dict[str, Any]],
    s_base_kb: float,
    target_acc: float,
    beta1: float,
    beta2: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    joined = []
    missing = []
    for row in hawq_rows:
        sweep = sweep_rows.get((row["alpha"], row["config"]))
        if sweep is None:
            missing.append(row)
            continue
        merged = {**row, **sweep}
        merged["ace"] = ace_score(
            acc=merged["mean_acc"],
            size_kb=merged["size_kb"],
            s_base_kb=s_base_kb,
            target_acc=target_acc,
            beta1=beta1,
            beta2=beta2,
        )
        joined.append(merged)
    return joined, missing


def print_rows(rows: list[dict[str, Any]], title: str, limit: int) -> None:
    selected = rows if limit == 0 else rows[:limit]
    print(title)
    print(
        f"{'rank':>4}  {'alpha':>5}  {'config':<14}  {'acc':>9}  "
        f"{'size_kB':>9}  {'ACE':>12}  {'omega':>14}  {'hawq_rank':>9}"
    )
    print("-" * 88)
    for rank, row in enumerate(selected, 1):
        print(
            f"{rank:>4}  {row['alpha']:>5}  {row['config']:<14}  "
            f"{row['mean_acc']:>9.4f}  {row['size_kb']:>9.3f}  "
            f"{row['ace']:>12.9f}  {row['omega']:>14.12g}  {row['hawq_rank']:>9}"
        )


def write_report_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ace_rank",
        "alpha",
        "config",
        "mean_acc",
        "std_acc",
        "size_kb",
        "ace",
        "omega",
        "hawq_rank",
        "hawq_compression_ratio",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "ace_rank": index,
                    "alpha": row["alpha"],
                    "config": row["config"],
                    "mean_acc": row["mean_acc"],
                    "std_acc": row["std_acc"],
                    "size_kb": row["size_kb"],
                    "ace": row["ace"],
                    "omega": row["omega"],
                    "hawq_rank": row["hawq_rank"],
                    "hawq_compression_ratio": row["hawq_compression_ratio"],
                }
            )


def write_report_md(
    payload: dict[str, Any],
    output_path: Path,
    top_n: int = 20,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# HAWQ + ACE Summary")
    lines.append("")
    lines.append(f"- Candidate set: `{payload['candidate_set']}`")
    lines.append(f"- Target accuracy: `{payload['target_acc']}`")
    lines.append(f"- ACE weights: `({payload['beta1']}, {payload['beta2']})`")
    lines.append(f"- Baseline size: `{payload['s_base_kb']:.3f} kB`")
    lines.append(f"- Joined HAWQ candidates: `{payload['num_joined_candidates']}`")
    lines.append("")

    selected = payload.get("selected")
    if selected is not None:
        lines.append("## Selected")
        lines.append("")
        lines.append(f"- Config: `{selected['config']}`")
        lines.append(f"- Alpha: `{selected['alpha']}`")
        lines.append(f"- Accuracy: `{selected['mean_acc']:.4f}%`")
        lines.append(f"- Size: `{selected['size_kb']:.3f} kB`")
        lines.append(f"- ACE: `{selected['ace']:.9f}`")
        lines.append(f"- Omega: `{selected['omega']:.12g}`")
        lines.append(f"- HAWQ rank within alpha frontier: `{selected['hawq_rank']}`")
        lines.append("")

    lines.append(f"## Top {min(top_n, len(payload['rows']))} ACE-ranked candidates")
    lines.append("")
    lines.append("| ACE Rank | Alpha | Config | Acc. (%) | Size (kB) | ACE | Omega | HAWQ Rank |")
    lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for index, row in enumerate(payload["rows"][:top_n], 1):
        lines.append(
            f"| {index} | {row['alpha']} | {row['config']} | "
            f"{row['mean_acc']:.4f} | {row['size_kb']:.3f} | "
            f"{row['ace']:.9f} | {row['omega']:.12g} | {row['hawq_rank']} |"
        )
    lines.append("")

    by_alpha: dict[int, int] = {}
    for row in payload["rows"]:
        by_alpha[row["alpha"]] = by_alpha.get(row["alpha"], 0) + 1
    lines.append("## Frontier Counts by Alpha")
    lines.append("")
    for alpha in sorted(by_alpha.keys(), reverse=True):
        lines.append(f"- `alpha={alpha}`: `{by_alpha[alpha]}` candidates")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if abs((args.beta1 + args.beta2) - 1.0) > 1e-9:
        raise ValueError("--beta1 and --beta2 must sum to 1.0")

    sweep_rows, csv_max_size = load_sweep_rows(args.sweep_csv)
    s_base_kb = args.s_base_kb if args.s_base_kb is not None else csv_max_size

    hawq_rows = []
    for item in args.item:
        alpha, path = parse_item(item)
        hawq_rows.extend(load_hawq_candidates(alpha, path, args.candidate_set))

    joined, missing = join_rows(
        hawq_rows=hawq_rows,
        sweep_rows=sweep_rows,
        s_base_kb=s_base_kb,
        target_acc=args.target_acc,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    joined.sort(key=lambda row: (-row["ace"], row["size_kb"], -row["mean_acc"], row["omega"]))

    payload = {
        "method": "hawq_frontier_plus_ace",
        "candidate_set": args.candidate_set,
        "target_acc": args.target_acc,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "s_base_kb": s_base_kb,
        "num_hawq_candidates": len(hawq_rows),
        "num_joined_candidates": len(joined),
        "num_missing_candidates": len(missing),
        "selected": joined[0] if joined else None,
        "rows": joined,
        "missing": missing,
    }

    print(f"candidate_set: {args.candidate_set}")
    print(f"target_acc: {args.target_acc}")
    print(f"beta: ({args.beta1}, {args.beta2})")
    print(f"s_base_kb: {s_base_kb:.3f}")
    print(f"joined: {len(joined)} / {len(hawq_rows)} HAWQ candidates")
    if missing:
        print(f"missing sweep rows: {len(missing)}")
    print()

    if joined:
        selected = joined[0]
        print(
            "selected: "
            f"{selected['config']} @ alpha={selected['alpha']} "
            f"acc={selected['mean_acc']:.4f}% "
            f"size={selected['size_kb']:.3f} kB "
            f"ACE={selected['ace']:.9f} "
            f"omega={selected['omega']:.12g}"
        )
        print()
    print_rows(joined, "ACE-ranked HAWQ candidates", args.limit)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"Wrote {output}")
    if args.report_csv:
        report_csv = Path(args.report_csv)
        write_report_csv(joined, report_csv)
        print(f"Wrote {report_csv}")
    if args.report_md:
        report_md = Path(args.report_md)
        write_report_md(payload, report_md)
        print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
