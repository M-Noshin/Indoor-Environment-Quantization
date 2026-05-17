#!/usr/bin/env python3
"""Apply ACE selection to HAWQ-v2 candidates using pre-QAT evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ACC_COLUMNS = ("mean_acc", "test_accuracy", "accuracy", "ptq_acc", "fp32_acc")
STD_COLUMNS = ("std_acc", "ptq_std_acc", "fp32_std_acc")
SIZE_COLUMNS = ("size_KB_total", "size_kb", "size_KB", "model_size_kb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deployment candidate from HAWQ-v2 outputs using ACE. "
            "The accuracy input must come from a pre-QAT evaluation stage "
            "(for example PTQ on the HAWQ frontier), not from the exhaustive QAT sweep."
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
        "--eval-csv",
        required=True,
        help=(
            "CSV with candidate-specific pre-QAT evaluation results. Required columns: "
            "input_length or alpha, config, and one accuracy column such as mean_acc "
            "or test_accuracy. PTQ summary CSVs are the intended input."
        ),
    )
    parser.add_argument(
        "--eval-source-label",
        default="PTQ",
        help="Label used in reports for the evaluation accuracy source.",
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
        help=(
            "Baseline model size in kB. Defaults to the estimated uniform INT8 "
            "model size at the largest provided alpha."
        ),
    )
    parser.add_argument(
        "--size-csv",
        default=None,
        help=(
            "Optional deterministic size table with input_length/alpha, config, "
            "and a size column. If omitted, sizes are estimated from the indoor "
            "model architecture."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Rows to print/report after ACE sorting. Use 0 to print/report all.",
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


def normalize_config(config: str) -> str:
    config = config.strip()
    if config.startswith("INT_"):
        config = "INT " + config[len("INT_") :]
    return config.replace("_", "-")


def parse_config_bits(config: str) -> tuple[int, int, int, int]:
    config = normalize_config(config)
    if not config.startswith("INT "):
        raise ValueError(f"Invalid config {config!r}; expected e.g. 'INT 8-8-2-2'")
    parts = config[len("INT ") :].split("-")
    if len(parts) != 4:
        raise ValueError(f"Invalid config {config!r}; expected four bit-widths")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def config_from_layer_bits(row: dict[str, Any], layer_names: list[str]) -> str:
    bits = row["layer_bits"]
    return "INT " + "-".join(str(bits[layer]) for layer in layer_names)


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def row_alpha(row: dict[str, str]) -> int:
    value = row.get("input_length", "") or row.get("alpha", "")
    if not value:
        raise ValueError("CSV row is missing input_length/alpha")
    return int(value)


def load_eval_rows(path: str | Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alpha = row_alpha(row)
            config = normalize_config(row["config"])
            acc_text = first_present(row, ACC_COLUMNS)
            if acc_text is None:
                raise ValueError(
                    f"{path} must contain one accuracy column from {ACC_COLUMNS}"
                )
            std_text = first_present(row, STD_COLUMNS)
            size_text = first_present(row, SIZE_COLUMNS)
            rows[(alpha, config)] = {
                "alpha": alpha,
                "config": config,
                "eval_acc": float(acc_text),
                "eval_std_acc": None if std_text is None else float(std_text),
                "runs": int(float(row["runs"])) if row.get("runs") else None,
                "eval_size_kb": None if size_text is None else float(size_text),
            }
    return rows


def load_size_rows(path: str | Path) -> dict[tuple[int, str], float]:
    rows: dict[tuple[int, str], float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alpha = row_alpha(row)
            config = normalize_config(row["config"])
            size_text = first_present(row, SIZE_COLUMNS)
            if size_text is None:
                raise ValueError(f"{path} must contain one size column from {SIZE_COLUMNS}")
            rows[(alpha, config)] = float(size_text)
    return rows


def estimate_indoor_size_kb(alpha: int, config: str) -> float:
    """Estimate parameter memory for ai85indoorenvnetv2 in decimal kB.

    This mirrors the size accounting used in the paper tables:
    constant overhead 0.3 kB, 60 conv1 weights, 300 conv2 weights,
    2000*alpha fc1 weights, and 800 fc2 weights.
    """
    b1, b2, b3, b4 = parse_config_bits(config)
    return 0.3 + 0.0075 * b1 + 0.0375 * b2 + 0.25 * alpha * b3 + 0.1 * b4


def ace_score(
    acc: float,
    size_kb: float,
    s_base_kb: float,
    target_acc: float,
    beta1: float,
    beta2: float,
) -> float:
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
    eval_rows: dict[tuple[int, str], dict[str, Any]],
    size_rows: dict[tuple[int, str], float],
    s_base_kb: float,
    target_acc: float,
    beta1: float,
    beta2: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    joined = []
    missing = []
    for row in hawq_rows:
        key = (row["alpha"], row["config"])
        eval_row = eval_rows.get(key)
        if eval_row is None:
            missing.append(row)
            continue

        size_kb = size_rows.get(key)
        if size_kb is None:
            size_kb = eval_row["eval_size_kb"]
        if size_kb is None:
            size_kb = estimate_indoor_size_kb(row["alpha"], row["config"])

        merged = {**row, **eval_row, "size_kb": size_kb}
        merged["ace"] = ace_score(
            acc=merged["eval_acc"],
            size_kb=merged["size_kb"],
            s_base_kb=s_base_kb,
            target_acc=target_acc,
            beta1=beta1,
            beta2=beta2,
        )
        joined.append(merged)
    return joined, missing


def print_rows(rows: list[dict[str, Any]], title: str, limit: int, eval_source_label: str) -> None:
    selected = rows if limit == 0 else rows[:limit]
    print(title)
    print(
        f"{'rank':>4}  {'alpha':>5}  {'config':<14}  {eval_source_label + ' acc':>9}  "
        f"{'size_kB':>9}  {'ACE':>12}  {'omega':>14}  {'hawq_rank':>9}"
    )
    print("-" * 88)
    for rank, row in enumerate(selected, 1):
        print(
            f"{rank:>4}  {row['alpha']:>5}  {row['config']:<14}  "
            f"{row['eval_acc']:>9.4f}  {row['size_kb']:>9.3f}  "
            f"{row['ace']:>12.9f}  {row['omega']:>14.12g}  {row['hawq_rank']:>9}"
        )


def write_report_csv(rows: list[dict[str, Any]], output_path: Path, eval_source_label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ace_rank",
        "alpha",
        "config",
        "eval_source",
        "eval_acc",
        "eval_std_acc",
        "runs",
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
                    "eval_source": eval_source_label,
                    "eval_acc": row["eval_acc"],
                    "eval_std_acc": "" if row["eval_std_acc"] is None else row["eval_std_acc"],
                    "runs": "" if row["runs"] is None else row["runs"],
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
    eval_label = payload["eval_source_label"]
    lines: list[str] = []
    lines.append("# HAWQ + ACE Summary")
    lines.append("")
    lines.append(f"- Candidate set: `{payload['candidate_set']}`")
    lines.append(f"- Evaluation source: `{eval_label}`")
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
        lines.append(f"- {eval_label} accuracy: `{selected['eval_acc']:.4f}%`")
        if selected["eval_std_acc"] is not None:
            lines.append(f"- {eval_label} std.: `{selected['eval_std_acc']:.4f}%`")
        lines.append(f"- Size: `{selected['size_kb']:.3f} kB`")
        lines.append(f"- ACE: `{selected['ace']:.9f}`")
        lines.append(f"- Omega: `{selected['omega']:.12g}`")
        lines.append(f"- HAWQ rank within alpha frontier: `{selected['hawq_rank']}`")
        lines.append("")

    rows = payload["rows"] if top_n == 0 else payload["rows"][:top_n]
    lines.append(f"## Top {len(rows)} ACE-ranked candidates")
    lines.append("")
    lines.append(f"| ACE Rank | Alpha | Config | {eval_label} Acc. (%) | Size (kB) | ACE | Omega | HAWQ Rank |")
    lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {row['alpha']} | {row['config']} | "
            f"{row['eval_acc']:.4f} | {row['size_kb']:.3f} | "
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

    items = [parse_item(item) for item in args.item]
    max_alpha = max(alpha for alpha, _ in items)
    s_base_kb = (
        args.s_base_kb
        if args.s_base_kb is not None
        else estimate_indoor_size_kb(max_alpha, "INT 8-8-8-8")
    )

    eval_rows = load_eval_rows(args.eval_csv)
    size_rows = load_size_rows(args.size_csv) if args.size_csv else {}

    hawq_rows = []
    for alpha, path in items:
        hawq_rows.extend(load_hawq_candidates(alpha, path, args.candidate_set))

    joined, missing = join_rows(
        hawq_rows=hawq_rows,
        eval_rows=eval_rows,
        size_rows=size_rows,
        s_base_kb=s_base_kb,
        target_acc=args.target_acc,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    joined.sort(key=lambda row: (-row["ace"], row["size_kb"], -row["eval_acc"], row["omega"]))

    payload = {
        "method": "hawq_frontier_plus_pre_qat_eval_ace",
        "candidate_set": args.candidate_set,
        "eval_source_label": args.eval_source_label,
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
    print(f"eval_source: {args.eval_source_label}")
    print(f"target_acc: {args.target_acc}")
    print(f"beta: ({args.beta1}, {args.beta2})")
    print(f"s_base_kb: {s_base_kb:.3f}")
    print(f"joined: {len(joined)} / {len(hawq_rows)} HAWQ candidates")
    if missing:
        print(f"missing eval rows: {len(missing)}")
    print()

    if joined:
        selected = joined[0]
        print(
            "selected: "
            f"{selected['config']} @ alpha={selected['alpha']} "
            f"{args.eval_source_label.lower()}_acc={selected['eval_acc']:.4f}% "
            f"size={selected['size_kb']:.3f} kB "
            f"ACE={selected['ace']:.9f} "
            f"omega={selected['omega']:.12g}"
        )
        print()
    print_rows(joined, "ACE-ranked HAWQ candidates", args.limit, args.eval_source_label)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"Wrote {output}")
    if args.report_csv:
        report_csv = Path(args.report_csv)
        write_report_csv(joined, report_csv, args.eval_source_label)
        print(f"Wrote {report_csv}")
    if args.report_md:
        report_md = Path(args.report_md)
        top_n = len(joined) if args.limit == 0 else args.limit
        write_report_md(payload, report_md, top_n=top_n)
        print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
