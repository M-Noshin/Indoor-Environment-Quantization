#!/usr/bin/env python3
"""
QAT sweep over HAWQ-selected mixed-precision candidates.

This runner is intentionally separate from train_indoor_1D_mixed_sweep.py so the
original exhaustive 891-point QAT driver stays simple. The expected workflow is:

1) Export HAWQ candidates to CSV with hawqv2/tools/export_hawq_candidates.py.
2) QAT-train only those candidates with this script.
3) Apply ACE to this script's QAT summary.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


PYTHON_EXECUTABLE = sys.executable


def _ensure_dir_symlink(link: Path, target: Path) -> None:
    """Create link -> target (dir). Replace broken or mismatched symlinks."""
    target_res = target.resolve()
    if link.is_symlink():
        try:
            if link.resolve(strict=True) == target_res:
                return
        except (FileNotFoundError, OSError):
            pass
        link.unlink(missing_ok=True)
    elif link.exists():
        raise FileExistsError(
            f"{link} exists and is not a symlink; remove it or use a different --out-dir."
        )
    link.symlink_to(target_res, target_is_directory=True)


def _resolve_ai8x_path(
    cli: str | None,
    env_key: str,
    default_path: Path,
    *,
    flag_name: str,
    label: str,
) -> Path:
    """First existing path wins: CLI flag, then env, then *default_path* (testMax layout)."""
    attempts: list[tuple[str, Path]] = []
    if cli and str(cli).strip():
        attempts.append((flag_name, Path(cli).expanduser()))
    env_raw = (os.environ.get(env_key) or "").strip()
    if env_raw:
        attempts.append((env_key, Path(env_raw).expanduser()))

    for where, raw in attempts:
        try:
            resolved = raw.resolve()
        except OSError:
            print(f"Note: {where}={raw!r} could not be resolved; skipping.", file=sys.stderr)
            continue
        if resolved.exists():
            return resolved
        print(f"Note: {where}={raw!r} not found on this machine; skipping.", file=sys.stderr)

    print(f"Using default {label}: {default_path}", file=sys.stderr)
    return default_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-csv", required=True,
                        help="HAWQ candidate CSV from export_hawq_candidates.py.")
    parser.add_argument("--input-lengths", type=int, nargs="+",
                        default=[101, 91, 81, 71, 61, 51, 41, 31, 21, 11, 5])
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers for ai8x train/eval calls.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--z-score", type=float, default=2.0)
    parser.add_argument("--device", default="MAX78002")
    parser.add_argument("--data-dir", default="data/indoor_environment")
    parser.add_argument("--ai8x-training-root", default=None,
                        help="Path to ai8x-training. Tried in order with env "
                        "AI8X_TRAINING_ROOT, then <parent-of-Indoor-repo>/testMax/ai8x-training "
                        "(first path that exists wins; stale paths are skipped).")
    parser.add_argument("--ai8x-synthesis-root", default=None,
                        help="Path to ai8x-synthesis. Tried in order with env "
                        "AI8X_SYNTHESIS_ROOT, then <parent-of-Indoor-repo>/testMax/ai8x-synthesis "
                        "(first path that exists wins; stale paths are skipped).")
    parser.add_argument("--out-dir", default="hawq_qat_sweep_out")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print planned workload without running QAT.")
    return parser.parse_args()


args = parse_args()
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
_default_training = REPO_ROOT.parent / "testMax" / "ai8x-training"
_default_synthesis = REPO_ROOT.parent / "testMax" / "ai8x-synthesis"
AI8X_TRAINING_ROOT = _resolve_ai8x_path(
    args.ai8x_training_root,
    "AI8X_TRAINING_ROOT",
    _default_training,
    flag_name="--ai8x-training-root",
    label="ai8x-training",
)
AI8X_SYNTHESIS_ROOT = _resolve_ai8x_path(
    args.ai8x_synthesis_root,
    "AI8X_SYNTHESIS_ROOT",
    _default_synthesis,
    flag_name="--ai8x-synthesis-root",
    label="ai8x-synthesis",
)
OUTPUT_DIR = Path(args.out_dir)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = SCRIPT_DIR / OUTPUT_DIR

LOGS_DIR = OUTPUT_DIR / "logs"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
POLICY_DIR = OUTPUT_DIR / "policies"
RUN_CWD = OUTPUT_DIR

DATA_DIR = Path(args.data_dir)
if not DATA_DIR.is_absolute():
    DATA_DIR = AI8X_TRAINING_ROOT / DATA_DIR
TRAIN_PY = AI8X_TRAINING_ROOT / "train.py"
QUANTIZE_PY = AI8X_SYNTHESIS_ROOT / "quantize.py"
SCHEDULE_YAML = AI8X_TRAINING_ROOT / "policies" / "schedule-indoor-env.yaml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
POLICY_DIR.mkdir(parents=True, exist_ok=True)

models_link = RUN_CWD / "models"
models_target = AI8X_TRAINING_ROOT / "models"
_ensure_dir_symlink(models_link, models_target)

datasets_link = RUN_CWD / "datasets"
datasets_target = AI8X_TRAINING_ROOT / "datasets"
_ensure_dir_symlink(datasets_link, datasets_target)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    runtime_home = OUTPUT_DIR / "runtime_home"
    mpl_config = OUTPUT_DIR / "mplconfig"
    runtime_home.mkdir(parents=True, exist_ok=True)
    mpl_config.mkdir(parents=True, exist_ok=True)
    (runtime_home / ".pyffmpeg").mkdir(parents=True, exist_ok=True)
    python_paths = [
        str(AI8X_TRAINING_ROOT),
        str(AI8X_TRAINING_ROOT / "distiller"),
    ]
    env["PYTHONPATH"] = ":".join(python_paths + [env.get("PYTHONPATH", "")])
    env["HOME"] = str(runtime_home)
    env["MPLCONFIGDIR"] = str(mpl_config)
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    if torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def run_cmd_tee(cmd_list: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> None:
    cmd = " ".join(str(item) for item in cmd_list)
    tee_cmd = f"set -o pipefail; {cmd} 2>&1 | tee {log_path}"
    subprocess.run(tee_cmd, shell=True, executable="/bin/bash", cwd=str(cwd), env=env, check=True)


def normalize_config(config: str) -> str:
    config = str(config).strip()
    if config.startswith("INT_"):
        config = "INT " + config[len("INT_"):]
    return config.replace("_", "-")


def parse_config_bits(config: str) -> tuple[int, int, int, int]:
    config = normalize_config(config)
    if not config.startswith("INT "):
        raise ValueError(f"Invalid config {config!r}; expected e.g. INT 8-8-2-2")
    bits = tuple(int(part) for part in config[len("INT "):].split("-"))
    if len(bits) != 4:
        raise ValueError(f"Invalid config {config!r}; expected four bit-widths")
    return bits


def config_name(bits: tuple[int, int, int, int]) -> str:
    return f"INT {bits[0]}-{bits[1]}-{bits[2]}-{bits[3]}"


def load_candidate_configs(path: Path, input_lengths: set[int]) -> dict[int, list[tuple[int, tuple[int, int, int, int], str]]]:
    df = pd.read_csv(path)
    by_length: dict[int, list[tuple[int, tuple[int, int, int, int], str]]] = {}
    for row_idx, row in df.iterrows():
        in_len = int(row.get("input_length", row.get("alpha")))
        if in_len not in input_lengths:
            continue
        if "config" in row and not pd.isna(row["config"]):
            bits = parse_config_bits(row["config"])
        else:
            bits = (
                int(row["conv1_bits"]),
                int(row["conv2_bits"]),
                int(row["fc1_bits"]),
                int(row["fc2_bits"]),
            )
        item = (int(row_idx) + 1, bits, config_name(bits))
        by_length.setdefault(in_len, [])
        if item not in by_length[in_len]:
            by_length[in_len].append(item)
    return by_length


def validate_inputs() -> None:
    required = {
        "ai8x training root": AI8X_TRAINING_ROOT,
        "ai8x synthesis root": AI8X_SYNTHESIS_ROOT,
        "train.py": TRAIN_PY,
        "quantize.py": QUANTIZE_PY,
        "schedule-indoor-env.yaml": SCHEDULE_YAML,
        "data directory": DATA_DIR,
        "candidate CSV": Path(args.configs_csv),
        "models link": models_link,
        "datasets link": datasets_link,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  - " + "\n  - ".join(missing))


def write_policy_yaml(path: Path, bits: tuple[int, int, int, int]) -> None:
    b1, b2, b3, b4 = bits
    policy = {
        "start_epoch": 8,
        "weight_bits": 8,
        "shift_quantile": 0.95,
        "outlier_removal_z_score": float(args.z_score),
        "overrides": {
            "conv1": {"weight_bits": int(b1)},
            "conv2": {"weight_bits": int(b2)},
            "fc1": {"weight_bits": int(b3)},
            "fc2": {"weight_bits": int(b4)},
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(policy, handle, sort_keys=False)


def detect_weight_bits_label(ckpt_path: str) -> tuple[str, str]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", {})
        bits_found = set()
        for key, value in sd.items():
            if key.endswith("weight_bits"):
                try:
                    bit = int(float(value.detach().cpu().numpy()))
                except Exception:
                    try:
                        bit = int(value)
                    except Exception:
                        continue
                if bit:
                    bits_found.add(bit)
        if len(bits_found) == 1:
            bit = next(iter(bits_found))
            return f"INT{bit}", f"_q{bit}"
        if len(bits_found) > 1:
            return "mixed", "_qmixed"
    except Exception:
        pass
    return "INT8", "_q8"


def extract_metric_from_log(log_file: Path) -> float | None:
    try:
        content = log_file.read_text(encoding="utf-8")
    except Exception:
        return None
    patterns = [
        r"==>\s*Top1:\s*(\d+\.?\d*)",
        r"Test.*?Top1.*?(\d+\.?\d*)",
        r"Prec@1\s+(\d+\.?\d*)",
        r"Test.*?Accuracy.*?(\d+\.?\d*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except Exception:
                continue
    return None


def write_results(rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "hawq_qat_sweep_results.csv", index=False)
    succ = df[df["status"] == "success"] if "status" in df.columns else pd.DataFrame()
    if len(succ) > 0:
        summary = succ.groupby(["input_length", "config"]).agg(
            runs=("test_accuracy", "count"),
            mean_acc=("test_accuracy", "mean"),
            std_acc=("test_accuracy", "std"),
            total_time=("train_seconds", "sum"),
        ).reset_index()
        summary["conv1_bits"] = summary["config"].str.extract(r"INT (\d+)-")[0].astype(int)
        summary["conv2_bits"] = summary["config"].str.extract(r"-(\d+)-")[0].astype(int)
        summary["fc1_bits"] = summary["config"].str.extract(r"-(\d+)-(\d+)$")[0].astype(int)
        summary["fc2_bits"] = summary["config"].str.extract(r"-(\d+)$")[0].astype(int)
        summary.sort_values(by=["mean_acc", "input_length"], ascending=[False, False], inplace=True)
    else:
        summary = pd.DataFrame(columns=["input_length", "config", "runs", "mean_acc", "std_acc"])
    summary.to_csv(OUTPUT_DIR / "hawq_qat_sweep_summary.csv", index=False)


def append_failure(rows: list[dict[str, object]], in_len: int, seed: int, cfg_idx: int,
                   bits: tuple[int, int, int, int], cfg_name: str, status: str,
                   elapsed: float) -> list[dict[str, object]]:
    rows.append({
        "input_length": in_len,
        "seed": seed,
        "candidate_index": cfg_idx,
        "config": cfg_name,
        "conv1_bits": bits[0],
        "conv2_bits": bits[1],
        "fc1_bits": bits[2],
        "fc2_bits": bits[3],
        "status": status,
        "test_accuracy": 0.0,
        "train_seconds": elapsed,
    })
    write_results(rows)
    return rows


def run_one(in_len: int, cfg_idx: int, bits: tuple[int, int, int, int],
            cfg_name: str, seed: int, rows: list[dict[str, object]],
            env: dict[str, str]) -> list[dict[str, object]]:
    start_time = time.time()
    bit_tag = "_".join(str(bit) for bit in bits)
    run_name = f"indoor_hawq_qat_seed_{seed}__L{in_len}__{bit_tag}"
    policy_file = POLICY_DIR / f"qat_policy_hawq_L{in_len}_seed_{seed}_{bit_tag}.yaml"
    write_policy_yaml(policy_file, bits)

    train_log = LOGS_DIR / f"train_L{in_len}_cand{cfg_idx:03d}_seed_{seed}.log"
    train_cmd = [
        PYTHON_EXECUTABLE, str(TRAIN_PY),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--optimizer", "Adam",
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--use-bias",
        "--deterministic",
        "--model", "ai85indoorenvnetv2",
        "--dataset", "IndoorEnvironment_1D",
        "--data", str(DATA_DIR),
        "--input-1d-length", str(in_len),
        "--compress", str(SCHEDULE_YAML),
        "--qat-policy", str(policy_file),
        "--device", args.device,
        "--compiler-mode", "none",
        "--out-dir", str(OUTPUT_DIR),
        "--seed", str(seed),
        "--name", run_name,
    ]
    print(f"Training QAT: {cfg_name} (L={in_len}, seed={seed})")
    try:
        run_cmd_tee(train_cmd, train_log, RUN_CWD, env)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: training failed for {cfg_name}, L={in_len}, seed={seed}, code={exc.returncode}")
        return append_failure(rows, in_len, seed, cfg_idx, bits, cfg_name, "train_failed", time.time() - start_time)

    run_dirs = sorted(glob.glob(str(OUTPUT_DIR / f"{run_name}*")))
    if not run_dirs:
        return append_failure(rows, in_len, seed, cfg_idx, bits, cfg_name, "no_run_dir", time.time() - start_time)
    run_dir = Path(run_dirs[-1])

    best_qat = None
    for pattern in ("*_qat_best.pth.tar", "*qat_best*.pth.tar", "*best*.pth.tar"):
        found = sorted(run_dir.glob(pattern))
        if found:
            best_qat = found[-1]
            break
    if best_qat is None:
        return append_failure(rows, in_len, seed, cfg_idx, bits, cfg_name, "no_best_qat", time.time() - start_time)

    label, suffix = detect_weight_bits_label(str(best_qat))
    quant_path = Path(str(best_qat)[:-8] + f"{suffix}.pth.tar") if str(best_qat).endswith(".pth.tar") else run_dir / f"{best_qat.stem}{suffix}.pth.tar"
    quant_log = LOGS_DIR / f"quant_L{in_len}_cand{cfg_idx:03d}_seed_{seed}.log"
    quant_cmd = [
        PYTHON_EXECUTABLE, str(QUANTIZE_PY), str(best_qat), str(quant_path),
        "--device", args.device, "-v",
    ]
    print(f"Quantizing ({label}) -> {quant_path.name}")
    try:
        run_cmd_tee(quant_cmd, quant_log, RUN_CWD, env)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: quantize failed for {cfg_name}, L={in_len}, seed={seed}, code={exc.returncode}")
        return append_failure(rows, in_len, seed, cfg_idx, bits, cfg_name, "quant_failed", time.time() - start_time)

    eval_log = LOGS_DIR / f"eval_L{in_len}_cand{cfg_idx:03d}_seed_{seed}.log"
    eval_cmd = [
        PYTHON_EXECUTABLE, str(TRAIN_PY),
        "--deterministic",
        "--workers", str(args.workers),
        "--optimizer", "Adam",
        "--model", "ai85indoorenvnetv2",
        "--dataset", "IndoorEnvironment_1D",
        "--data", str(DATA_DIR),
        "--input-1d-length", str(in_len),
        "--device", args.device,
        "--qat-policy", str(policy_file),
        "--use-bias",
        "--weight-decay", str(args.weight_decay),
        "--evaluate",
        "--exp-load-weights-from", str(quant_path),
        "-8",
        "--print-freq", "10",
        "--compiler-mode", "none",
        "--out-dir", str(OUTPUT_DIR),
        "--seed", str(seed),
        "--name", f"{run_name}_{label.lower()}_eval",
    ]
    print(f"Evaluating ({label} weights, INT8 activations)")
    try:
        run_cmd_tee(eval_cmd, eval_log, RUN_CWD, env)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: evaluation failed for {cfg_name}, L={in_len}, seed={seed}, code={exc.returncode}")
        return append_failure(rows, in_len, seed, cfg_idx, bits, cfg_name, "eval_failed", time.time() - start_time)

    test_acc = extract_metric_from_log(eval_log)
    elapsed = time.time() - start_time
    try:
        shutil.copy2(quant_path, CHECKPOINTS_DIR / f"{run_name}_{quant_path.name}")
    except Exception:
        pass

    rows.append({
        "input_length": in_len,
        "seed": seed,
        "candidate_index": cfg_idx,
        "config": cfg_name,
        "conv1_bits": bits[0],
        "conv2_bits": bits[1],
        "fc1_bits": bits[2],
        "fc2_bits": bits[3],
        "label_detected": label,
        "test_accuracy": test_acc or 0.0,
        "train_seconds": elapsed,
        "status": "success" if test_acc is not None else "no_metric",
    })
    write_results(rows)
    print(f"  {cfg_name}: {test_acc if test_acc is not None else 0.0:.2f}% (took {elapsed:.1f}s)")
    return rows


def main() -> None:
    input_lengths = sorted(set(args.input_lengths), reverse=True)
    seeds = [args.start_seed + idx for idx in range(args.num_seeds)]
    candidates = load_candidate_configs(Path(args.configs_csv), set(input_lengths))
    validate_inputs()

    total_candidates = sum(len(candidates.get(in_len, [])) for in_len in input_lengths)
    total_runs = total_candidates * len(seeds)
    print("\n" + "=" * 80)
    print("HAWQ-Reduced Mixed-Precision QAT Sweep")
    print("=" * 80)
    print(f"Input lengths: {input_lengths}")
    print(f"Seeds: {seeds}")
    print(f"Candidate source: {args.configs_csv}")
    print(f"Candidate count: {total_candidates}")
    print(f"Total QAT runs: {total_runs}")
    print(f"DataLoader workers: {args.workers}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)

    if args.dry_run:
        print("\nDry run only. Candidate counts by input length:")
        for in_len in input_lengths:
            configs_for_length = candidates.get(in_len, [])
            sample = ", ".join(name for _, _, name in configs_for_length[:5])
            suffix = " ..." if len(configs_for_length) > 5 else ""
            print(f"  L={in_len}: {len(configs_for_length)} configs"
                  f"{f' ({sample}{suffix})' if sample else ''}")
        print("\nNo QAT commands were executed.")
        return

    env = build_env()
    rows: list[dict[str, object]] = []
    write_results(rows)
    for in_len in input_lengths:
        configs_for_length = candidates.get(in_len, [])
        if not configs_for_length:
            print(f"No HAWQ candidates for L={in_len}; skipping.")
            continue
        print("\n" + "#" * 80)
        print(f"INPUT LENGTH: {in_len} ({len(configs_for_length)} candidates)")
        print("#" * 80)
        for cfg_idx, bits, cfg_name in configs_for_length:
            for seed in seeds:
                rows = run_one(in_len, cfg_idx, bits, cfg_name, seed, rows, env)

    write_results(rows)
    summary = pd.read_csv(OUTPUT_DIR / "hawq_qat_sweep_summary.csv")
    print("\n" + "=" * 80)
    print("HAWQ QAT Sweep Completed")
    print("=" * 80)
    print(summary.head(20).to_string(index=False) if len(summary) else "No successful runs.")
    print(f"\nResults saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
