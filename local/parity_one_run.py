#!/usr/bin/env python3
"""
Single (input_length, config, seed) parity check: run two local pipelines that mirror
train_indoor_1D_mixed_sweep.py vs train_indoor_1D_hawq_qat_sweep.py subprocess CLIs,
while forcing the same:

  - cwd (ai8x-training root)
  - interpreter (sys.executable)
  - PYTHONPATH (repo root + vendored distiller)
  - absolute --data
  - same quantize.py

Outputs under --out-root/<mixed_style|hawq_style>/; prints Top1 from each eval log.

By default deletes those two subdirectories under --out-root before each run so stale
train/eval folders cannot confuse checkpoint discovery. Pass --keep-output to skip.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-len", type=int, default=91)
    p.add_argument("--config", type=str, default="INT 8-8-2-2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--training-root",
        type=Path,
        default=Path(os.environ.get("AI8X_TRAINING_ROOT", "/shared/b00090279/testMax/ai8x-training")),
    )
    p.add_argument(
        "--synthesis-root",
        type=Path,
        default=Path(os.environ.get("AI8X_SYNTHESIS_ROOT", "/shared/b00090279/testMax/ai8x-synthesis")),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Defaults to <training-root>/data/indoor_environment",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Defaults to <this>/parity_runs (next to this script)",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--device", type=str, default="MAX78002")
    p.add_argument("--z-score", type=float, default=2.0)
    p.add_argument("--hawq-workers", type=int, default=4)
    p.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not remove mixed_style/ and hawq_style/ under --out-root before running.",
    )
    return p.parse_args()


def normalize_config(config: str) -> str:
    c = config.strip()
    if c.startswith("INT_"):
        c = "INT " + c[4:]
    return c.replace("_", "-")


def parse_bits(config: str) -> tuple[int, int, int, int]:
    c = normalize_config(config)
    if not c.startswith("INT "):
        raise ValueError(f"expected INT b-b-b-b, got {config!r}")
    parts = c[len("INT ") :].split("-")
    if len(parts) != 4:
        raise ValueError(config)
    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))


def write_policy_yaml(path: Path, bits: tuple[int, int, int, int], z_score: float) -> None:
    b1, b2, b3, b4 = bits
    policy = {
        "start_epoch": 8,
        "weight_bits": 8,
        "shift_quantile": 0.95,
        "outlier_removal_z_score": float(z_score),
        "overrides": {
            "conv1": {"weight_bits": int(b1)},
            "conv2": {"weight_bits": int(b2)},
            "fc1": {"weight_bits": int(b3)},
            "fc2": {"weight_bits": int(b4)},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(policy, handle, sort_keys=False)


def build_env(training_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    distiller_path = str(training_root / "distiller")
    env["PYTHONPATH"] = f"{training_root}:{distiller_path}:{env.get('PYTHONPATH', '')}"
    if torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def run_cmd(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _progress(msg: str) -> None:
    print(msg, flush=True)


def detect_weight_bits_label(ckpt_path: Path) -> tuple[str, str]:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = ckpt.get("state_dict", {})
        bits_found: set[int] = set()
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


def find_best_qat(output_dir: Path, run_name: str) -> Path:
    # Train run dirs look like:  <run_name>___<timestamp>
    # Eval dirs look like:       <run_name>_<label>_eval___<timestamp>
    # A naive f"{run_name}*" glob sorts eval after train and breaks sorted(...)[-1] when
    # a previous eval folder is still present under out_dir.
    train_glob = sorted(glob.glob(str(output_dir / f"{run_name}___*")))
    if train_glob:
        run_dir = Path(train_glob[-1])
    else:
        candidates = sorted(glob.glob(str(output_dir / f"{run_name}*")))
        train_dirs = [p for p in candidates if "_eval_" not in Path(p).name]
        if not train_dirs:
            raise FileNotFoundError(
                f"No train run dir for {run_name!r} under {output_dir} "
                f"(expected {run_name}___* or * without _eval_)"
            )
        run_dir = Path(train_dirs[-1])
    for pattern in ("*_qat_best.pth.tar", "*qat_best*.pth.tar", "*best*.pth.tar"):
        found = sorted(run_dir.glob(pattern))
        if found:
            return found[-1]
    raise FileNotFoundError(f"No QAT best checkpoint in {run_dir}")


def extract_top1(log_path: Path) -> float | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    patterns = [
        r"==>\s*Top1:\s*(\d+\.?\d*)",
        r"Test.*?Top1.*?(\d+\.?\d*)",
        r"Prec@1\s+(\d+\.?\d*)",
        r"Test.*?Accuracy.*?(\d+\.?\d*)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    return None


def run_pipeline(
    *,
    training_root: Path,
    data_dir: Path,
    quantize_py: Path,
    out_dir: Path,
    in_len: int,
    bits: tuple[int, int, int, int],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    z_score: float,
    hawq_workers: int,
    mixed_style: bool,
) -> float | None:
    bit_tag = "_".join(str(b) for b in bits)
    if mixed_style:
        run_name = f"indoor_mixed_seed_{seed}__L{in_len}__{bit_tag}"
        policy_path = out_dir / "policies" / f"qat_policy_parity_mixed_L{in_len}_seed_{seed}_{bit_tag}.yaml"
    else:
        run_name = f"indoor_hawq_qat_seed_{seed}__L{in_len}__{bit_tag}"
        policy_path = out_dir / "policies" / f"qat_policy_parity_hawq_L{in_len}_seed_{seed}_{bit_tag}.yaml"

    logs = out_dir / "logs"
    checkpoints = out_dir / "checkpoints"
    for d in (out_dir, logs, checkpoints, policy_path.parent):
        d.mkdir(parents=True, exist_ok=True)

    write_policy_yaml(policy_path, bits, z_score)
    env = build_env(training_root)
    py = sys.executable
    train_py = training_root / "train.py"
    tag = "mixed_style" if mixed_style else "hawq_style"
    _progress("")
    _progress(f"========== [{tag}] L={in_len} seed={seed} bits={bit_tag} ==========")
    _progress(f"[{tag}] out_dir: {out_dir}")
    schedule_rel = "policies/schedule-indoor-env.yaml"
    schedule_abs = training_root / "policies" / "schedule-indoor-env.yaml"

    train_cmd: list[str] = [
        py,
        str(train_py),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    if not mixed_style:
        train_cmd += ["--workers", str(hawq_workers)]
    compress_arg = schedule_rel if mixed_style else str(schedule_abs)
    train_cmd += [
        "--optimizer",
        "Adam",
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--use-bias",
        "--deterministic",
        "--model",
        "ai85indoorenvnetv2",
        "--dataset",
        "IndoorEnvironment_1D",
        "--data",
        str(data_dir),
        "--input-1d-length",
        str(in_len),
        "--compress",
        compress_arg,
        "--qat-policy",
        str(policy_path),
        "--device",
        device,
        "--compiler-mode",
        "none",
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--name",
        run_name,
    ]

    train_log = logs / f"train_L{in_len}_parity_seed_{seed}.log"
    _progress(f"[{tag}] TRAIN starting (stdout/stderr → {train_log}) — often 10+ min per epoch; no terminal spam until done.")
    run_cmd(train_cmd, train_log, training_root, env)
    _progress(f"[{tag}] TRAIN finished.")

    best_qat = find_best_qat(out_dir, run_name)
    label, suffix = detect_weight_bits_label(best_qat)
    stem = str(best_qat)
    if stem.endswith(".pth.tar"):
        quant_path = Path(stem[:-8] + f"{suffix}.pth.tar")
    else:
        quant_path = best_qat.parent / f"{best_qat.stem}{suffix}.pth.tar"

    quant_log = logs / f"quant_L{in_len}_parity_seed_{seed}.log"
    quant_cmd = [py, str(quantize_py), str(best_qat), str(quant_path), "--device", device, "-v"]
    _progress(f"[{tag}] QUANTIZE → {quant_log}")
    run_cmd(quant_cmd, quant_log, training_root, env)
    _progress(f"[{tag}] QUANTIZE finished.")

    eval_log = logs / f"eval_L{in_len}_parity_seed_{seed}.log"
    if mixed_style:
        eval_cmd = [
            py,
            str(train_py),
            "--deterministic",
            "--optimizer",
            "Adam",
            "--model",
            "ai85indoorenvnetv2",
            "--dataset",
            "IndoorEnvironment_1D",
            "--data",
            str(data_dir),
            "--input-1d-length",
            str(in_len),
            "--device",
            device,
            "--qat-policy",
            str(policy_path),
            "--use-bias",
            "--weight-decay",
            str(weight_decay),
            "--evaluate",
            "--exp-load-weights-from",
            str(quant_path),
            "-8",
            "--confusion",
            "--print-freq",
            "10",
            "--save-sample",
            "10",
            "--compiler-mode",
            "none",
            "--out-dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--name",
            f"{run_name}_{label.lower()}_eval",
        ]
    else:
        eval_cmd = [
            py,
            str(train_py),
            "--deterministic",
            "--workers",
            str(hawq_workers),
            "--optimizer",
            "Adam",
            "--model",
            "ai85indoorenvnetv2",
            "--dataset",
            "IndoorEnvironment_1D",
            "--data",
            str(data_dir),
            "--input-1d-length",
            str(in_len),
            "--device",
            device,
            "--qat-policy",
            str(policy_path),
            "--use-bias",
            "--weight-decay",
            str(weight_decay),
            "--evaluate",
            "--exp-load-weights-from",
            str(quant_path),
            "-8",
            "--print-freq",
            "10",
            "--compiler-mode",
            "none",
            "--out-dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--name",
            f"{run_name}_{label.lower()}_eval",
        ]

    _progress(f"[{tag}] EVAL → {eval_log}")
    run_cmd(eval_cmd, eval_log, training_root, env)
    acc = extract_top1(eval_log)
    _progress(f"[{tag}] EVAL finished. Parsed Top1: {acc}")

    try:
        shutil.copy2(quant_path, checkpoints / f"{run_name}_{quant_path.name}")
    except OSError:
        pass

    return acc


def main() -> None:
    args = parse_args()
    training_root = args.training_root.resolve()
    synthesis_root = args.synthesis_root.resolve()
    data_dir = (args.data_dir or (training_root / "data" / "indoor_environment")).resolve()
    quantize_py = (synthesis_root / "quantize.py").resolve()
    script_dir = Path(__file__).resolve().parent
    out_root = (args.out_root or (script_dir / "parity_runs")).resolve()

    for name, path in (
        ("train.py", training_root / "train.py"),
        ("quantize.py", quantize_py),
        ("schedule", training_root / "policies" / "schedule-indoor-env.yaml"),
        ("data", data_dir),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    cfg_name = normalize_config(args.config)
    bits = parse_bits(cfg_name)
    print("Parity one-shot (unified env)")
    print(f"  training_root: {training_root}")
    print(f"  synthesis_root: {synthesis_root}")
    print(f"  quantize.py:   {quantize_py}")
    print(f"  data_dir:      {data_dir}")
    print(f"  python:        {sys.executable}")
    print(f"  case:          L={args.in_len} config={cfg_name} seed={args.seed}")
    print(f"  out_root:      {out_root}")
    print()
    _progress("Note: subprocess output goes only to per-phase *.log files under each out_dir;")
    _progress("      the terminal stays quiet until each phase completes (train is the long step).")
    _progress("")

    mixed_dir = out_root / "mixed_style"
    hawq_dir = out_root / "hawq_style"

    if not args.keep_output:
        for label, path in (("mixed_style", mixed_dir), ("hawq_style", hawq_dir)):
            if path.exists():
                _progress(f"Removing previous {label} outputs: {path}")
                shutil.rmtree(path)

    acc_mixed = run_pipeline(
        training_root=training_root,
        data_dir=data_dir,
        quantize_py=quantize_py,
        out_dir=mixed_dir,
        in_len=args.in_len,
        bits=bits,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        z_score=args.z_score,
        hawq_workers=args.hawq_workers,
        mixed_style=True,
    )
    acc_hawq = run_pipeline(
        training_root=training_root,
        data_dir=data_dir,
        quantize_py=quantize_py,
        out_dir=hawq_dir,
        in_len=args.in_len,
        bits=bits,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        z_score=args.z_score,
        hawq_workers=args.hawq_workers,
        mixed_style=False,
    )

    print("Results (Top1 % from eval logs)")
    print(f"  mixed_style eval Top1: {acc_mixed}")
    print(f"  hawq_style  eval Top1: {acc_hawq}")
    print()
    print("Logs:")
    print(f"  {mixed_dir / 'logs'}")
    print(f"  {hawq_dir / 'logs'}")


if __name__ == "__main__":
    main()
