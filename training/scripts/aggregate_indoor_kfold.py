#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch


def find_best_checkpoint(run_dir: str) -> Optional[str]:
    """Return a path to a 'best' checkpoint inside run_dir, if any.

    Tries several common patterns used by this repo and Distiller:
      - best.pth.tar
      - <run_name>_best.pth.tar
      - any file matching '*best*.pth.tar'
    """
    # 1) best.pth.tar
    p = os.path.join(run_dir, 'best.pth.tar')
    if os.path.isfile(p):
        return p

    # 2) <run_name>_best.pth.tar
    base = os.path.basename(run_dir)
    p = os.path.join(run_dir, f'{base}_best.pth.tar')
    if os.path.isfile(p):
        return p

    # 3) any '*best*.pth.tar'
    candidates = [os.path.join(run_dir, f) for f in os.listdir(run_dir)
                  if 'best' in f and f.endswith('.pth.tar')]
    if candidates:
        # choose newest by mtime
        candidates.sort(key=lambda s: os.path.getmtime(s), reverse=True)
        return candidates[0]
    return None


def extract_top1_from_checkpoint(ckpt_path: str) -> Optional[float]:
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"WARN: failed to load '{ckpt_path}': {e}")
        return None

    extras = checkpoint.get('extras', {}) if isinstance(checkpoint, dict) else {}
    # Prefer validation best, fallback to current
    for key in ('best_top1', 'current_top1'):
        v = extras.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return None


def collect_runs(logs_root: str, prefix: Optional[str]) -> List[Tuple[str, str]]:
    """Return list of (run_dir, checkpoint_path) filtered by prefix if provided."""
    if not os.path.isdir(logs_root):
        return []

    runs: List[Tuple[str, str]] = []
    for entry in os.listdir(logs_root):
        run_dir = os.path.join(logs_root, entry)
        if not os.path.isdir(run_dir):
            continue
        if prefix and prefix not in entry:
            continue
        ckpt = find_best_checkpoint(run_dir)
        if ckpt:
            runs.append((run_dir, ckpt))
    return runs


def parse_test_best_top1(log_path: str) -> Optional[float]:
    """Parse 'test (best)' Top1 value from the run log.

    Strategy:
      - find the last occurrence of a line containing '--- test (best)'
      - from there, scan forward to the first line like '==> Top1: <val>' and return it
      - fallback: return the last '==> Top1: <val>' that appears after any '--- test'
    """
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return None

    test_best_idx = None
    test_any_idx = None
    for i, line in enumerate(lines):
        if '--- test (best)' in line:
            test_best_idx = i
        if '--- test (' in line:
            test_any_idx = i

    def extract_after(start_idx: int) -> Optional[float]:
        top1_re = re.compile(r"==> Top1:\s*([0-9]+\.?[0-9]*)")
        for j in range(start_idx + 1, len(lines)):
            m = top1_re.search(lines[j])
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
        return None

    if test_best_idx is not None:
        v = extract_after(test_best_idx)
        if v is not None:
            return v
    if test_any_idx is not None:
        v = extract_after(test_any_idx)
        if v is not None:
            return v
    return None


def main():
    ap = argparse.ArgumentParser(description='Aggregate K-Fold results (mean/std of Top1).')
    ap.add_argument('--logs-root', default=os.path.join(os.path.dirname(__file__), '..', 'logs'),
                    help='Root directory containing run subfolders (default: ../logs)')
    ap.add_argument('--prefix', default='indoor_k',
                    help='Filter runs by name containing this prefix (default: indoor_k)')
    args = ap.parse_args()

    logs_root = os.path.abspath(args.logs_root)
    runs = collect_runs(logs_root, prefix=args.prefix)
    if not runs:
        print(f"No runs found under '{logs_root}' with prefix '{args.prefix}'.")
        sys.exit(1)

    values = []
    print('Runs considered:')
    for run_dir, ckpt in sorted(runs):
        run_name = os.path.basename(run_dir)
        log_path = os.path.join(run_dir, run_name + '.log')
        test_top1 = parse_test_best_top1(log_path)
        if test_top1 is None:
            # fallback to checkpoint extras (validation top1)
            test_top1 = extract_top1_from_checkpoint(ckpt)
            source = os.path.basename(ckpt) if test_top1 is not None else 'N/A'
        else:
            source = os.path.basename(log_path)

        if test_top1 is not None:
            values.append(test_top1)
            print(f"- {run_name}: Top1={test_top1:.3f} ({source})")
        else:
            print(f"- {run_name}: Top1=N/A ({source})")

    if not values:
        print('No Top1 values extracted.')
        sys.exit(2)

    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    print('\nAggregate:')
    print(f"- N={len(arr)}  Mean Top1={mean:.3f}  Std={std:.3f}")


if __name__ == '__main__':
    main()


