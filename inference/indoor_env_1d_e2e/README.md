# indoor_env_1d_e2e — MAX78002 end-to-end latency bench

Firmware for **α = 91**, **INT 8-8-2-4** (`q8824`), derived from synthesized [`indoor_env_1d_91_q8824`](../indoor_env_1d_91_q8824/) with **on-MCU preprocessing** and a **simulated VNA buffer** in SRAM.

Full measurement model and paper alignment: [`docs/e2e_latency_measurement.md`](../../docs/e2e_latency_measurement.md).

## What this project measures

Each loop (default `E2E_NUM_RUNS = 100`):

1. **Untimed:** `fill_ctf_raw_from_recording()` — `memcpy` from Flash `const` recording → `ctf_raw[91][2]` in SRAM (bench stand-in for “sweep finished, raw CTF in RAM”).
2. **Timed —** `preprocess_ctf()` → global min–max → int8 into CNN input SRAM (`T_prep`, TMR0).
3. **Timed —** `cnn_start()` → wait for IRQ (`T_INP+INF`, `CNN_INFERENCE_TIMER` if enabled).
4. **Timed —** `softmax_layer()` (`T_act`).

`T_sense` and `T_link` are **not** on the EVKIT path; see the doc above.

### Per-stage latency (UART)

After all runs, `main.c` prints **min / max / mean / last** for each timed stage:

```text
--- Per-stage latency (TMR0 / CNN inference timer) ---
T_prep over 100 runs: min=... us, max=... us, mean=... us, last=... us
T_INP+INF over 100 runs: ...
T_act over 100 runs: ...
T_e2e (sum of last-run stages): ... us
```

- **`T_prep` / `T_act`:** `MXC_TMR0` around `preprocess_ctf()` and `softmax_layer()`.
- **`T_INP+INF`:** `cnn_time` from synthesis (`CNN_INFERENCE_TIMER`), or `MXC_TMR0` around `cnn_start()` if that define is absent.

Previously only **`T_prep`** stats and a single **last-run** CNN line were printed; all three stages are summarized now.

### Per-stage energy (external PMON)

GPIO triggers match the ADI synthesized demo (`cnn.h`):

| Pin | Macro | High during |
|-----|--------|-------------|
| **P1.6** | `SYS_START` / `SYS_COMPLETE` | `T_prep`, then `T_act` (M4F) |
| **P1.7** | `CNN_START` / `CNN_COMPLETE` | `T_INP+INF` (accelerator) |

Integrate supply current on the EVKIT power monitor during each high window (one prep pulse + one CNN pulse + one act pulse per classification loop). UART reminds you after the timing block; energy is **not** read on-chip.

### Live path (VNA → MCU)

Training-time `.mat` loading, grid splits, and labels stay on the PC. One live classification is only:

Assume the sweep is done and the interface has written complex CTF into MCU SRAM.

```text
  VNA + link                MAX78002
  ─────────                 ─────────────────────────────────────────
  RF sweep finishes    →    SRAM: float ctf_raw[α][2]   (raw, not CNN-ready)
                            │
                            ▼
                      Cortex-M4F (ARM)  — preprocess_ctf()     ← T_prep
                            │  • optional: center 101→91
                            │  • global min–max (fixed constants)
                            │  • float → int8 per Re / Im
                            │  • pack into CNN input SRAM
                            ▼
                      CNN accelerator      — cnn_start()           ← T_INP+INF
                            │  (ARM sleeps until IRQ)
                            ▼
                      Cortex-M4F (ARM)  — softmax_layer()        ← T_act
                            │
                            ▼
                      class + probabilities (UART / app)
```

On the EVKIT bench, the first step is emulated by untimed `fill_ctf_raw_from_recording()` (`memcpy` from Flash) instead of a VNA write.

## Synthesis vs. this tree

| Stage | Tool / artifact |
|--------|------------------|
| CNN graph, `cnn.c`, `weights.h`, `sampleoutput.h` | `ai8xize` / synthesis (`indoor_env_1d_91_q8824`) |
| E2E bench | Copy to `indoor_env_1d_e2e`, add `preprocess.c`, `ctf_recordings.h`, rewrite `main.c` |

Do **not** re-run synthesis for bench-only changes; re-synthesize only when the quantized network changes.

## Regenerate `ctf_recordings.h`

From the repo root (dataset lives at `training/data/indoor_environment/`; needs `scipy`):

```bash
# Default: classroom static, sweep 0, alpha=91 (center-crop like training)
python3 misc/export_ctf_raw_h.py

# Other environment / sweep index
python3 misc/export_ctf_raw_h.py \
  --mat-key CTF_lab_mov --index 42

# Placeholder only (no .mat)
python3 misc/export_ctf_raw_h.py --synthetic
```

If `python3` has no `scipy`, use `misc/.venv` after `python3 -m venv misc/.venv && misc/.venv/bin/pip install scipy numpy`.

Preprocess uses **training-set global** min/max constants in `preprocess.c` (must match the exported model’s normalization).

## Build and flash

Prerequisites: [MSDK](https://analogdevicesinc.github.io/msdk/USERGUIDE/) with `MAXIM_PATH` set, MAX78002 EVKIT.

```bash
cd inference/indoor_env_1d_e2e
make -r -j 8 TARGET=MAX78002 BOARD=EvKit_V1
```

Output: `build/indoor_env_1d_e2e.elf`.

Flash once over SWD (OpenOCD / VS Code “Flash” task). UART prints `T_prep` min/max/mean and optional known-answer check vs. `sampleoutput.h` (`E2E_RUN_KAT`, default on).

Optional defines (e.g. in `project.mk`):

- `E2E_NUM_RUNS` — benchmark iterations (default 100).
- `E2E_RUN_KAT=0` — skip `check_output()` after the loop.

## Source layout

| File | Role |
|------|------|
| `main.c` | Bench loop, timers, UART |
| `preprocess.c` / `preprocess.h` | `ctf_raw`, `preprocess_ctf`, Flash → SRAM fill |
| `ctf_recordings.h` | `const` recorded sweeps (generated) |
| `cnn.c`, `weights.h`, `sampleoutput.h` | From synthesis (unchanged graph) |

## Related docs

- [`docs/e2e_latency_measurement.md`](../../docs/e2e_latency_measurement.md) — latency budget, memory map, UART rules
- [`docs/max78002_m4f_cnn_examples.md`](../../docs/max78002_m4f_cnn_examples.md) — ADI MSDK reference flows
