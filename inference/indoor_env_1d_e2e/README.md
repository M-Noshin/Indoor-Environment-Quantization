# indoor_env_1d_e2e — MAX78002 end-to-end latency bench

Firmware for **on-MCU preprocessing** + **e2e latency/energy bench** (simulated VNA buffer in SRAM). This tree is **retargeted per α / MPQ model**; the active synthesized source and α are defined by `E2E_SYNTH_SOURCE` and `CTF_ALPHA` in `preprocess_config.h`.

Full measurement model and paper alignment: [`docs/e2e_latency_measurement.md`](../../docs/e2e_latency_measurement.md).
EVKIT PMON mode details follow ADI's [`MAX7800x Power Monitor and Energy Benchmarking Guide`](https://github.com/analogdevicesinc/MaximAI_Documentation/blob/main/Guides/MAX7800x%20Power%20Monitor%20and%20Energy%20Benchmarking%20Guide.md).

## What this project measures

Each loop (default `E2E_NUM_RUNS = 100`):

1. **Untimed:** `fill_ctf_raw_from_recording()` — `memcpy` from Flash `const` recording → `ctf_raw[CTF_ALPHA][2]` in SRAM (bench stand-in for “sweep finished, raw CTF in RAM”).
2. **Timed —** `preprocess_ctf_pack()` → global min–max → packed int8 words in SRAM (`T_prep`, TMR0).
3. **Timed —** fused `preprocess_ctf_load_input()` + `cnn_start()` → packed int8 words into CNN input SRAM, then wait for IRQ (`T_INP+INF`, TMR0).
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

- **`T_prep` / `T_INP+INF` / `T_act`:** `MXC_TMR0` around `preprocess_ctf_pack()`, fused `preprocess_ctf_load_input()` + `cnn_start()`, and `softmax_layer()`.

Previously only **`T_prep`** stats and a single **last-run** CNN line were printed; all three stages are summarized now.

### ADI-style PMON blocks (compare with `indoor_env_1d_*_q8824`)

When `E2E_PMON_ADI_BLOCKS=1` (default), firmware runs the **same characterization sequence** as the synthesis demos **before** the e2e loop:

| Step | Pin | What (× `E2E_PMON_ADI_REPS`, default 100) |
|------|-----|---------------------------------------------|
| 1. Idle baseline | P1.6 | 1 s delay (`SYS_START` … `SYS_COMPLETE`) |
| 2. Weight load | P1.7 | `cnn_load_weights()` |
| 3. Input load (ADI) | P1.7 | `load_input()` — `memcpy32` from `sampledata.h` |
| 4. Fused load + inference | P1.7 | `load_input()` + `cnn_start()` (ADI “input + inf” window) |

`MXC_TMR_Delay` (`E2E_PMON_SETTLE_TICKS`, default 500000) runs between blocks. UART prints a monitor guide; energy is read on the **EVKIT display**, not on-chip.

Disable ADI blocks for e2e-only runs: `PROJ_CFLAGS += -DE2E_PMON_ADI_BLOCKS=0` in `project.mk`.

For an EVKIT-screen-only comparison, build with `E2E_PMON_ADI_ONLY=1`; firmware stops after these 4 windows.

For e2e stage energy, use EVKIT **System Power Mode** in separate runs. Build one mode at a time with `E2E_SYS_PMON_MODE=N`; firmware skips the ADI CNN Power Mode sequence, emits one `SYS_START` / `SYS_COMPLETE` measurement window on P1.6, then stops. On PMON v2.0, the System Power `T` and `E` fields can be corrupted by the timer flag issue described below. For M4F stages, use the PMON `mW` field from a sustained run and combine it with UART stage latency.

During these System Power Mode windows, firmware emits the P1.6 `SYS_START` / `SYS_COMPLETE` measurement window. If the PMON serial/display `T` does not roughly match the UART `Firmware SYS PMON window`, the PMON missed or misinterpreted the intended window; verify the PMON is in triggered/System Power Mode and that JP18/JP19 are installed.

For System Power modes, the firmware configures P1.6/P1.7, forces both low, waits the arming delay, then emits the P1.6 SYS-window pulse. The default delay prints one dot per second so you can see the target is really waiting before the trigger window.

| Stage / mode | PMON mode | What | Energy interpretation |
|--------------|-----------|------|-----------------------|
| `T/E_prep`, `E2E_SYS_PMON_MODE=1` | System Power | `preprocess_ctf_pack()` × `E2E_SYS_PMON_REPS` (default 10000) | M4F math/pack energy; CNN accelerator remains disabled |
| `T/E_INP`, `E2E_SYS_PMON_MODE=4` | System Power | `preprocess_ctf_load_input()` × `E2E_SYS_PMON_REPS` (default 100000) | M4F copy/write into CNN input SRAM |
| `T/E_INF`, default ADI/CNN PMON run | CNN/ANN Power | `load_input()` and `load_input()` + `cnn_start()` windows | CNN inference energy = `E(input+inf) - E(input)` |
| `T/E_act`, `E2E_SYS_PMON_MODE=2` | System Power | `softmax_layer()` × `E2E_SYS_PMON_REPS` (default 100000) | M4F activation / output energy |
| Full-e2e check, `E2E_SYS_PMON_MODE=3` | System Power | full `preprocess_ctf_pack()` + `preprocess_ctf_load_input()` + CNN + `softmax_layer()` × `E2E_SYS_PMON_REPS` (default 2000) | Cross-check against `E_prep + E_INP + E_INF + E_act` |
| Trigger check, `E2E_SYS_PMON_MODE=5` | System Power | fixed 1 s `SYS_START` / `SYS_COMPLETE` pulse | PMON setup check only; PMON display `T` should be close to 1000 ms |
| Timer-clear check, `E2E_SYS_PMON_MODE=6` | CNN Power, then System Power | tiny CNN-mode trigger pulse, then fixed 1 s System pulse | Tests whether forcing PMON's CNN-mode `TMR32_ClearFlag()` fixes the 45.7 s System time |
| Prep sustained mW, `E2E_SYS_PMON_MODE=7` | System Power | `preprocess_ctf_pack()` × 1,200,000 | Read PMON `mW` as `P_prep_mW`; compute total prep energy from `P_prep_mW * T_prep` |
| Prep idle diagnostic, `E2E_SYS_PMON_MODE=8` | System Power | fixed M4F idle/busy-wait baseline, default 60 s | Optional diagnostic only; not required and not subtracted in the reported prep/activation energy |

#### Running System Power Mode from Eclipse

`E2E_SYS_PMON_MODE` is a compile-time firmware option. It is not selected from UART and it is not selected by the EVKIT PMON screen alone. If all `E2E_SYS_PMON_MODE` lines in `project.mk` stay commented, the firmware is the default ADI/CNN PMON + e2e latency run.

To measure one System Power stage:

1. Open `inference/indoor_env_1d_e2e/project.mk`.
2. Uncomment exactly one mode line. For prep energy, change:

   ```make
   # PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=1        # System Power Mode: prep only
   ```

   to:

   ```make
   PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=1        # System Power Mode: prep only
   ```

3. In Eclipse, clean/rebuild the project so the define is compiled into the ELF.
4. Leave the EVKIT PMON display in a non-windowed page, such as **AVG PWR**, while Eclipse programs and starts the target. This avoids arming PMON during reset/debugger GPIO transients.
5. Start the target from Eclipse.
6. When UART prints the arming-delay line, put PMON into **System Power** mode during that delay. From the PMON USB serial terminal, send the single character `s` for System Power mode. Do **not** send `t` for this run; in the PMON v2.0 source, `t` selects CNN Power mode.
7. Confirm UART prints this block:

   ```text
   --- EVKIT System Power Mode run ---
   [SYS PMON] Measuring ...
   Firmware SYS PMON window = ... us total, ... us/iteration over ... reps.
   ```

8. For sustained M4F-stage runs, record only the PMON `mW` value. Ignore PMON `E` and `T`; use the calculation below with the UART stage latency.
9. When done, comment the `E2E_SYS_PMON_MODE` line again and clean/rebuild to return to the default ADI/CNN + e2e latency firmware.

If UART still shows `--- ADI-style PMON energy ---`, you are not running a System Power firmware build. PMON System Power readings from that default firmware are not the prep/activation stage energy.

If fixed 1 s mode reports a stable PMON time near 45.7 s instead of 1000 ms, the EVKIT PMON is not using the firmware's P1.6 pulse as the displayed time window. The PMON v2.0 source suggests a likely stale TMR1 overflow flag path: CNN Power mode clears the TMR1 flag before timing, but System Power mode resets the count without clearing the flag and later adds `UINT32_MAX` if the stale flag is set.

To check this, reset/power-cycle the PMON MCU, then run `E2E_SYS_PMON_MODE=5` before doing any CNN Power Mode capture. The PMON TMR1 free-runs from PMON boot, so keep the time from PMON boot to target `SYS_START` well below the timer wrap interval, about 44-46 s. A tighter calibration build can use:

```make
PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=5
PROJ_CFLAGS += -DE2E_SYS_PMON_ARM_DELAY_MS=2000
```

Power-cycle PMON, send `s` on the PMON serial port as soon as it enumerates, then start/release the MAX78002 target. If PMON `T` becomes close to 1000 ms, the stale-flag hypothesis is confirmed.

If the tight mode 5 test still reports ~45.7 s, run mode 6:

```make
PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=6
PROJ_CFLAGS += -DE2E_SYS_PMON_ARM_DELAY_MS=2000
```

During the first `..`, send `t` to PMON serial for CNN Power mode. The firmware emits a tiny idle/active CNN trigger pulse, which should execute PMON's CNN-mode `TMR32_ClearFlag()` path. During the second `..`, send `s` for System Power mode. The following 1 s System pulse should show ~1000 ms if the overflow flag was the problem.

If the PMON time is still the fixed ~45.7 s interval, do not divide PMON `E` or `T` by the short firmware repetition count. In this condition, use **only the PMON `mW` field** for M4F stages. The PMON `mW` value comes from the power monitor AFE; the corrupted field is the timer-derived `T`, and therefore the displayed/inferred `E`.

Use a sustained run that keeps the measured stage active long enough for a stable `mW` reading. For System Power Mode, compute per-invocation M4F-stage energy directly from the measured total system power and the UART stage latency:

```text
E_stage_uJ = P_stage_mW * UART_stage_us / 1000
```

Equivalent in nJ:

```text
E_stage_nJ = P_stage_mW * UART_stage_us
```

Example using the observed prep latency:

```text
T_prep = 46 us
P_prep = 9.00 mW   (System Power sustained preprocess loop)

E_prep = 9.00 * 46 / 1000 = 0.414 uJ = 414 nJ
```

Do not subtract `E2E_SYS_PMON_MODE=8` from the reported M4F-stage energy. The ADI guide's idle-subtraction equation applies to CNN Power Mode, where idle and active CNN windows are paired. For these prep and activation measurements we are using System Power Mode, which measures total system power during the selected `SYS_START` / `SYS_COMPLETE` window. Mode 8 can still be used as a diagnostic board-power reference, but it is not part of the paper calculation.

For `T_prep`, the sustained loop should call `fill_ctf_raw_from_recording(0)` once before the PMON window, then repeat `preprocess_ctf_pack()` inside the window. Do not include `fill_ctf_raw_from_recording()` inside the repeated work unless you intentionally want to charge Flash-to-SRAM recording copy to preprocessing.

Use the dedicated sustained prep modes for this:

```make
# Get P_prep_mW from PMON System mW
PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=7
```

Mode 7 uses `E2E_SYS_PMON_REPS=1200000` by default. With `T_prep` around 46 us, it keeps the M4F in preprocessing for roughly 55 s, long enough for the EVKIT System Power `mW` reading to settle. Read only the PMON `mW` field.

For `T_act`, the sustained loop should prime one valid CNN output before the PMON window, then repeat `softmax_layer()` inside the window. This matches the e2e latency definition because `softmax_layer()` includes `cnn_unload()` plus softmax.

For the observed 45.7 s PMON interval, useful repetition overrides are:

| Stage | Mode | Suggested reps |
|-------|------|----------------|
| prep | `E2E_SYS_PMON_MODE=1` | `E2E_SYS_PMON_REPS=1200000` |
| input-copy | `E2E_SYS_PMON_MODE=4` | `E2E_SYS_PMON_REPS=5000000` |
| activation | `E2E_SYS_PMON_MODE=2` | `E2E_SYS_PMON_REPS=10000000` |
| full e2e | `E2E_SYS_PMON_MODE=3` | `E2E_SYS_PMON_REPS=300000` |

Example:

```make
PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=1
PROJ_CFLAGS += -DE2E_SYS_PMON_REPS=1200000
```

Isolation notes:

- Prep-only mode leaves the CNN accelerator disabled and does not enable the CNN IPLL.
- Input-only mode packs one input before the PMON window, then repeats only the CNN input SRAM copy inside the window.
- Activation-only mode primes one valid CNN output before the PMON window. The measured window runs `cnn_unload()` + softmax only; it does not call `cnn_start()`.
- Full-e2e mode intentionally includes CNN inference in the System Power window and is mainly a cross-check.
- Mode 8 is only an optional idle/busy-wait diagnostic. It is not required for `E_prep` or `E_act`, and the reported System Power energy does not subtract it.
- Fixed 1 s mode is a calibration check for PMON triggering. Run it before trusting prep/activation energy; if PMON `T` is not close to 1000 ms, fix PMON mode/setup before collecting stage energy. On PMON v2.0, arming after the target has started is more reliable than leaving PMON in System Power Mode through the Eclipse reset/program cycle.
- ADI recommends System Power measurement windows between about 100 ms and 20 s, so System Power runs use larger default repetition counts than the CNN/ANN Power Mode `x100` blocks.

Use `E_e2e = E_prep + E_INP + E_INF + E_act`; System Power Mode mode 3 is only a cross-check and may differ slightly because it measures one combined window.

### CNN wait while inference finishes

| α (after retarget) | Default wait | Why |
|--------------------|--------------|-----|
| **≤ 71** | `__NOP` busy-wait | Matches `indoor_env_1d_51_q8824` fix — `MXC_LP_EnterSleepMode()` can **hang** |
| **> 71** (e.g. 91 or 101) | Sleep until IRQ | Matches larger-α synthesis trees |

Auto-selected from `CTF_ALPHA` in `preprocess_config.h` (threshold 71). UART prints which mode is active at boot. Override with `-DE2E_CNN_WAIT_NOP=0|1` if needed.

### Per-stage energy — e2e loop (external PMON)

After the ADI blocks, **`E2E_NUM_RUNS`** iterations use **per-stage** GPIO for timing/oscilloscope-style checks:

| Pin | Macro | High during (each loop) |
|-----|--------|-------------------------|
| **P1.6** | `SYS_START` / `SYS_COMPLETE` | `T_prep`, then `T_act` |
| **P1.7** | `CNN_START` / `CNN_COMPLETE` | fused `T_INP+INF` |

Map ADI step 4 energy to **sampledata fused input+inference**. For e2e energy accounting, use separate System Power Mode runs for CPU-side prep/activation/full-e2e.

### Live path (VNA → MCU)

Training-time `.mat` loading, grid splits, and labels stay on the PC. One live classification is only:

Assume the sweep is done and the interface has written complex CTF into MCU SRAM.

```text
  VNA + link                MAX78002
  ─────────                 ─────────────────────────────────────────
  RF sweep finishes    →    SRAM: float ctf_raw[α][2]   (raw, not CNN-ready)
                            │
                            ▼
                      Cortex-M4F (ARM)  — preprocess_ctf_pack()  ← T_prep
                            │  • optional: center 101→α when α < 101
                            │  • global min–max (fixed constants)
                            │  • float → int8 per Re / Im
                            │  • pack into SRAM words
                            ▼
                      Cortex-M4F (ARM)  — preprocess_ctf_load_input()
                            │  • copy packed words into CNN input SRAM
                            ▼
                      CNN accelerator      — cnn_start()           ← fused T_INP+INF
                            │  (sleep or __NOP wait until IRQ; α≤71 → __NOP)
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
| CNN graph, `cnn.c`, `weights.h`, `sampleoutput.h` | `ai8xize` / synthesis (`indoor_env_1d_*_q8824`, retargeted per run) |
| E2E bench | Copy to `indoor_env_1d_e2e`, add `preprocess.c`, `ctf_recordings.h`, rewrite `main.c` |

Do **not** re-run synthesis for bench-only changes; re-synthesize only when the quantized network changes.

## Retarget for another α or MPQ model

Use **one** e2e tree (`indoor_env_1d_e2e`) for all experiments. After **ai8xize** produces a folder under `inference/` (e.g. `indoor_env_1d_91_q8824`, `indoor_env_1d_51_q8824`), run from repo root:

```bash
python3 misc/retarget_e2e_from_synthesis.py inference/indoor_env_1d_101_q8824
python3 misc/retarget_e2e_from_synthesis.py inference/indoor_env_1d_91_q8824
python3 misc/retarget_e2e_from_synthesis.py inference/indoor_env_1d_51_q8824
# If folder name alpha != CHW length in main.c (some trees use 71 in main.c):
python3 misc/retarget_e2e_from_synthesis.py inference/indoor_env_1d_51_q8824 --alpha 71
```

The script:

| Action | Files |
|--------|--------|
| Copies synthesis bundle | `cnn.c`, `cnn.h`, `weights.h`, `sampleoutput.h`, `softmax.c`, `log.txt` |
| Regenerates layout header | `preprocess_config.h` — `CTF_ALPHA`, CNN SRAM addresses, `CNN_INPUT_WORDS_PER_CH` (from `load_input()` in synthesis `main.c`) |
| Regenerates raw sweep (optional) | `ctf_recordings.h` via `export_ctf_raw_h.py --alpha …` |

**Keep unchanged:** `main.c`, `preprocess.c`, `preprocess.h` (bench logic is model-agnostic).

**MPQ rule:** each bitwidth / checkpoint needs its **own** synthesis output — swap the whole `cnn.*` + `weights.h` bundle; do not mix `q8824` weights with another graph.

**KAT:** `sampleoutput.h` matches the **synthesis golden input**, not an arbitrary `ctf_recordings.h` sweep. The e2e firmware leaves this diagnostic off by default; when enabled, `E2E_RUN_KAT=1` checks the ADI-style fused sampledata loop at the same point as the generated demos. Use `E2E_RUN_E2E_KAT=1` only when `ctf_recordings.h` reconstructs the same input used for synthesis.

### Manual swap (if you prefer)

| Step | What |
|------|------|
| 1 | Copy `cnn.c`, `cnn.h`, `weights.h`, `sampleoutput.h`, `softmax.c` from the synthesized project |
| 2 | From synthesis `main.c` `load_input()`, set `CNN_INPUT_ADDR_*` and `CNN_INPUT_WORDS_PER_CH` in `preprocess_config.h` |
| 3 | Set `CTF_ALPHA` to the model’s 1D length (CHW `Nx1` comment) |
| 4 | `python3 misc/export_ctf_raw_h.py --alpha N` |
| 5 | `make` and flash |

Other synthesis trees in this repo: [`indoor_env_1d_101_q8824`](../indoor_env_1d_101_q8824/), [`indoor_env_1d_91_q8824`](../indoor_env_1d_91_q8824/), [`indoor_env_1d_51_q8824`](../indoor_env_1d_51_q8824/) (verify α in that tree’s `main.c`).

## Regenerate `ctf_recordings.h`

From the repo root (dataset lives at `training/data/indoor_environment/`; needs `scipy`):

```bash
# Default export helper uses classroom static, sweep 0; pass --alpha N for the active model
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

Flash once over SWD (OpenOCD / VS Code “Flash” task). UART prints `T_prep` min/max/mean and optional diagnostic known-answer checks.

Optional defines (e.g. in `project.mk`):

- `E2E_NUM_RUNS` — benchmark iterations (default 100).
- `E2E_PMON_ADI_ONLY=1` — stop after the generated-demo-style PMON windows for cleaner EVKIT screen pages.
- `E2E_SYS_PMON_MODE=1` — System Power Mode run for prep only.
- `E2E_SYS_PMON_MODE=2` — System Power Mode run for activation only.
- `E2E_SYS_PMON_MODE=3` — System Power Mode run for full e2e.
- `E2E_SYS_PMON_MODE=4` — System Power Mode run for input-copy only.
- `E2E_SYS_PMON_MODE=5` — System Power Mode fixed 1 s trigger calibration.
- `E2E_SYS_PMON_MODE=6` — CNN-mode timer-clear pulse followed by System Power fixed 1 s trigger calibration.
- `E2E_SYS_PMON_MODE=7` — sustained prep loop for PMON System `P_prep_mW`.
- `E2E_SYS_PMON_MODE=8` — optional fixed prep-idle diagnostic window; not used in reported `E_prep` or `E_act`.
- `E2E_SYS_PMON_ARM_DELAY_MS` — delay before the System Power pulse so PMON can be armed after Eclipse starts the target (default 5000). Set to `0` only if your PMON reliably captures immediate windows.
- `E2E_SYS_PMON_REPS` — override System Power repetitions (defaults: prep 10000, input-copy 100000, activation 100000, full e2e 2000, sustained prep 1200000).
- `E2E_SYS_PMON_IDLE_MS` — optional mode 8 diagnostic duration in milliseconds (default 60000).
- `E2E_RUN_KAT=1` — run the optional ADI fused sampledata-loop KAT against `sampleoutput.h`.
- `E2E_RUN_E2E_KAT=1` — additionally check the last raw/preprocessed e2e output against `sampleoutput.h` when using a matching recording.

## Source layout

| File | Role |
|------|------|
| `main.c` | Bench loop, timers, UART |
| `preprocess.c` / `preprocess.h` | `ctf_raw`, `preprocess_ctf`, Flash → SRAM fill |
| `preprocess_config.h` | **α**, CNN input addresses (generated by retarget script) |
| `ctf_recordings.h` | `const` recorded sweeps (generated) |
| `cnn.c`, `weights.h`, `sampleoutput.h` | From synthesis (swap per model via retarget) |

## Related docs

- [`docs/e2e_latency_measurement.md`](../../docs/e2e_latency_measurement.md) — latency budget, memory map, UART rules
- [`docs/max78002_m4f_cnn_examples.md`](../../docs/max78002_m4f_cnn_examples.md) — ADI MSDK reference flows
