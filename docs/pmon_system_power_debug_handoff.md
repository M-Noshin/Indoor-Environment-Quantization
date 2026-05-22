# MAX78002 EVKIT PMON System Power Debug Handoff

## Context

Repo: `Indoor-Environment-Quantization`

Firmware under investigation:

- `inference/indoor_env_1d_e2e/main.c`
- `inference/indoor_env_1d_e2e/project.mk`
- `inference/indoor_env_1d_e2e/README.md`

Goal: measure end-to-end indoor environment inference energy on MAX78002 EVKIT:

- `E_prep`: Cortex-M4F preprocessing, `preprocess_ctf_pack()`.
- `E_INP`: Cortex-M4F copy/write into CNN input SRAM, `preprocess_ctf_load_input()`.
- `E_INF`: CNN accelerator inference.
- `E_act`: Cortex-M4F output unload + softmax, `softmax_layer()`.

The CNN/ANN PMON path is working. The unresolved problem is EVKIT PMON **System Power Mode** for the M4F stages.

## Known-Good Behavior

Default firmware mode, with all `E2E_SYS_PMON_MODE` lines commented out, works:

- Runs ADI-style CNN/ANN PMON windows:
  - P1.6 idle baseline
  - P1.7 weight load
  - P1.7 sampledata input load
  - P1.7 sampledata input + inference
- Runs e2e latency loop.
- Typical UART latency:

```text
T_prep over 100 runs: ~46 us
T_INP+INF over 100 runs: ~130 us
T_act over 100 runs: ~5 us
T_e2e: ~181 us
```

The EVKIT PMON screen for CNN/ANN mode reports reasonable input and input+inference windows. For paper consistency, the user expects to use PMON CNN/ANN values for the CNN path, e.g. around 127.6 us for ADI-style input+inference on the screen.

## Problem

System Power Mode on the EVKIT PMON reports a stable, incorrect time near **45.733 s**, even when firmware emits a fixed **1 s** P1.6 `SYS_START` / `SYS_COMPLETE` calibration pulse.

Observed mode 5 run:

```text
System PMON fixed-1s calibration mode: CNN block not enabled.

--- EVKIT System Power Mode run ---
Set PMON to System Power Mode before this window.
Measurement window: P1.6 SYS high, fixed 1000 ms calibration pulse.
Arming delay: 5000 ms. Put PMON in triggered/System Power Mode now.
.....
[SYS PMON] Measuring fixed 1 s calibration pulse - P1.6, 1000 ms...
System Power Mode run complete; use PMON screen/serial result for this one stage.
Firmware SYS PMON calibration window = 1000002 us total.
PMON T should be close to 1000 ms on the display.
```

PMON serial / LCD instead reports:

```text
0.49699,45.7337,0.0108671
```

LCD interpretation:

```text
496990 uJ
45733.7 ms
10.87 mW
```

This is not a firmware timing error: UART confirms a 1.000002 s firmware window. PMON is displaying a much longer interval.

## Hardware / Documentation Facts

Official references:

- ADI PMON guide:
  - `https://github.com/analogdevicesinc/MaximAI_Documentation/blob/main/Guides/MAX7800x%20Power%20Monitor%20and%20Energy%20Benchmarking%20Guide.md`
- PMON source:
  - `https://github.com/analogdevicesinc/max78000-powermonitor/blob/main/main.c`
- MAX78002 EVKIT datasheet:
  - `https://www.analog.com/media/en/technical-documentation/data-sheets/max78002evkit.pdf`
- PMON firmware binaries:
  - `https://github.com/analogdevicesinc/MaximAI_Documentation/tree/main/MAX78002_Evaluation_Kit/PMON_Firmware`

Known from docs:

- PMON windowing uses GPIO triggers P1.6 and P1.7.
- JP18 ties TRIG1 to P1.6.
- JP19 ties TRIG2 to P1.7.
- The user's board has PMON v2.0.
- The PMON source repository is named `max78000-powermonitor`, but it includes the relevant MAX32625 PMON state machine used by this PMON firmware family.
- In the PMON source, serial command `s` selects System Power mode. Serial command `t` selects CNN Power mode. Do not send `t` for System Power runs.

## Current Firmware State

Relevant compile-time modes in `project.mk`:

```make
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=1        # System Power Mode: prep only
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=2        # System Power Mode: activation only
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=3        # System Power Mode: full e2e
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=4        # System Power Mode: input-copy only
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=5        # System Power Mode: fixed 1 s trigger calibration
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=6        # System Power Mode: CNN-mode timer-clear + fixed 1 s calibration
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=7        # Sustained System mW: prep loop for P_prep_mW
# PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=8        # Optional System mW diagnostic: prep-idle loop
# PROJ_CFLAGS += -DE2E_SYS_PMON_REPS=2000     # override System Power repetitions
# PROJ_CFLAGS += -DE2E_SYS_PMON_ARM_DELAY_MS=0     # optional: fire System PMON window immediately
# PROJ_CFLAGS += -DE2E_SYS_PMON_FIXED_MS=10000 # mode 5: make P1.6 high long enough to probe with meter/scope
```

Default in `main.c`:

```c
#ifndef E2E_SYS_PMON_ARM_DELAY_MS
#define E2E_SYS_PMON_ARM_DELAY_MS 5000
#endif
```

Current System Power trigger path:

- Configure P1.6/P1.7 as GPIO outputs.
- Force both low.
- Wait 5 s, printing one dot per second.
- Emit P1.6-only `SYS_START` / `SYS_COMPLETE` window.
- Print firmware-measured window from `MXC_TMR0`.

This P1.6-only version is the version that causes PMON to trigger/show values. A previous TRIG2-high version did not work reliably on the user's setup, and the PMON source confirms that System Power mode is driven by the TRIG1/P1.6 path, not by holding TRIG2/P1.7 active.

## Things Already Tried

### 1. Default ADI/CNN PMON

Result: works. PMON outputs reasonable CNN/ANN windows. Not the source of the issue.

### 2. System Power mode with immediate trigger

Change:

```c
E2E_SYS_PMON_ARM_DELAY_MS = 0
```

Result: PMON did not trigger at all. This indicates the PMON must be armed after the Eclipse flash/reset/debug sequence.

### 3. System Power mode with 5 s arming delay

Result: PMON triggers, but reports the stable bad time around 45.733 s.

### 4. System Power fixed 1 s calibration mode

Build:

```make
PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=5
```

Firmware UART reports ~1,000,002 us, but PMON displays ~45,733.7 ms.

Conclusion from the first run: PMON is not using the firmware P1.6 pulse as its displayed time window. A subtler stale-flag variant remains possible: PMON TMR1 free-runs from PMON boot and can overflow after roughly 44-46 s even if PMON was freshly reset. If PMON boot to target `SYS_START` takes longer than that, the flag can be stale again before the 1 s pulse starts.

### 5. TRIG2/P1.7 high during System Power window

Rationale: Some guide wording was initially interpreted as "hold TRIG2 high." That interpretation was wrong.

Result on user's setup: did not trigger reliably. Reverted to P1.6-only window.

Source-code correction: the PMON System Power state machine reacts to the TRIG1/P1.6 active and deasserted states. It should not be driven with `CNN_START` / P1.7 held high.

### 6. Longer arming delay

Tried 20 s. Still reported ~45.733 s for the 1 s fixed calibration.

Conclusion: this is not just a "not enough time to arm PMON" problem.

## Current Working Hypothesis

The MAX32625 PMON v2.0 System Power mode is likely using a stale TMR1 overflow flag. In the PMON source, CNN Power mode clears the TMR1 flag before timing, while System Power mode resets TMR1 count but does not clear the flag before later checking it.

Observed source pattern:

```c
/* CNN Power Mode setup clears the flag. */
TMR32_SetCount(MXC_TMR1, 0);
TMR32_SetCompare(MXC_TMR1, 0xFFFFFFFF);
TMR32_ClearFlag(MXC_TMR1);

/* System Power Mode setup resets count but does not clear the flag. */
TMR32_SetCount(MXC_TMR1, 0);

/* Later, System Power Mode adds UINT32_MAX if the flag is set. */
if (TMR32_GetFlag(MXC_TMR1))
    time = ((double)t + (double)UINT32_MAX) / (double)SystemCoreClock;
else
    time = (double)t / (double)SystemCoreClock;
```

If the overflow flag remains set from an earlier PMON session, System Power time becomes roughly:

```text
UINT32_MAX / PMON_SystemCoreClock
```

which lands near the observed 45.733 s for a ~94 MHz PMON clock.

This is why the same PMON serial row keeps showing:

```text
E ~= 0.49 J
T ~= 45.733 s
P ~= 10.8 mW
```

The value is too stable and too far from the firmware's 1 s pulse to be interpreted as real stage time.

## Immediate Verification To Try

1. Reset or power-cycle the PMON MCU immediately before the System Power test.
2. Build/run only the fixed 1 s target calibration with a short arming delay:

   ```make
   PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=5
   PROJ_CFLAGS += -DE2E_SYS_PMON_ARM_DELAY_MS=2000
   ```

3. Keep the elapsed time from PMON boot to target `SYS_START` well below 44 s. Practical sequence:
   - Hold or delay the MAX78002 target.
   - Power-cycle/reset PMON.
   - As soon as PMON serial enumerates, send `s` to enter System Power mode.
   - Start/release MAX78002 quickly.
4. During the 2 s arming delay, PMON should already be in System Power mode.
5. Do not run a CNN Power Mode capture before this test.
6. If PMON `T` now reads close to 1000 ms, the stale TMR1 flag hypothesis is confirmed.

If the tight mode 5 test still reports ~45.7 s, run mode 6:

```make
PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=6
PROJ_CFLAGS += -DE2E_SYS_PMON_ARM_DELAY_MS=2000
```

Mode 6 sequence:

1. During the first arming delay, send `t` to PMON serial for CNN Power mode.
2. Target emits a tiny P1.6 idle pulse and P1.7 active pulse. PMON CNN mode should execute `TMR32_ClearFlag(MXC_TMR1)` on the active transition.
3. During the second arming delay, send `s` to PMON serial for System Power mode.
4. Target emits the fixed 1 s P1.6 System pulse.

Expected result if the overflow flag is the only issue: PMON System `T` becomes close to 1000 ms.

## Do Not Do

- Do not trust System Power stage energy if mode 5 reports ~45.7 s for a 1 s pulse.
- Do not divide PMON total energy by `E2E_SYS_PMON_REPS` unless PMON `T` roughly matches the firmware UART `Firmware SYS PMON window`.
- Do not use System Power readings from the default ADI/CNN firmware as M4F stage energy.
- Do not keep PMON in System Power mode through Eclipse flash/reset if trying to trigger from firmware; use the 5 s arming delay.
- Do not send PMON serial `t` for System Power. Use `s`.

## Current Workaround

If PMON insists on measuring a fixed ~45.7 s interval, run each M4F stage continuously for longer than 45.7 s and use PMON average power.

Formula:

```text
E_stage_uJ = PMON_avg_mW * UART_stage_us / 1000
```

Reason:

- `1 mW * 1 us = 1 nJ = 0.001 uJ`
- So `mW * us / 1000 = uJ`

Suggested sustained-run settings:

| Stage | Mode | Suggested reps |
|-------|------|----------------|
| prep | `E2E_SYS_PMON_MODE=1` | `E2E_SYS_PMON_REPS=1200000` |
| input-copy | `E2E_SYS_PMON_MODE=4` | `E2E_SYS_PMON_REPS=5000000` |
| activation | `E2E_SYS_PMON_MODE=2` | `E2E_SYS_PMON_REPS=10000000` |
| full e2e | `E2E_SYS_PMON_MODE=3` | `E2E_SYS_PMON_REPS=300000` |

Example prep run:

```make
PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=7
```

Optional prep-idle diagnostic run:

```make
PROJ_CFLAGS += -DE2E_SYS_PMON_MODE=8
```

Mode 7 uses `E2E_SYS_PMON_REPS=1200000` by default. It should keep prep active for about 55 s (`~46 us * 1,200,000`), so a 45.7 s PMON integration interval is mostly inside actual prep work. Mode 8 is only a board-power diagnostic and is not subtracted from the reported System Power energy.

Compute:

```text
E_prep_uJ = P_prep_mW * T_prep_us / 1000
```

## Open Questions For Next Agent

1. Does a fresh PMON reset make mode 5 report ~1000 ms?
2. Is the displayed 45.733 s interval exactly `UINT32_MAX / PMON_SystemCoreClock` on this board?
3. Are JP18/JP19 physically installed in the correct positions, and can P1.6/P1.7 be verified at the PMON side with a scope/logic analyzer?
4. Does another official MSDK example successfully use System Power Mode on this exact MAX78002 EVKIT + PMON v2.0 combination?

## Recommended Next Debug Steps

1. Probe P1.6 and P1.7 at JP18/JP19 during `E2E_SYS_PMON_MODE=5`.
   - P1.6 should go high for ~1 s.
   - P1.7 should remain low in the current P1.6-only version.
2. If P1.6 is correct at the MAX78002 pin but not at the PMON side, inspect JP18 routing.
3. If P1.6 is correct at the PMON side and PMON still reports 45.7 s, the issue is in MAX32625 PMON mode/firmware behavior, not target firmware.
4. Try an official ADI low-power/System Power example, if available, on the same board and PMON firmware.
5. If no official System Power example works, use the sustained-workload average-power method above for the paper and document the PMON limitation.

## Current Build Commands Used For Verification

From `inference/indoor_env_1d_e2e`:

```bash
PATH=/Users/hamza/MaximSDK/Tools/GNUTools/10.3/bin:$PATH \
make -B -r -j 8 TARGET=MAX78002 BOARD=EvKit_V1 MAXIM_PATH=/Users/hamza/MaximSDK \
PROJ_CFLAGS=-DE2E_SYS_PMON_MODE=5
```

Prep System Power build:

```bash
PATH=/Users/hamza/MaximSDK/Tools/GNUTools/10.3/bin:$PATH \
make -B -r -j 8 TARGET=MAX78002 BOARD=EvKit_V1 MAXIM_PATH=/Users/hamza/MaximSDK \
PROJ_CFLAGS=-DE2E_SYS_PMON_MODE=1
```

Both compile successfully.
