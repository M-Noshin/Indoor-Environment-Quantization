# End-to-End Latency Model (VNA → MAX78002)

This document captures the **system-level latency model** for indoor environment classification, as agreed for the paper tables and the on-device measurement plan. It consolidates the dataset semantics, VNA settings, preprocessing on the MCU, memory placement, and what is measured vs. illustrative.

---

## 1. Scope and assumptions

### Problem setup (conceptual — live edge pipeline)

This is the **system we analyze** in the paper, not necessarily a single wired lab bench:

- **VNA** performs frequency-domain sounding and delivers **one α-point complex CTF per classification** (see §4).
- **MAX78002**: preprocessing runs on the **Cortex-M4F**; CNN inference runs on the **accelerator**.
- **No PC** in the timed data path (VNA interfaces directly to the MCU, or equivalent edge setup).
- **Link / transfer** (VNA → MCU) is assumed **fast enough** and is **not** included in the budget ($T_{\mathrm{link}}$ omitted).
- **One classification** = **one VNA sweep** → **one preprocess** → **one forward pass** → **one label** (no averaging of multiple sweeps unless a separate mode is defined).

### What we measure on the MAX78002 (bench — simulated acquisition)

The EVKIT **is not wired to a VNA**. We **do not** time RF acquisition or host transfer on the MCU board.

| Stage | On MAX78002? | How |
|-------|----------------|-----|
| $T_{\mathrm{sense}}(\alpha)$ | **No** | Reported separately (VNA campaign, literature, or assumed); added only when forming **total** $T_{\mathrm{e2e}}$ in the paper table. |
| $T_{\mathrm{link}}$ | **No** | Neglected by assumption. |
| $T_{\mathrm{prep}}$, $T_{\mathrm{INP+INF}}$, $T_{\mathrm{act}}$ | **Yes** | Timed on-device starting from a **raw CTF already in MCU SRAM**, as if the sweep had just completed. |

### Simulation rule (bench — how we emulate “sensing is done”)

**Important:** On the EVKIT we do **not** re-flash SRAM from the PC on every loop. We **flash the firmware once** (`.elf` over SWD). Recorded sweeps sit in **Flash** as `const` arrays. Each benchmark iteration:

1. **`memcpy` (untimed):** `const` recording in Flash → **`ctf_raw[α][2]` in SRAM**
2. **START TIMER**
3. **Timed:** `preprocess_ctf` → CNN → softmax

That state — **raw CTF already in SRAM** — is what we mean by “$T_{\mathrm{sense}}$ finished; data ready in RAM.” The `memcpy` is **not** part of $T_{\mathrm{prep}}$ and **not** $T_{\mathrm{link}}$; it only sets up the same memory state a live VNA+interface would leave behind.

```text
  Flash (in firmware image)          SRAM (runtime)
  ---------------------------        ---------------------------
  const float recording[]  ---memcpy (untimed)--->  ctf_raw[α][2]
                                                           |
                                                   START TIMER
                                                           v
                                                   preprocess → CNN → act
```

| Step | Where | Timed? |
|------|--------|--------|
| Build: export `.mat` slice → `ctf_recordings.h` | PC | — |
| Flash firmware (code + `const` recordings) | PC → device Flash | Once |
| `memcpy` → `ctf_raw` | Flash → **SRAM** | **No** |
| $T_{\mathrm{prep}}$, $T_{\mathrm{INP+INF}}$, $T_{\mathrm{act}}$ | M4F + CNN | **Yes** |

**Not the same as KWS “streaming”:** the keyword demo streams from an **onboard mic (I2S/DMA → SRAM)**, not from a laptop each frame. We have no VNA on-chip; **Flash → SRAM `memcpy`** is our equivalent of “the sweep is already in RAM.”

### UART from PC each loop — possible, not for paper timings

You **can** write firmware that **UART-reads** a raw CTF from the host **every loop**. Rules:

| UART read placement | Allowed? | Notes |
|---------------------|----------|--------|
| **Before** `START TIMER` | Yes (debug / bring-up) | Host transfer is **not** $T_{\mathrm{sense}}$; keep it **outside** all reported MCU stages |
| **Inside** the timed region (with prep + CNN) | **Do not** use for the paper table | That times **PC → MCU delivery**, which we explicitly omit ($T_{\mathrm{link}}$) and is not preprocessing |

So: streaming over UART is a **test harness**, not the recommended emulation of VNA acquisition. For published $T_{\mathrm{prep}}$ … $T_{\mathrm{act}}$, use **Flash `const` + untimed `memcpy` → SRAM** (above).

### Raw CTF buffer (live vs. bench)

In a **real** live scenario:

1. Each sweep completes → the interface writes **this sweep’s** raw CTF into a **fixed SRAM buffer** (`ctf_raw`, α complexes or α×{Re, Im}).
2. M4F runs $T_{\mathrm{prep}}$ on that buffer → writes int8 into **CNN accelerator input memory**.
3. After classification, the **next sweep overwrites** the same buffer (or a ping-pong pair if capture and prep overlap).

**Memory roles (MAX78002):**

| Resource | Typical use |
|----------|-------------|
| **SRAM** | **Current raw sweep** (`ctf_raw`) — volatile, overwritten every classification. Working stack/heap for prep. |
| **Flash** | Program, CNN weights, optional **const golden sweeps** used to *initialize* `ctf_raw` before timing (not the rolling live buffer). |
| **CNN data memory** | Preprocessed **int8** input tensor (`[2, α]`) at synthesized addresses (e.g. `0x5180…` in indoor projects). |

Copying from Flash → `ctf_raw` in SRAM **before** the timed region is the **canonical bench method** (see diagram above). In a live system, the same SRAM slot would be filled by the VNA interface (UART/SPI/DMA), not by the host PC during the timed window.

### Frequency support (live VNA vs. training crop)

- The **original public dataset** was acquired with **101 points**, **100 MHz** span, **1 MHz** spacing, **2.4–2.5 GHz**, **~20.3 ms** per sweep, **200 sweeps** per grid location (slow-fading). Bandwidth ablations in the original CNN work used **software subsampling** of those 101-point sweeps, not separate VNA campaigns per bandwidth.
- The **live benchmark** configures the VNA to acquire **only** the frequency bins that deployed models use: **α points**, **1 MHz** step, **(α − 1) MHz** span, **centered at 2.45 GHz**. This matches **center crop** in training (`Slice1DLength`, `mode='center'`).
- **No center crop on the MCU** when the VNA is configured for α bins directly (VNA already outputs the correct α bins).

### Weight loading

- Steady-state streaming assumes weights are **already in accelerator memory**.
- Optional row: **$T_{\mathrm{W}}/N$** when weights are reloaded every $N$ classifications (see §6).

### What the CNN input is

- Semantic input: **CTF** $H(f)$ at α frequencies → **Re** and **Im** per bin.
- After preprocessing: **int8** tensor shaped **`[2, α]`** (channel 0 = Re, channel 1 = Im), same as QAT/deploy training with `--8-bit-mode`.

---

## 2. End-to-end pipeline (diagram)

### 2.1 Conceptual system (paper $T_{\mathrm{e2e}}$)

```text
   [VNA]  α complex samples H(f_k), k = 1..α
      |
      |  T_sense  (RF acquisition — measured / assumed off-EVKIT)
      v
   [MCU SRAM]  ctf_raw[]  ← raw sweep lands here in a live system
      |
      |  T_prep  (M4F: Re/Im, global min–max, int8 map, pack)
      v
   [CNN data memory]  int8 [2, α]
      |
      |  T_INP+INF
      v
   [MCU M4F]  T_act  (softmax, label)
      v
   environment label (4 classes)
```

### 2.2 On-device benchmark (what we actually time on MAX78002)

```text
   [Recorded sweep]  --(setup only, not in T_prep)-->  ctf_raw[] in SRAM
                                                          |
                    START TIMER  -------------------------+
                                                          v
   [M4F]  T_prep:  ctf_raw → CNN input memory (int8 [2, α])
      |
      v
   [CNN]  T_INP+INF
      |
      v
   [M4F]  T_act
      |
      END TIMER
```

**Not in on-device timed path:** PC, VNA, $T_{\mathrm{link}}$, Flash→SRAM copy (if defined as pre-acquisition setup), center crop on MCU (when VNA already outputs α bins).

**Not in conceptual path only:** N/A — $T_{\mathrm{sense}}$ is included when summing full $T_{\mathrm{e2e}}$ in the table.

---

## 3. Timing equation

### Per-classification (steady state, weights resident)

$$
\boxed{
T_{\mathrm{e2e}}(\alpha) = T_{\mathrm{sense}}(\alpha) + T_{\mathrm{prep}}(\alpha) + T_{\mathrm{INP+INF}}(\alpha) + T_{\mathrm{act}}
}
$$

| Symbol | Meaning | Where measured |
|--------|---------|----------------|
| $T_{\mathrm{sense}}(\alpha)$ | One VNA sweep (α frequency points) | **Off EVKIT** — VNA / trigger → data ready in MCU RAM (conceptual); not measured on MAX78002 |
| $T_{\mathrm{prep}}(\alpha)$ | Read `ctf_raw` in SRAM → Re/Im, global min–max, int8 map, pack to CNN buffers | Cortex-M4F; timer starts with raw buffer already valid |
| $T_{\mathrm{INP+INF}}(\alpha)$ | Input load into CNN SRAM + accelerator inference | MAX78002 (Tables in paper) |
| $T_{\mathrm{act}}$ | Post-processing (softmax, optional UART) | Cortex-M4F; usually small |

Definitions (aligned with paper notation):

$$
T_{\mathrm{INP+INF}} = T_{\mathrm{INP}} + T_{\mathrm{INF}}
$$

If preprocessing **writes directly** into CNN input SRAM, $T_{\mathrm{INP}}$ may be negligible; state the convention used when reporting.

### Optional: amortized weight load

When the model is reloaded every $N$ inputs:

$$
T_{\mathrm{total}}(N) = \frac{T_{\mathrm{W}}}{N} + T_{\mathrm{sense}} + T_{\mathrm{prep}} + T_{\mathrm{INP+INF}} + T_{\mathrm{act}}
$$

$T_{\mathrm{W}}$ is measured separately (existing weight-loading tables).

---

## 4. VNA settings vs. training crop

Training uses **center crop** on the stored 101-point grid (`training/datasets/indoor_environment_1D.py`):

- `start = (101 - α) // 2`, keep indices `start … start+α-1`
- Bin $k$ on full grid: $f_k = 2.4 + k$ MHz (MHz), $k = 0,\ldots,100$

Live VNA configuration for each α:

| Parameter | Value |
|-----------|--------|
| Points | α |
| Spacing | 1 MHz |
| Span | (α − 1) MHz |
| Center | 2.45 GHz |

### Mapping table (odd α values used in sweeps)

| α | Dataset indices (101-pt grid) | VNA start–stop | Effective span |
|---|------------------------------|----------------|----------------|
| 101 | 0–100 | 2.400–2.500 GHz | 100 MHz |
| 91 | 5–95 | 2.405–2.495 GHz | 90 MHz |
| 81 | 10–90 | 2.410–2.490 GHz | 80 MHz |
| 71 | 15–85 | 2.415–2.485 GHz | 70 MHz |
| 61 | 20–80 | 2.420–2.480 GHz | 60 MHz |
| 51 | 25–75 | 2.425–2.475 GHz | 50 MHz |
| 41 | 30–70 | 2.430–2.470 GHz | 40 MHz |
| 31 | 35–65 | 2.435–2.465 GHz | 30 MHz |
| 21 | 40–60 | 2.440–2.460 GHz | 20 MHz |
| 11 | 45–55 | 2.445–2.455 GHz | 10 MHz |
| 5 | 48–52 | 2.448–2.452 GHz | **4 MHz** (5 points ⇒ 4 intervals) |

**Note:** α = 5 means **5 frequency points**, not 5 MHz span.

All listed α are **odd**, so the crop is symmetric and includes the center bin at **2.45 GHz**.

---

## 5. Preprocessing detail ($T_{\mathrm{prep}}$ on M4F)

$T_{\mathrm{prep}}$ is a **single sequential stage** on the MCU. It includes what was previously called “pack” (complex → Re/Im and layout into CNN buffers).

### Steps (match training / deploy)

Constants from `GlobalMinMaxNormalize` in `indoor_environment_1D.py`:

- `global_min = -0.011066`
- `global_max =  0.011379`
- `global_range = global_max - global_min`

For each frequency bin $k = 0,\ldots,\alpha-1$ (Re and Im separately):

1. **Split complex** (if needed): $\mathrm{Re}\{H(f_k)\}$, $\mathrm{Im}\{H(f_k)\}$ as floats.
2. **Global min–max** (fixed stats, not per-sample):
   $$
   x' = \frac{x - \texttt{global\_min}}{\texttt{global\_range}} \in [0,1]
   $$
3. **8-bit activation map** (`ai8x.normalize` with `act_mode_8bit`):
   $$
   q = \mathrm{clamp}\big(\mathrm{round}((x' - 0.5) \times 256),\,-128,\,127\big)
   $$
4. **Pack** into **CHW** layout expected by the synthesized project (two channel buffers; see `sampledata.h` / `load_input()` addresses in `inference/*/main.c`).

### Training / test (PyTorch) — same math, different order wrapper

On each dataloader sample (`indoor_environment_1D.py`):

1. Load raw `[101, 2]` from `.mat`
2. Optional `Slice1DLength(α, center)` if α < 101 (training only; **skipped on MCU** when VNA outputs α bins directly)
3. `GlobalMinMaxNormalize` → `[0, 1]`
4. `ai8x.normalize` → int8 (with `--8-bit-mode`)
5. `permute` → `[2, α]` for Conv1d

Train and test use the **same** transform pipeline.

---

## 6. What existing firmware does today vs. planned bench

Example: `inference/indoor_env_1d_51_q8824/main.c`

- **`load_input()`**: `memcpy32` from **pre-baked** `sampledata.h` in **Flash** (already normalized int8) into CNN data memory.
- **Skips** `ctf_raw` and $T_{\mathrm{prep}}$ — fine for existing $T_{\mathrm{INP+INF}}$ tables, **not** sufficient for the full e2e breakdown row.

**Planned bench firmware:**

1. Define **`ctf_raw`** in **SRAM** (float or int16 Re/Im per bin; size $\mathcal{O}(\alpha)$).
2. **Setup (untimed):** fill `ctf_raw` from a recorded dataset sweep (const array in Flash or debugger init).
3. **Timed region:** `preprocess_ctf()` (`ctf_raw` → CNN input memory) = $T_{\mathrm{prep}}$; then `cnn_start()` / unload = $T_{\mathrm{INP+INF}}$; softmax = $T_{\mathrm{act}}$.
4. Repeat with other sweeps by overwriting `ctf_raw` between runs (mimics successive live classifications).

**Code foundation (hybrid):**

| Source | Use for |
|--------|---------|
| MSDK `Examples/MAX78002/CNN/kws20_demo` | Project layout, main loop, timing brackets, M4F → CNN orchestration |
| This repo `inference/indoor_env_1d_<α>_q*/` | `cnn.c`, `weights.h`, CNN SRAM addresses, your MPQ models |

See [max78002_m4f_cnn_examples.md](./max78002_m4f_cnn_examples.md) for other references.

### Where code lives (Cortex-M4F vs. accelerator)

All **Arm (M4F) application logic** — buffers, preprocessing math, timing, and the benchmark loop — belongs in the **application layer**, mainly **`main.c`**. Do **not** put preprocessing inside `cnn.c` (that file stays the izer-generated accelerator driver).

| File | Responsibility | Timed stage |
|------|----------------|-------------|
| **`main.c`** | `ctf_raw[]` in SRAM; untimed sweep load; main loop; GPIO/timer start–stop; calls prep + CNN + softmax | Orchestrates all on-device stages |
| **`preprocess_ctf()`** in `main.c` **or** `preprocess.c` / `preprocess.h` | Global min–max, int8 map, pack to CNN input addresses (§5) | **$T_{\mathrm{prep}}$** |
| **`cnn.c` / `cnn.h`** | Load weights/bias, `cnn_start()`, unload activations — **leave as synthesis output** | **$T_{\mathrm{INP+INF}}$** |
| **`softmax.c`** (or block in `main.c`) | Unpack logits, softmax, class index | **$T_{\mathrm{act}}$** |
| **`sampledata.h`** (optional) | Const **raw** or golden vectors in Flash for setup only | **Not** in timed path if only used to init `ctf_raw` |
| **`weights.h`** | Quantized kernels in Flash | Loaded once (or amortized $T_{\mathrm{W}}$) |

### `main()` flow (on-device benchmark)

Maps §2.2 to concrete functions:

```text
  SETUP (untimed)
  ---------------
  fill_ctf_raw_from_recorded_sweep();   // Flash const / exported .mat → SRAM ctf_raw[]

  TIMED (per run)
  ---------------
  preprocess_ctf(ctf_raw, ...);         // SRAM → CNN input memory     →  T_prep
  cnn_load_weights();  cnn_load_bias(); // if not already resident
  cnn_start();                          // accelerator inference       →  T_INP+INF
  (wait; cnn_unload → ml_data)
  softmax_layer();                      // M4F post-process            →  T_act
```

Pseudocode sketch:

```c
// SRAM — simulates "VNA just wrote this sweep"
static float ctf_raw[ALPHA][2];   // or complex_t ctf_raw[ALPHA]

void main(void) {
    board_init();
    cnn_enable();

    for (int run = 0; run < N_RUNS; run++) {
        fill_ctf_raw_from_recorded_sweep(run);   // untimed

        t0 = timer_start();
        preprocess_ctf(ctf_raw, ALPHA);          // T_prep: §5 math → 0x5180…
        cnn_load_weights();
        cnn_load_bias();
        cnn_start();                             // T_INP+INF
        while (!cnn_time);
        cnn_unload(ml_data);
        softmax_layer();                         // T_act
        t1 = timer_stop();
    }
}
```

Replace `load_input()` + `memcpy32` from pre-baked int8 `sampledata.h` on the benchmark path; keep existing `load_input()` addresses as the **destination** of `preprocess_ctf()`.

### Loading `ctf_raw` (simulation) — practical options

**When the timer for $T_{\mathrm{prep}}$ starts, `ctf_raw` must already hold one sweep in SRAM** — see **Simulation rule** above (Flash `const` in the `.elf`, untimed `memcpy` into SRAM). You do **not** need the PC during the timed loop for paper numbers.

| Approach | PC every loop? | Use for paper timings? |
|--------|----------------|-------------------------|
| **A. Const in Flash → `memcpy` to SRAM** | No (flash once at build) | **Yes — recommended** |
| **B. Many sweeps in Flash; rotate index** | No | **Yes** |
| **C. UART/USB from PC each loop** | Yes | **Debug only** — host transfer is not $T_{\mathrm{sense}}$ and must stay **outside** the timer |
| **D. Debugger RAM write** | Manual | Bring-up only |

#### A. Recommended: export `.mat` → C array → untimed `memcpy` (same idea as `sampledata.h`)

**On the PC (once, or when you change α / test set):**

1. Load a sweep from `training/data/indoor_environment/*.mat` — shape **`[101, 2]`** Re/Im **before** `GlobalMinMaxNormalize` (raw dataset values).
2. If α &lt; 101, **center-crop** the same way as `Slice1DLength(α, center)` in `training/datasets/indoor_environment_1D.py`.
3. Emit a header, e.g. `ctf_recordings.h`:

```c
// one sweep, α=51, float Re/Im per bin
static const float ctf_recording_0[51][2] = { {re, im}, ... };
```

Or embed **N** sweeps (`ctf_recording_0 … ctf_recording_{N-1}`) for variety across loops.

**On the MCU (every loop iteration, untimed):**

```c
static float ctf_raw[ALPHA][2];   // lives in SRAM at runtime

static void fill_ctf_raw_from_recording(int idx) {
    // const source in Flash — NOT timed
    memcpy(ctf_raw, ctf_recordings[idx], sizeof(ctf_raw));
}
```

This matches a live system: the sweep already sits in SRAM; only **prep → CNN → act** are measured.

**Build flow:** flash the ELF once (USB SWD / OpenOCD, same as today’s indoor demos). The “sample” travels inside the firmware image in Flash, not over UART during the benchmark.

#### B. Multiple sweeps without a PC

```c
for (int run = 0; run < N_RUNS; run++) {
    fill_ctf_raw_from_recording(run % N_RECORDINGS);  // untimed
    // ... start timer → preprocess_ctf → cnn → softmax ...
}
```

Optional: at boot, copy all N recordings into a **SRAM ring** once (still untimed), then only update an index each loop.

#### C. PC streams raw CTF over UART each loop (debug only)

1. Firmware: blocking **UART read** into `ctf_raw` (binary: `α × 2 × sizeof(float)`).
2. Host Python: `serial.write(struct.pack(...))` after loading the same slice from `.mat`.
3. Place the UART read **before** `START TIMER` if you use this at all.
4. **Never** wrap UART + prep + CNN in one timed block for the paper — that would conflate host link time with $T_{\mathrm{prep}}$. For published results, use **A** or **B** (Flash → SRAM `memcpy`).

#### What not to do

- Do **not** time “PC → UART → SRAM” and call it $T_{\mathrm{prep}}$ or $T_{\mathrm{sense}}$.
- Do **not** use pre-normalized int8 from `sampledata.h` as `ctf_raw` — that skips the min–max + int8 steps you need in $T_{\mathrm{prep}}$.

#### Minimal PC export sketch (add a script under `misc/` if needed)

```python
import scipy.io
import numpy as np

mat = scipy.io.loadmat("training/data/indoor_environment/CTF_Class_static_final.mat")
arr = mat[list(mat.keys())[3]].T   # [N, 101, 2] — match dataset loader
sweep = arr[0]                     # one snapshot
# center-crop to alpha, then write ctf_recordings.h with float Re, Im
```

---

## 7. Original dataset (reference only)

From `paper/full-3.tex` / Noshin et al. measurement campaign:

| Item | Value |
|------|--------|
| Band | 2.4–2.5 GHz (100 MHz), center 2.45 GHz |
| Points per sweep | 101 (1 MHz spacing) |
| Sweeps per grid location | 200 |
| Sweep time (101 pts) | ~20.3 ms |
| One training sample | One sweep → 101×{Re, Im} |
| α in our MPQ work | Software center-crop of 101-pt vectors (or centered per-α VNA on live bench) |

The **~20.3 ms** figure is **sensing time for a full 101-point sweep** on the lab VNA, not MCU inference time. Expect $T_{\mathrm{sense}}(\alpha)$ to scale roughly with α when the VNA is programmed for fewer points (measure, do not assume linear scaling without data).

---

## 8. Illustrative vs. measured rows (paper table)

| Stage | Source |
|-------|--------|
| $T_{\mathrm{INP+INF}}(\alpha)$ | **Measured** on MAX78002 (e.g. Table `hw_representatives_horizontal_kb`) |
| $T_{\mathrm{W}}$ | **Measured** (weight load tables) |
| $T_{\mathrm{sense}}(\alpha)$ | **Off EVKIT** — VNA / literature / separate campaign (centered per-α configuration) |
| $T_{\mathrm{prep}}(\alpha)$ | **To measure** on M4F (new firmware; input = `ctf_raw` in SRAM) |
| $T_{\mathrm{act}}$ | **To measure** or small constant |
| Earlier illustrative $T_{\mathrm{sense}}$ placeholders | Stylized scaling with α; replace with measurements |

Example structure for the breakdown table (INT 8-8-2-4 or chosen config):

| Stage | α=101 | α=51 | α=5 |
|-------|-------|------|-----|
| $T_{\mathrm{sense}}$ | measure | measure | measure |
| $T_{\mathrm{prep}}$ | measure | measure | measure |
| $T_{\mathrm{INP+INF}}$ | from HW table | from HW table | from HW table |
| $T_{\mathrm{act}}$ | measure | measure | measure |
| **Total $T_{\mathrm{e2e}}$** | sum | sum | sum |
| $T_{\mathrm{W}}/N$ (optional) | from HW table | … | … |

Report **mean ± std** over many runs per α (e.g. 50–100). Each run: refresh `ctf_raw` from a recorded sweep (or rotate through a small set), then time prep → CNN → act only.

**Sum for paper table:** $T_{\mathrm{e2e}} = T_{\mathrm{sense}} + T_{\mathrm{prep}} + T_{\mathrm{INP+INF}} + T_{\mathrm{act}}$ with $T_{\mathrm{sense}}$ from VNA-side data and MCU stages from the EVKIT.

---

## 9. What to ask the VNA operator

> For each α ∈ {101, 91, 81, 71, 61, 51, 41, 31, 21, 11, 5}, configure the VNA with **α points**, **1 MHz spacing**, centered at **2.45 GHz** (span = (α−1) MHz), so acquired frequencies match the center-cropped bins used in training. Provide **sweep/acquisition time** per setting and a way to trigger/read α complex CTF samples into the MCU (no PC in the timed path).

---

## 10. Short paper sentence

> We model end-to-end latency assuming a VNA delivers one α-point CTF per classification into MCU SRAM, with $T_{\mathrm{sense}}$ measured or reported separately from VNA-side experiments. On the MAX78002, $T_{\mathrm{prep}}$ through $T_{\mathrm{act}}$ are measured using recorded sweeps placed in a raw SRAM buffer to emulate post-acquisition data; the Cortex-M4F performs preprocessing in a single stage (Re/Im handling, global min–max normalization, and 8-bit mapping) before accelerator inference $T_{\mathrm{INP+INF}}$. Link latency is neglected, and each classification uses one sweep’s CTF.

---

*Last updated: conceptual VNA→MCU SRAM buffer; EVKIT simulates `ctf_raw` in SRAM (no VNA on board); $T_{\mathrm{sense}}$ off-device; centered per-α VNA; combined $T_{\mathrm{prep}}$; no $T_{\mathrm{link}}$; one sweep per classification.*
