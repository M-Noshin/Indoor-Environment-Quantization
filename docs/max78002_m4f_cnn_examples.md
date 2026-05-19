# MAX78002: M4F Preprocessing → CNN Accelerator — Public Code Survey

Reference notes for the indoor-environment e2e bench: which Analog Devices examples split work between the **Cortex-M4F** and the **CNN accelerator**, and what is *not* published (e.g. MFCC-on-Arm from the KWS design note).

**MSDK:** [analogdevicesinc/msdk](https://github.com/analogdevicesinc/msdk)  
**Training / synthesis:** [ai8x-training](https://github.com/analogdevicesinc/ai8x-training), [ai8x-synthesis](https://github.com/analogdevicesinc/ai8x-synthesis)  
**Docs:** [MaximAI_Documentation](https://github.com/analogdevicesinc/MaximAI_Documentation)

---

## 1. Design note vs. shipping demos

The ADI article [*Keywords Spotting Using the MAX78000*](https://www.analog.com/en/resources/technical-articles/keywords-spotting-using-the-max78000.html) describes **Methodology 1**: MFCC on the Arm core (windowing, FFT, Mel filter bank, log, DCT) and CNN inference on the accelerator (“Figure 5 — MFCC processing on Arm”). That path was **initially investigated**.

**Public MSDK today:** a full-tree search shows **no files named or path-containing `mfcc`**. The maintained KWS demo does **not** run MFCC on-device; it feeds the CNN a prepared **int8** audio window. Training in `ai8x-training` (`datasets/kws20.py`, KWS20 v3) also does not use MFCC for the shipping pipeline.

So: the **split architecture** (M4F + accelerator) is real and documented in firmware; the **MFCC-on-Arm** pipeline from the article is background, not a copy-paste reference implementation.

---

## 2. MAX78002 CNN examples (MSDK `main` branch)

All under: `Examples/MAX78002/CNN/`

### 2.1 Live M4F preprocessing → CNN (best references)

| Example | Link | M4F role | CNN role | Notes |
|---------|------|----------|----------|--------|
| **KWS20 demo** | [kws20_demo](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/kws20_demo) | I2S mic, optional HPF, voice activity, scale 16-bit → **int8**, circular buffer, `AddTranspose()` into CNN memory | `cnn.c`: load weights, `cnn_start()`, unload | **`ENABLE_MIC_PROCESSING`**: live path. Undefine → header test vector (offline). **Closest analog** to VNA → `T_prep` → inference. |
| **Pascal VOC RetinaNet** | [pascalvoc-retinanetv7_3](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/pascalvoc-retinanetv7_3) | `camera_capture_and_load_cnn()` — OV5640 capture, buffer (external SRAM), feed CNN | Object-detection CNN | Post: **NMS on M4F** (`nms.c`). Good for **sensor ingest + timing**, not 1D CTF math. |
| **Facial recognition** | [facial_recognition](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/facial_recognition) | Camera + DMA, crop/offset, orchestration | **Two** CNNs: FaceDetection, then FaceID | `facedetection.c`, `faceID.c`, `post_process.c` on M4F. Heavier than needed, shows multi-stage M4F + CNN. |

**KWS20 key files**

- `main.c` — mic loop, `SAMPLE_SCALE_FACTOR`, `micBuff`, `AddTranspose()`, thresholds  
- `cnn.c` / `cnn.h` — accelerator API  
- `softmax.c` — classification on M4F  
- README — flowchart, v3 model takes **128×128 = 16,384** int8 samples (not MFCC features)

**Pascal VOC key files**

- `src/camera/camera.c` — `camera_capture_and_load_cnn()`  
- `src/cnn/cnn.c` — inference  
- `src/cnn/nms.c` — post-processing on M4F  

---

### 2.2 Accelerator-only (pre-baked `sampledata.h`)

Same pattern as this repo’s `inference/indoor_env_*`: M4F **`memcpy32`** from `sampledata.h` into CNN SRAM, then `cnn_start()`. No live feature extraction on M4F.

| Example | Link |
|---------|------|
| ImageNet | [imagenet](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/imagenet) |
| Kinetics | [kinetics](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/kinetics) |
| CIFAR-100 EfficientNet | [cifar-100-effnet2](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/cifar-100-effnet2) |
| CIFAR-100 MobileNet | [cifar-100-mobilenet-v2-0.75](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/cifar-100-mobilenet-v2-0.75) |
| MobileFaceNet (standalone) | [mobilefacenet-112](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/mobilefacenet-112) |

Useful for **CNN memory layout**, `load_input()` addresses, and `cnn.c` API — not for implementing live `T_prep`.

---

### 2.3 Other MAX78002 CNN `main.c` projects

| Path | Notes |
|------|--------|
| `imagenet-riscv` | RISC-V variant; check README for split vs M4F |

---

## 3. MAX78000 (same family, more examples)

If MAX78002-specific trees are thin, patterns port from MAX78000:

| Example | Link | Notes |
|---------|------|--------|
| KWS20 demo | [MAX78000/CNN/kws20_demo](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78000/CNN/kws20_demo) | Same structure as MAX78002 port; longer history |
| KWS20 v3 (minimal) | [kws20_v3](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78000/CNN/kws20_v3) | Izer-generated bare demo |
| KWS20 RISC-V | [kws20_demo-riscv](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78000/CNN/kws20_demo-riscv) | Coprocessor variant |

---

## 4. Toolchain (PC — not live M4F prep)

| Repo | Role |
|------|------|
| [ai8x-training](https://github.com/analogdevicesinc/ai8x-training) | Train / QAT (e.g. `./scripts/train_kws20_v3.sh`, `models/ai85net-kws20-v3.py`) |
| [ai8x-synthesis](https://github.com/analogdevicesinc/ai8x-synthesis) | Quantize + `izer` → `main.c`, `cnn.c`, `sampledata.h`, `weights.h` |
| [MaximAI_Documentation](https://github.com/analogdevicesinc/MaximAI_Documentation) | [Custom KWS / audio guide](https://github.com/analogdevicesinc/MaximAI_Documentation/blob/master/Guides/Making%20Your%20Own%20Audio%20and%20Image%20Classification%20Application%20Using%20Keyword%20Spotting%20and%20Cats-vs-Dogs.md) |

Generated indoor projects in this repo (`inference/indoor_env_1d_*`, `synthesis/`) are **izer output + demo wrapper** — same class as static MSDK CNN demos.

---

## 5. Mapping to indoor / VNA e2e bench

| KWS20-style step | Indoor target |
|------------------|---------------|
| Sensor capture (I2S mic) | VNA → MCU (α complex CTF samples) |
| M4F prep (scale, buffer, `AddTranspose`) | `T_prep`: Re/Im, global min–max → $[0,1]$, map → $[-127,127]$, pack to CNN SRAM |
| `cnn.c` / `cnn_start()` | Same as existing indoor firmware |
| `softmax.c` on M4F | Environment label (4 classes) |

**This repo today:** `inference/indoor_env_1d_*` — `load_input()` only copies **pre-baked** `sampledata.h` (like KWS **offline** mode). Live VNA requires new `preprocess_ctf_live()` on M4F, modeled on **KWS mic path** (role of prep), not MFCC.

See also: [e2e_latency_measurement.md](./e2e_latency_measurement.md).

---

## 6. Practical recommendations

| Goal | Start here |
|------|------------|
| Live prep + CNN + post on MAX78002 | [kws20_demo](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/kws20_demo) — `main.c`, `cnn.c` |
| Live sensor + timed capture/load/infer | [pascalvoc-retinanetv7_3](https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002/CNN/pascalvoc-retinanetv7_3) |
| CNN API / SRAM addresses only | Any static demo or local `inference/indoor_env_*` |
| MFCC-on-Arm as in design note | **Not in public MSDK** — would be custom firmware |

---

## 7. Summary

- **Yes:** Public MAX78002 firmware exists for **Cortex-M4F preprocessing (or capture/pack) → CNN accelerator → M4F post-processing**.
- **Best match for a custom 1D prep pipeline:** **`kws20_demo`** (live mic mode).
- **Also useful:** **`pascalvoc-retinanetv7_3`** (camera + timing), **`facial_recognition`** (multi-CNN + camera).
- **No:** Published **MFCC-on-Arm** KWS firmware; shipping KWS uses **int8 audio windows** directly.
- **Most MAX78002 CNN examples** are **static `sampledata.h`** inference demos — same as pre-baked indoor inference today.

---

*Survey date: MSDK `main` branch; links valid as of documentation pass.*

**Community / third-party projects:** [community_m4f_cnn_examples.md](./community_m4f_cnn_examples.md)
