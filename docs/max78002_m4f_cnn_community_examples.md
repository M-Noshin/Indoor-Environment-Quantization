# Community & Non-MSDK Examples: M4F Preprocessing → CNN Accelerator

Companion to [max78002_m4f_cnn_examples.md](./max78002_m4f_cnn_examples.md) (ADI MSDK only). This survey lists **third-party**, **forked**, and **reference-design** projects that use the same split: **general-purpose core prepares data → hardware CNN runs inference → core post-processes**.

**Scope:** MAX78000 / MAX78002 and close analogs. Not exhaustive; discovered via GitHub search, articles, and issue threads (2024–2026).

---

## 1. Takeaway

| Finding | Detail |
|---------|--------|
| **Few standalone “MFCC-on-M4” repos** | Same as MSDK: public KWS paths rarely implement full MFCC on-device. |
| **Many projects extend MSDK demos** | Fork `kws20_demo`, add UART/GPIO, or swap models — still M4F prep + `cnn.c` pattern. |
| **Best non-MSDK analogs for 1D spectra** | **SPECTROLUX** (1D CNN, 288 spectral bins, apples). |
| **Best MAX78002 audio pipeline write-up** | **clooey/MAX78002_thumbs_up** (I2S ring buffer → CNN, documented stages). |
| **Edge Impulse** | No widely used, maintained **native MAX78002** deploy target found; ADI flow remains ai8x + MSDK. |

---

## 2. Closest to indoor / spectral 1D work

### SPECTROLUX — spectral 1D CNN on MAX78000

| | |
|--|--|
| **Repo** | [ferielboudjatit/SPECTROLUX](https://github.com/ferielboudjatit/SPECTROLUX---Embedded-Spectral-Analysis-and-Classification-System-of-Varieties-of-Apples) |
| **Idea** | Hamamatsu spectrometer (288 bands, 340–850 nm) → **1D CNN** on MAX78000 for apple variety classification. |
| **Architecture** | **STM32F303** acquires spectra and talks to PC / **MAX78000**; MAX runs embedded model. |
| **Firmware** | `MSDKproject/ai8x-synthesis/demos/ai8x-apple_discrimination/` — izer output: **288×1** int8 input, `load_input()` → `cnn_start()` → `softmax`. |
| **M4F prep today** | Shipped `main.c` still loads **`sampledata.h`** (`memcpy32` to `0x50400000`); live spectrometer→tensor path is described in README but not a clean standalone “prep library” in repo. |
| **Relevance** | **Very high** — 1D spectral classification on MAX78000, same toolchain as this repo. |

---

### Hand gesture recognition (EIT → MAX78000)

| | |
|--|--|
| **Repo** | [MaroueneKaaniche/CNN_Implementation_on_hardware_accelerator_for_hand_gesture_recognition](https://github.com/MaroueneKaaniche/CNN_Implementation_on_hardware_accelerator_for_hand_gesture_recognition) |
| **Idea** | EIT sensor → serial → **MAX78000 FTHR**; custom PyTorch model + ai8x train/synthesize. |
| **M4F prep** | Custom **data loader** on PC for range/transform; on-device: standard synthesized `load_input` / CNN flow (see repo README). |
| **Relevance** | Medium — shows **custom 1D/sensor** model port, not RF/CTF. |

---

## 3. Vision: camera / capture on M4F → CNN

### Fruit classifier (MAX78000 FTHR + TFT)

| | |
|--|--|
| **Repo** | [hehung/MAX78000_fruit_cnn](https://github.com/hehung/MAX78000_fruit_cnn) (~14★) |
| **Idea** | Live **camera** → CNN → 11-class fruit IDs on TFT. |
| **Flow** | Train/quantize (PyTorch) → ai8x synthesis → deploy with **camera + LCD** code in `fruit_cnn` / `fruit_cnn_simple`. |
| **Relevance** | Good **end-to-end student/project** template for live sensor + display. |

### Cat feeder / SSD detection

| | |
|--|--|
| **Repo** | [xzqiaochu/cat-demo](https://github.com/xzqiaochu/cat-demo) (~16★) |
| **Idea** | MAX78000 **SSD** cat detection + motor; `main.c` uses **`camera.h`**, `load_input()`, `cnn.c`. |
| **M4F prep** | Camera capture path + optional static `memcpy32` test input. |
| **Relevance** | Detection + actuator; heavier than 1D CTF. |

### MSDK issue — CIFAR-10 + onboard camera (community recipe)

| | |
|--|--|
| **Link** | [msdk#1061 — CIFAR-10 FTHR camera mod](https://github.com/analogdevicesinc/msdk/issues/1061) |
| **Idea** | Line-by-line **camera DMA streaming**: `get_camera_stream_buffer()`, RGB → CNN range **[-128,127]**, `release_camera_stream_buffer()`, then `cnn_load_input` / `cnn_start`. |
| **Relevance** | Official issue thread with **copy-pasteable preprocessing loop** (not a separate repo). |

### ADI camera streaming guide

| | |
|--|--|
| **Link** | [MaximAI_Documentation — Camera Streaming Guide](https://github.com/analogdevicesinc/MaximAI_Documentation/blob/main/Guides/Camera_Streaming_Guide.md) |
| **Note** | ADI doc, but the **pattern** (M4 streams pixels, CNN uses FIFO) is what community camera mods follow. |

---

## 4. Audio: M4F buffering / scaling → CNN

### CoughNet (KWS fork)

| | |
|--|--|
| **Repo** | [mhbenabda/CoughNet](https://github.com/mhbenabda/CoughNet) |
| **Idea** | Cough detection on MAX78000; includes modified **`kws20_demo`** + training notebooks. |
| **M4F prep** | Based on ADI KWS mic path (scale, buffer, CNN) — see bundled `kws20_demo/`. |

### Elektor / magazine projects

| | |
|--|--|
| **Repos** | [ClemensAtElektor/MAX78000](https://github.com/ClemensAtElektor/MAX78000) (zips: `kws20_demo_cpv.zip`, cats-dogs), [ElektorLabs/210162-MAX78000](https://github.com/ElektorLabs/210162-MAX78000) (`kws20_demo/` tree) |
| **Idea** | GPIO/UART extensions on top of ADI KWS (e.g. JSON keyword + confidence on UART2). |
| **Relevance** | **Integration** examples, not new prep math. |

### MAX78002 speech / audio pipeline (documented)

| | |
|--|--|
| **Repo** | [clooey/MAX78002_thumbs_up](https://github.com/clooey/MAX78002_thumbs_up) |
| **Idea** | **MAX78002**: I2S in → ring buffer / history → **CNN** → I2S out; stages for passthrough and speech-separation models. |
| **Docs** | `HARDWARE_IMPLEMENTATION.md`, `READMEv4.0pipeline78002quasiworks.md` — chunk size 128, Q7, frame timing. |
| **Note** | Doc title mentions FTHR in places; repo targets **78002** pipeline design. |
| **Relevance** | **High for MAX78002** — clearest **non-MSDK** write-up of continuous M4F audio prep + accelerator. |

---

## 5. Reference design firmware (ADI, not under `msdk/Examples`)

### MAXREFDES178 (“AI cube camera”)

| | |
|--|--|
| **Repo** | [analogdevicesinc/MAX78xxx-RefDes](https://github.com/analogdevicesinc/MAX78xxx-RefDes) |
| **Hardware** | **MAX32666** (Bluetooth/UI) + **two MAX78000** (video + audio). |
| **Apps** | `maxrefdes178-CatsDogs`, `FacialRecognition`, `DigitDetection`, `UNet`, `ImageCapture`, etc. |
| **M4F prep** | Wiki/doc: edit `max78000_video_main.c` / `max78000_audio_main.c` — **peripheral init, preprocessing**, then `load_input()` / `cnn_start()` (FIFO order documented). |
| **Relevance** | Production-style **multi-processor** split; firmware is separate from MSDK example tree but same CNN API. |

---

## 6. Training-only / PC prep (not on-device M4F)

These use the ai8x flow but do **not** ship notable on-M4F live prep repos:

| Repo | Notes |
|------|--------|
| [InES-HPMM/MAX7800x-Jupyter-training](https://github.com/InES-HPMM/MAX7800x-Jupyter-training) | Jupyter MNIST/QAT; still needs ai8x-synthesis for device C. |
| [geffencooper/ai8x-synthesis_ECE196](https://github.com/geffencooper/ai8x-synthesis_ECE196) | Course fork of synthesis. |
| [aniktash/MAX78000_SDK-1](https://github.com/aniktash/MAX78000_SDK-1) | Old MSDK snapshot, not a new prep pattern. |
| [Elios-Lab/ai8x-tsd](https://github.com/Elios-Lab/ai8x-tsd) | Training/synthesis helpers. |

---

## 7. Analogous non-ADI platforms (same *idea*, different silicon)

Useful when citing “M4 + NPU” in related work; **not** MAX78002 code.

| Platform | Example | M4/cluster prep → accelerator |
|----------|---------|--------------------------------|
| **GreenWaves GAP8** | [keyword_spotting](https://github.com/GreenWaves-Technologies/keyword_spotting) | **Quantized MFCC on MCU** + CNN on cluster (Autotiler / nntool). |
| **GreenWaves GAP8** | [face_detection](https://github.com/GreenWaves-Technologies/face_detection), [MobilenetV1_Pytorch](https://github.com/GreenWaves-Technologies/MobilenetV1_Pytorch) | Image/sensor → cluster CNN. |

ADI’s public KWS choice (skip on-device MFCC, feed int8 time-domain windows to CNN) aligns with papers noting MFCC cost on the host MCU (e.g. [arXiv:2111.04988](https://arxiv.org/abs/2111.04988)).

---

## 8. Edge Impulse / TFLite Micro

| Tool | MAX78002 status (survey) |
|------|---------------------------|
| **Edge Impulse** | Generic Cortex-M4F targets exist; **no standard, documented MAX78002 NPU integration** comparable to ai8x+MSDK. |
| **TFLite Micro on M4 only** | Would **not** use the CNN accelerator — different deployment class. |

For MAX78002, third-party work almost always stays on **ai8x-training → ai8x-synthesis → MSDK-style `cnn.c`**.

---

## 9. Mapping to this repo (indoor CTF)

| External example | What to borrow |
|------------------|----------------|
| **kws20_demo** (MSDK or CoughNet/Elektor fork) | Mic buffering, int8 scaling, `AddTranspose`, `cnn_start` timing |
| **clooey/MAX78002_thumbs_up** | Documented **chunked** real-time pipeline on 78002 |
| **SPECTROLUX** | 1D spectral CNN sizing (288 bins), apple discrimination synthesis layout |
| **msdk#1061** | If you ever use camera/FIFO streaming |
| **MAXREFDES178** | Multi-app product structure, `load_input` / FIFO ordering notes |

**Gap:** No public repo found for **VNA/CTF live prep on MAX78002 M4F** — same gap as MFCC-on-Arm in ADI’s KWS article.

---

## 10. Quick reference table

| Project | MAX part | Live M4F prep? | Domain | Link |
|---------|----------|----------------|--------|------|
| SPECTROLUX | 78000 | Partial / STM32+sampledata | 1D spectra | [GitHub](https://github.com/ferielboudjatit/SPECTROLUX---Embedded-Spectral-Analysis-and-Classification-System-of-Varieties-of-Apples) |
| MAX78002_thumbs_up | 78002 | Yes (I2S pipeline) | Audio | [GitHub](https://github.com/clooey/MAX78002_thumbs_up) |
| fruit_cnn | 78000 | Yes (camera) | Vision | [GitHub](https://github.com/hehung/MAX78000_fruit_cnn) |
| cat-demo | 78000 | Yes (camera) | SSD | [GitHub](https://github.com/xzqiaochu/cat-demo) |
| CoughNet | 78000 | Via kws20 fork | Audio | [GitHub](https://github.com/mhbenabda/CoughNet) |
| EIT gestures | 78000 | Serial + synthesized | EIT | [GitHub](https://github.com/MaroueneKaaniche/CNN_Implementation_on_hardware_accelerator_for_hand_gesture_recognition) |
| MAXREFDES178 | 78000×2 | Yes (video/audio mains) | Multi-app | [GitHub](https://github.com/analogdevicesinc/MAX78xxx-RefDes) |
| Elektor KWS mods | 78000 | Yes (ADI base) | Audio | [GitHub](https://github.com/ElektorLabs/210162-MAX78000) |
| GAP8 KWS | GAP8 | Yes (MFCC on MCU) | Audio | [GitHub](https://github.com/GreenWaves-Technologies/keyword_spotting) |

---

*See also: [e2e_latency_measurement.md](./e2e_latency_measurement.md).*
