# Sub-Millisecond, Microjoule Edge Inference for Indoor Environment Classification via Layer-Wise Mixed-Precision Quantization

_Hamza A. Abushahla, Muhammed Noshin, Dr. Mohamed I. AlHajri, and Dr. Nazar T. Ali_ 

This repository contains code and resources for the paper: [Sub-Millisecond, Microjoule Edge Inference for Indoor Environment Classification via Layer-Wise Mixed-Precision Quantization](https://ieeexplore.ieee.org/document/11048877).

<div align="center">
  <img src="figs/Indoor.jpg" height="350px" alt="E2E" />
</div>
<p align="center"><em>Figure 1: System overview of the proposed indoor environment identification framework. The pipeline illustrates the complete workflow, including data preprocessing, model training and quantization, and deployment on the MAX78002.</em></p>

## 📌 Overview

This work presents a **hardware-aware framework that integrates Quantization-Aware Training (QAT) with Layer-Wise Mixed-Precision Quantization (MPQ)** to enable sub-millisecond, microjoule inference for indoor environment classification on the **MAX78002 microcontroller**.  
The main contributions of this work are summarized as follows:

- We redesign the CNN from [^1] into a **hardware-aware architecture** optimized for the MAX78002, enabling efficient low-precision inference under tight on-chip memory and energy constraints.

- Using **QAT**, we systematically explore **layer-wise precision assignments** across multiple input bandwidths, showing that MPQ consistently outperforms uniform INT8 quantization in the accuracy–efficiency trade-off.

- The **most-optimal MPQ configuration** achieves a **77.12% smaller model size**, **9.89% faster inference**, and **21.88% lower inference energy** than the uniform INT8 baseline, while maintaining a high **99.22% accuracy**.

- **Real-time deployment** on the MAX78002 achieves **127.7 µs latency** and **27 µJ energy per inference**, with MPQ further reducing **weight-loading time and energy by 85.15% and 56.14%**, respectively.

- **More compact MPQ configurations** achieve **latencies as low as 75.5 µs** and **15.6 µJ per inference**, corresponding to **reductions of 86.95% in model size, 46.74% in inference time, 54.85% in inference energy, 91.54% in weight-loading time,** and **74.78% in weight-loading energy**, offering flexible trade-offs under a relaxed **98% accuracy** requirement.

- **Different clocking modes** are evaluated on the MAX78002, revealing distinct energy–latency trade-offs and showing that MPQ maintains efficiency across frequencies.

- **Deployment insights** are provided for managing frequent model switching in multi-DNN deployments, aligning quantization, memory, and clocking modes with application constraints.

[^1]: https://ieeexplore.ieee.org/abstract/document/11021689

---

## 📁 Repository Structure

This repository is organized as **overlays** on top of the official Analog Devices ai8x toolchain:

- `envs/`
  - `max_linux.yml` / `max_mac.yml`: recommended conda environments for reproducibility
  - `requirements.txt`: fallback pip requirements (Python 3.11)

- `training/`
  Overlay files for `ai8x-training`:
  - dataset (`data/`)
  - dataloader (`datasets/`)
  - model definitions (`models/`)
  - training policies (LR schedule + QAT/MPQ policies) (`policies/`)
  - scripts and sweep drivers (`scripts/`, `sweeps/`)

- `synthesis/`
  Overlay files for `ai8x-synthesis`:
  - izer configs / network YAMLs
  - generation scripts for hardware projects

- `inference/`
  Example exported projects and ready-to-run MAX78002 deployments:
  - generated izer project folders (e.g., `indoor_env_1d_51_q8824/`)
  - example quantized checkpoints (when applicable)

- `figs/`
  Figures used in this repo/README

---

## 1) Environment Setup

### Option A (Recommended): Conda environment
From the repo root:
```bash
conda env create -f envs/max_linux.yml
# or
conda env create -f envs/max_mac.yml

conda activate max
````

### Option B: Pip environment (fallback)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r envs/requirements.txt
```

---

## 2) Create a Working Directory (ai8x root)

We recommend a single workspace that contains both toolchains:

```bash
mkdir max_workdir && cd max_workdir
git clone --recursive https://github.com/analogdevicesinc/ai8x-training.git
git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git
```

Expected layout:

```text
max_workdir/
├── ai8x-training/
└── ai8x-synthesis/
```

---

## 3) Copy This Repo’s Overlay Files Into ai8x

### 3.1 ai8x-training overlay

Copy the contents of this repo’s `training/` into your local `ai8x-training/` (merge folders; do not nest).

Example mapping (will expand as repo evolves):

| Target folder (inside `ai8x-training/`) | Copy from (this repo)                   | Purpose                        |
| --------------------------------------- | --------------------------------------- | ------------------------------ |
| `data/`                                 | `training/data/indoor_environment/`     | Dataset (.mat files)           |
| `datasets/`                             | `training/datasets/`                    | Dataloader(s)                  |
| `models/`                               | `training/models/`                      | ai8x model(s)                  |
| `policies/`                             | `training/policies/`                    | LR schedule + QAT/MPQ policies |
| `scripts/` / `sweeps/`                  | `training/scripts/`, `training/sweeps/` | training/eval + sweep drivers  |

### 3.2 ai8x-synthesis overlay

Similarly, merge this repo’s `synthesis/` into your local `ai8x-synthesis/`.

---

## 4) Simulation Pipeline (Training → Quantization → Evaluation)

All commands below assume you are inside:

```bash
cd max_workdir/ai8x-training
```

### 4.1 Single-run training (QAT optional)

**QAT-enabled run:**

```bash
python train.py --epochs 10 --batch-size 256 \
  --optimizer Adam --lr 0.001 --weight-decay 0.0005 \
  --use-bias --deterministic \
  --model ai85indoorenvnetv2 --dataset IndoorEnvironment_1D --data data/indoor_environment \
  --compress policies/schedule-indoor-env.yaml \
  --qat-policy policies/qat_policy_indoor_v2.yaml \
  --input-1d-length 101 \
  --device MAX78002 --name indoor_run
```

**PTQ-only (no QAT):**
Set `--qat-policy` to `None` (or remove it, depending on your local script conventions).

**Outputs (logs directory):**

* `checkpoint.pth.tar`, `best.pth.tar`: float checkpoints
* `qat_checkpoint.pth.tar`, `qat_best.pth.tar`: checkpoints after QAT starts
  Note: these are not yet “izer-ready” until quantization/export steps are run by the provided scripts.

**MPQ configuration:**
Layer-wise bitwidths are controlled in the relevant QAT policy YAML (e.g., `qat_policy_indoor_v2.yaml`).

---

## 5) Automated Sweeps Used in the Paper

### 5.1 QAT Mixed-Precision Sweep

Script: `train_indoor_1D_mixed_sweep.py`

What it does:

* Enumerates **all 891 MPQ configs** (3^4 over {8,4,2} bits for conv1/conv2/fc1/fc2)
* Sweeps **multiple input lengths (α)** and **multiple seeds**
* Runs **full QAT**, then quantizes and evaluates each run
* Writes detailed and aggregated CSV summaries

Run:

```bash
python train_indoor_1D_mixed_sweep.py
```

Default output folder (example):

```text
ai8x_seed_runs_out/
├── logs_mixed/
├── checkpoints_mixed/
├── policies/sweep/
├── mixed_precision_sweep_results.csv
└── mixed_precision_sweep_summary.csv
```

### 5.2 PTQ Mixed-Precision Sweep

Script: `train_indoor_1D_mixed_sweep_ptq.py`

Run (example):

```bash
python -u train_indoor_1D_mixed_sweep_ptq.py \
  --num-seeds 5 \
  --start-seed 42 \
  --input-lengths 101 \
  --epochs 10 \
  --z-score 2.0 \
  --calib-split train
```

Outputs (example):

```text
ai8x_ptq_sweep_out/
├── ptq_sweep_results.csv
├── ptq_sweep_summary.csv
├── logs_ptq/
└── checkpoints_ptq/
```

---

## 6) Hardware Evaluation (Quantized Checkpoint → Synthesis → MAX78002 Deployment)

### 6.1 Quantized checkpoint naming

Quantized checkpoints follow:

* Uniform: `_q8.pth.tar`, `_q4.pth.tar`, `_q2.pth.tar`
* Mixed precision: `_qmixed.pth.tar`

Example:

```text
indoor_mixed_seed_46__L101__8_8_8_8_*_qat_best_q8.pth.tar
```

### 6.2 Synthesis (ai8x-synthesis / ai8xize)

Go to:

```bash
cd ../ai8x-synthesis
```

Edit the generation script (example):

* `scripts/gen_indoor_1d.sh`

Set:

* `LENGTH=101`
* `CONFIG="8-8-8-8"` (or any MPQ config like `8-8-2-4`)
* `CHECKPOINT=<path-to-quantized-checkpoint>`

The script calls `ai8xize.py` to generate a deployable C project, e.g.:

```bash
python ai8xize.py \
  --test-dir "$TARGET" \
  --prefix "$PREFIX" \
  --checkpoint-file "$CHECKPOINT" \
  --config-file networks/indoorenvnet-v2-chw-${LENGTH}.yaml \
  --sample-input tests/sample_indoorenvironment_1d_${LENGTH}.npy \
  --overwrite --softmax --compact-data --mexpress --max-speed --energy
```

### 6.3 Generated project output

A C/C++ project folder will be created (example):

```text
HW_Evaluation/indoor_env_1d_101_q8888/
```

This includes:

* `main.c` (entry point)
* `cnn.c`, `cnn.h` (generated network)
* Makefile / Eclipse launch files (depending on izer output)

### 6.4 Flash/run on MAX78002 (MSDK + Eclipse)

Import the generated project into Eclipse after setting up the Analog Devices MSDK:

* [https://github.com/analogdevicesinc/msdk](https://github.com/analogdevicesinc/msdk)

For paper-style evaluation, we modify `main.c` to report:

* inference latency
* energy/power measurements (as used in our evaluation scripts)

### 6.5 Provided examples

This repo includes example exported projects and checkpoints under:

* `inference/`

Example:

* `inference/indoor_env_1d_51_q8824/`
* `inference/indoor_env_1d_91_q8824/`
* `inference/indoor_mixed_seed_46__L51__8_8_2_4_indoor_mixed_seed_46__L51__8_8_2_4_qat_best_qmixed.pth.tar`
* `indoor_mixed_seed_46__L91__8_8_2_4_indoor_mixed_seed_46__L91__8_8_2_4_qat_best_qmixed.pth.tar`

---

## Citation & Reaching out
If you use our work for your own research, please cite us with the below: 

```bibtex

```

You can also reach out through email to: 
- Hamza Abushahla - b00090279@alumni.aus.edu
- Dr. Mohamed AlHajri - mialhajri@aus.edu