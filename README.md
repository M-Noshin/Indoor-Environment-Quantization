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

## 📁 Repository Structure

* `training/`
  Files related to [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training), including the dataset, dataloader, model, training, and quantization scripts.

* `synthesis/`
  Files related to [ai8x-synthesis](https://github.com/MaximIntegratedAI/ai8x-synthesis), including model synthesis and deployment on hardware (e.g., MAX78002 configuration YAMLs).


## 📦 Getting Started

### 1️⃣ Install `ai8x-training`

Clone and install the official repository:

```bash
git clone https://github.com/MaximIntegratedAI/ai8x-training.git
cd ai8x-training
# Follow their installation steps (e.g., conda environment, requirements)
```

---

### 2️⃣ Copy project files

Copy the contents of this repo’s `training/` folder into the corresponding folders of your local `ai8x-training` installation:

| Target folder (inside `ai8x-training/`) | Copy from (`training/` in this repo)                                                     | Purpose                                    |
| --------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------ |
| `./data/`                               | `training/data/indoor_environment/` (contains 8 `.mat` files)                            | Dataset                                    |
| `./datasets/`                           | `training/datasets/indoor_environment.py`                                                | Dataloader script                          |
| `./models/`                             | `training/models/ai85net_indoor_env_v1.py`                                               | Indoor-environment model using ai8x layers |
| `./policies/`                           | `training/policies/schedule-indoor-env.yaml`, `training/policies/qat_policy_indoor.yaml` | LR schedule & QAT config                   |
| `./scripts/`                            | `training/scripts/train_indoor.sh`, `training/scripts/evaluate_indoor.sh`                | Training & evaluation scripts              |

---

### 3️⃣ Run training & evaluation

From inside the `ai8x-training` directory:

```bash
# Training
sh scripts/train_indoor.sh
```

Edit `train_indoor.sh` to adjust epochs, batch size, etc.

This will run the training + initiate QAT at the epoch in the policy file (e.g., epoch 8) and will save the best model in a checkpoint in the corresponding logs folder. 
The logs folder will contain `checkpoint.pth.tar` and `best.pth.tar`, these are the floating point models, and `qat_checkpoint.pth.tar` and `qat_best.pth.tar` are the models from QAT. Note these models are not quantized yet. 

---

## ⚙️ Quantization

1️⃣ **Clone and install [ai8x-synthesis](https://github.com/MaximIntegratedAI/ai8x-synthesis)**

Make sure you have both `ai8x-training` and `ai8x-synthesis` in the same root folder.

2️⃣ **Run quantization:**

Assuming training saved a checkpoint, e.g., `logs/indoor_run___2025.07.11-174158/indoor_run_qat_best.pth.tar`:

```bash
python ai8x-synthesis/quantize.py \
  ai8x-training/logs/indoor_run___2025.07.11-174158/indoor_run_qat_best.pth.tar \
  ai8x-training/logs/indoor_run___2025.07.11-174158/indoor_run_qat_best.pth_q8.tar \
  --device MAX78000
```

This will generate `indoor_run_qat_best.pth_q8.tar`, the INT8 quantized model in this case.

---

## ✅ Evaluation

1️⃣ Edit `scripts/evaluate_indoor.sh` to point to the quantized model.

2️⃣ Run evaluation:

```bash
(ai8x-training) $ sh scripts/evaluate_indoor.sh
```

This evaluates the quantized model (prepared for MAX78000 deployment).

## Citation & Reaching out
If you use our work for your own research, please cite us with the below: 

```bibtex

```

You can also reach out through email to: 
- Hamza Abushahla - b00090279@alumni.aus.edu
- Dr. Mohamed AlHajri - mialhajri@aus.edu


