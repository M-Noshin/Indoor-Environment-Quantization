# Indoor-Environment-Quantization

_Muhammed Noshin, Hamza A. Abushahla, and Dr. Mohamed I. AlHajri_

This repository contains code and resources for the paper: [Title Here](https://ieeexplore.ieee.org/document/11048877).

## 📌 Overview
This work presents...

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



