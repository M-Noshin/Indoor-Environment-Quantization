# Indoor-Environment-Quantization

## 📦 Setup and Training

1️⃣ **Clone and install [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training)**

```bash
git clone https://github.com/MaximIntegratedAI/ai8x-training.git
cd ai8x-training
# Follow their install instructions (e.g., create conda env, install requirements)
```

2️⃣ **Copy project files into `ai8x-training`:**

| Target folder              | Copy from this repo                               | Purpose                      |
|---------------------------|--------------------------------------------------|-----------------------------|
| `./data/`     | `data/indoor_environment/` folder (contains 8 `.mat` files) | Dataset                   |
| `./datasets/` | `datasets/indoor_environment.py`                          | Data loader script         |
| `./models/`   | `models/ai85net_indoor_env_v1.py`                       | Model using ai8x layers    |
| `./policies/` | `policies/schedule-indoor-env.yaml`, `policies/qat_policy_indoor.yaml` | LR schedule, QAT config |
| `./scripts/`  | `scripts/train_indoor.sh`, `scripts/evaluate_indoor.sh`         | Training/eval scripts      |

3️⃣ **Train the model:**

```bash
(ai8x-training) $ sh scripts/train_indoor.sh
```

Edit `train_indoor.sh` to adjust epochs, batch size, etc.

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

This will generate `indoor_run_qat_best.pth_q8.tar`, the INT8 quantized model.

---

## ✅ Evaluation

1️⃣ Edit `scripts/evaluate_indoor.sh` to point to the quantized model.

2️⃣ Run evaluation:

```bash
(ai8x-training) $ sh scripts/evaluate_indoor.sh
```

This evaluates the quantized model (prepared for MAX78000 deployment).



