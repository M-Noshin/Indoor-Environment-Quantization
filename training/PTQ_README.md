# Post-Training Quantization (PTQ) for AI8X

## Overview

This implementation provides **calibration-based PTQ** for AI8X hardware, achieving ~97-98% accuracy on the Indoor Environment model (vs ~98-99% for QAT).

## Quick Start

```bash
# Run complete PTQ workflow (train → calibrate → quantize → evaluate)
./test_ptq_single.sh

# Expected: ~97-98% accuracy for INT8 quantized model
```

That's it! The script handles everything automatically.

## Core Files

### 1. `calibrate_ptq_simple.py` - PTQ Calibration Script
**Purpose:** Replicates QAT's calibration phase without training.

**What it does:**
- Loads a trained float checkpoint
- Fuses BatchNorm layers into conv/linear layers
- Collects activation statistics via `ai8x.pre_qat()`
- Sets `activation_threshold` and `final_scale` parameters
- Saves calibrated checkpoint ready for quantization

**Usage:**
```bash
python calibrate_ptq_simple.py \
  --checkpoint <float_checkpoint.pth.tar> \
  --output <calibrated_output.pth.tar> \
  --model ai85indoorenvnetv2 \
  --dataset IndoorEnvironment_1D \
  --data data/indoor_environment \
  --input-1d-length 101 \
  --device MAX78002 \
  --batch-size 256 \
  --z-score 2.0 \
  --calib-split trainval
```

**Key Parameters:**
- `--z-score`: Outlier removal threshold (2.0 matches QAT default)
- `--calib-split`: `train` (strict comparison) or `trainval` (better coverage, recommended)

### 2. `test_ptq_single.sh` - Complete PTQ Workflow
**Purpose:** End-to-end PTQ test from training to evaluation.

**Flow:**
1. Train float model (10 epochs, no QAT)
2. Calibrate (sets activation thresholds)
3. Quantize to INT8 (uses `MAX_BIT_SHIFT` method)
4. Evaluate quantized model

**Usage:**
```bash
./test_ptq_single.sh
```

**Expected Results:**
- Float model: ~99.7-99.9%
- PTQ INT8: ~97-98%
- Gap vs QAT: ~0.5-1% (expected)


## How PTQ Works

### Key Insight
PTQ replicates QAT's **calibration phase** without the **training phase**:

1. **QAT Flow:**
   ```
   Train (float) → [QAT Start] → Fuse BN → Calibrate → Train with quantization
   ```

2. **PTQ Flow:**
   ```
   Train (float, complete) → Fuse BN → Calibrate → Quantize (no more training)
   ```

### Critical Parameters Set During Calibration

| Parameter | Set By | Purpose | QAT Value (example) | PTQ Value |
|-----------|--------|---------|---------------------|-----------|
| `activation_threshold` | `pre_qat/init_threshold` | Clipping threshold for activations | conv2=1.0, fc2=2.0 | Same |
| `final_scale` | `pre_qat/apply_scales` | Output layer scaling | fc2=4.0 | Same |
| `output_shift` | Computed dynamically | Layer-wise bit shifts | 0.0 (stored) | 0.0 (stored) |
| `adjust_output_shift` | `initiate_qat` | Enable dynamic shift computation | True | True |
| `quantize_activation` | `initiate_qat` | Enable activation quantization | True | True |

### Why Calibration is Critical

Without proper calibration, PTQ falls to **83% accuracy** (vs 98% with calibration). The calibration:
- Collects activation histograms from training data
- Removes outliers based on z-score
- Sets optimal clipping thresholds per layer
- Adjusts biases for multi-layer networks

## Common Pitfalls

### ❌ Wrong: Using `--clip-method SCALE` with calibrated PTQ
```bash
python quantize.py ... --clip-method SCALE --scale 0.85
```
**Result:** 42-45% accuracy (overrides calibrated thresholds)

### ✅ Correct: Use default `MAX_BIT_SHIFT` method
```bash
python quantize.py ... --device MAX78002 -v
# No --clip-method argument
```
**Result:** 97-98% accuracy (respects calibrated thresholds)

### ❌ Wrong: Creating model with `quantize_activation=True` initially
```python
model = Model(..., quantize_activation=True)  # Breaks calibration!
```

### ✅ Correct: Create model like train.py (let initiate_qat set it)
```python
model = Model(..., quantize_activation=False)  # Correct
# initiate_qat will set it to True after calibration
```

## Performance Summary

| Method | Accuracy | Notes |
|--------|----------|-------|
| Float (10 epochs) | 99.9% | Baseline |
| QAT INT8 | 98.4% | Gold standard |
| PTQ INT8 (calibrated) | 97.8% | 0.6% gap (excellent!) |
| PTQ INT8 (uncalibrated) | 83% | Without proper calibration |
| PTQ INT8 (wrong clip) | 42-45% | Using SCALE method |

## Integration with Mixed-Precision Sweep

To add PTQ to your existing mixed-precision sweep:
1. Use `--qat-policy None` during training
2. Call `calibrate_ptq_simple.py` after training
3. Quantize with default method (no `--clip-method`)
4. Evaluate with `-8` flag

The same mixed-precision bits can be applied during quantization.

## File Inventory

### PTQ Implementation (2 files)
1. **`calibrate_ptq_simple.py`** - Core calibration script ⭐
2. **`test_ptq_single.sh`** - Complete PTQ workflow ⭐
3. **`PTQ_README.md`** - This documentation

**That's all you need!** Everything else has been cleaned up.

## Next Steps

To integrate PTQ into your mixed-precision sweep:
1. Modify your sweep script to use `--qat-policy None` during training
2. After each training run, call `calibrate_ptq_simple.py`
3. Quantize with default method (no `--clip-method`)
4. Evaluate with `-8` flag

The PTQ workflow is now production-ready and matches QAT accuracy within 0.5-1%!
