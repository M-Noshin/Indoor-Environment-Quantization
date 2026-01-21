#!/usr/bin/env python3
"""
Simple PTQ Calibration Script
Replicates ai8x.pre_qat() for pure PTQ (no QAT training)
"""
import argparse
import torch
import ai8x

def main():
    parser = argparse.ArgumentParser(description='PTQ Calibration (no training)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (e.g., best.pth.tar)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save calibrated checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., IndoorEnvironment_1D)')
    parser.add_argument('--data', type=str, default='data',
                        help='Path to dataset')
    parser.add_argument('--device', type=str, default='MAX78002',
                        help='Target device')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for calibration')
    parser.add_argument('--z-score', type=float, default=2.0,
                        help='Outlier removal z-score (default: 2.0, same as QAT)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model architecture (if different from checkpoint)')
    parser.add_argument('--input-1d-length', type=int, default=101,
                        help='Input 1D length')
    parser.add_argument('--calib-split', type=str, default='trainval',
                        choices=['train', 'trainval'],
                        help='Calibration data to use. IndoorEnvironment_1D "train" dataset '
                             'contains train+val internally; choose "train" to exclude val '
                             '(apples-to-apples with QAT), or "trainval" to include val '
                             '(often improves PTQ calibration stability).')
    args = parser.parse_args()

    # Configure device
    device_dict = {'MAX78000': 85, 'MAX78002': 87}
    device_id = device_dict.get(args.device, 87)
    ai8x.set_device(device_id, simulate=False, round_avg=False, verbose=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get model architecture from checkpoint (or override)
    model_name = args.model if args.model else checkpoint['arch']
    print(f"Creating model: {model_name}")
    
    # Dynamically load all models to find the right module (same as train.py)
    import os
    import fnmatch
    from pydoc import locate
    supported_models = []
    for _, _, files in sorted(os.walk('models')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                fn = 'models.' + name[:-3]
                m = locate(fn)
                try:
                    for i in m.models:
                        i['module'] = fn
                    supported_models += m.models
                except AttributeError:
                    pass
    
    # Find the model
    module = next((item for item in supported_models if item['name'] == model_name), None)
    if not module:
        raise RuntimeError(f"Model {model_name} not found")
    
    # Import the model function
    Model = locate(module['module'] + '.' + model_name)
    if not Model:
        raise RuntimeError(f"Model function {model_name} not found in {module['module']}")
    
    # Create model instance WITHOUT quantization parameters (same as train.py when act_mode_8bit=False)
    # initiate_qat will set these later
    model = Model(pretrained=False, num_classes=4, num_channels=2, 
                  dimensions=(args.input_1d_length, 1), bias=True,
                  weight_bits=None, bias_bits=None, quantize_activation=False)
    
    # Load weights from float checkpoint (strict=True since model structure matches exactly)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    # Create dummy args for stat_collect (it expects args.device)
    class CalibArgs:
        def __init__(self, device):
            self.device = device
            # IMPORTANT:
            # QAT's pre_qat() calibration in train.py runs with simulate=False (no "-8"),
            # so inputs are in the normal (float) range. Matching that here is critical
            # for apples-to-apples thresholds and output_shift.
            self.act_mode_8bit = False
            self.input_1d_length = args.input_1d_length
    
    calib_args = CalibArgs('cpu')  # Use CPU for calibration

    # Load dataset (same as train.py)
    print(f"Loading dataset: {args.dataset} from {args.data}")
    
    # Dynamically load datasets
    supported_sources = []
    for _, _, files in sorted(os.walk('datasets')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                ds = locate('datasets.' + name[:-3])
                try:
                    supported_sources += ds.datasets
                except AttributeError:
                    pass
    
    # Find the dataset
    selected_source = next((item for item in supported_sources if item['name'] == args.dataset), None)
    if not selected_source:
        raise RuntimeError(f"Dataset {args.dataset} not found")
    
    # Load the dataset using its loader function
    train_dataset, _ = selected_source['loader']((args.data, calib_args))

    # IndoorEnvironment_1D returns a *combined* train+val dataset when train=True and
    # exposes the validation split via `valid_indices` (see datasets/indoor_environment_1D.py).
    # PTQ calibration can be quite sensitive to calibration-set size; including val often helps.
    if args.calib_split == 'train' and hasattr(train_dataset, 'valid_indices') and train_dataset.valid_indices:
        # In this dataset implementation, valid_indices are a contiguous range at the end.
        train_len = min(train_dataset.valid_indices)
        if train_len > 0:
            from torch.utils.data import Subset
            train_dataset = Subset(train_dataset, list(range(train_len)))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # === PTQ Calibration (EXACT same order as QAT in train.py:615-638) ===
    print(f"Calibrating with z-score={args.z_score}...")
    
    # Step 1: Fuse BatchNorm (train.py:620)
    print("Fusing BatchNorm layers...")
    ai8x.fuse_bn_layers(model)
    
    # Step 2: Calibrate via pre_qat (train.py:624)
    # This calls: init_hist -> stat_collect -> init_threshold -> release_hist -> apply_scales
    print("Collecting statistics (pre_qat: init_hist, stat_collect, init_threshold, apply_scales)...")
    qat_policy_for_calib = {
        'weight_bits': 8,
        'outlier_removal_z_score': args.z_score
    }
    ai8x.pre_qat(model, train_loader, calib_args, qat_policy_for_calib)
    
    # Step 3: Initialize QAT (train.py:638)
    # This sets quantize_activation=True, adjust_output_shift=True, etc.
    print("Initializing quantization...")
    qat_policy = {
        'weight_bits': 8,
        'outlier_removal_z_score': args.z_score,
        'overrides': {
            'conv1': {'weight_bits': 8},
            'conv2': {'weight_bits': 8},
            'fc1': {'weight_bits': 8},
            'fc2': {'weight_bits': 8}
        }
    }
    ai8x.initiate_qat(model, qat_policy)
    
    print("✓ Calibration complete (activation_threshold and final_scale set by pre_qat)!")
    
    # Update checkpoint with calibrated model
    checkpoint['state_dict'] = model.state_dict()
    
    # Save calibrated checkpoint
    torch.save(checkpoint, args.output)
    print(f"Calibrated checkpoint saved to: {args.output}")
    
    # Verify BN fusion
    bn_keys = [k for k in checkpoint['state_dict'].keys() if '.bn.' in k]
    print(f"BN keys remaining: {len(bn_keys)} (should be 0 after fusing)")
    
    # Check key calibration values
    activation_threshold_keys = [k for k in checkpoint['state_dict'].keys() if 'activation_threshold' in k]
    final_scale_keys = [k for k in checkpoint['state_dict'].keys() if 'final_scale' in k]
    output_shift_keys = [k for k in checkpoint['state_dict'].keys() if 'output_shift' in k and not 'adjust' in k]
    
    if activation_threshold_keys:
        print(f"\nActivation Thresholds (set by pre_qat/init_threshold):")
        for k in sorted(activation_threshold_keys):
            val = checkpoint['state_dict'][k]
            if hasattr(val, 'item'):
                print(f"  {k}: {val.item():.1f}")
    
    if final_scale_keys:
        print(f"\nFinal Scales (set by pre_qat/apply_scales):")
        for k in sorted(final_scale_keys):
            val = checkpoint['state_dict'][k]
            if hasattr(val, 'item'):
                print(f"  {k}: {val.item():.1f}")
    
    if output_shift_keys:
        print(f"\nOutput Shifts (not used - computed dynamically with adjust_output_shift=True):")
        for k in sorted(output_shift_keys):
            val = checkpoint['state_dict'][k]
            if hasattr(val, 'item'):
                print(f"  {k}: {val.item():.1f}")

if __name__ == '__main__':
    main()
