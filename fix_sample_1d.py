
#!/usr/bin/env python3
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Fix IndoorEnvironment 1D sample to expected (int64, CHW=(2,101))')
parser.add_argument('--in', dest='inp', default='tests/sample_indoorenvironment_1d.npy', help='Input .npy path')
parser.add_argument('--out', dest='out', default=None, help='Output .npy path (default: overwrite input)')
parser.add_argument('--assume-float-0-1', action='store_true', help='If input is float in [0,1], map to hardware-like range via (x-0.5)*256 and round (kept as int64)')
args = parser.parse_args()

x = np.load(args.inp)
orig_shape, orig_dtype = x.shape, x.dtype

# If data is (101,2), transpose to (2,101)
if x.ndim == 2 and x.shape == (101, 2):
    x = x.T

# If data has a trailing singleton channel, squeeze
x = np.squeeze(x)

# Validate 2D
if x.ndim != 2:
    raise ValueError(f'Expected 2D array, got shape {x.shape}')

# Handle floats if present
if np.issubdtype(x.dtype, np.floating):
    if args.assume_float_0_1:
        x = np.round((x - 0.5) * 256.0)
    else:
        max_abs = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0
        x = np.round(x / max_abs * 127.0)

# Ensure integer type int64 as expected by ai8xize
x = x.astype(np.int64)

# Ensure final shape is (2,101)
if x.shape == (101, 2):
    x = x.T
elif x.shape != (2, 101):
    raise ValueError(f'Unexpected shape after processing: {x.shape} (expected (2,101))')

out_path = args.out or args.inp
np.save(out_path, x)
print('Fixed sample saved.')
print('Original:', orig_shape, orig_dtype)
print('Final   :', x.shape, x.dtype, 'range:', int(x.min()), int(x.max()))
